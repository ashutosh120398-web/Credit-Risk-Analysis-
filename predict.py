"""Predict loan default risk for one applicant.

Usage (interactive):   python predict.py
Usage (one line):      python predict.py --age 24 --income 30000 --home RENT --emp 2 --intent MEDICAL \
                                         --amount 12000 --prior N --hist 3
Optional (if the lender has already assigned them):  --grade D --rate 15.5
"""
import argparse, os, joblib, pandas as pd
from train_model import design  # noqa  (reuses the exact feature builder used in training)

_B = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'credit_risk_model.joblib'))
HOME, INTENT = _B['cat']['person_home_ownership'], _B['cat']['loan_intent']

LABEL = {'DEBTCONSOLIDATION':'Debt consolidation','HOMEIMPROVEMENT':'Home improvement','MEDICAL':'Medical','EDUCATION':'Education','PERSONAL':'Personal','VENTURE':'Venture'}

def band(p):
    return 'LOW' if p < .15 else 'MEDIUM' if p < .40 else 'HIGH'

def decision(b):
    return {'LOW':'Approve (standard checks)','MEDIUM':'Manual review / ask for more documents','HIGH':'Decline or restructure (smaller loan, collateral, co-signer)'}[b]

def reasons(a):
    r, lti = [], a['loan_amnt'] / a['person_income']
    if lti >= .30: r.append(f'Loan is {lti:.0%} of annual income (30%+ is where default jumps sharply)')
    elif lti >= .20: r.append(f'Loan is {lti:.0%} of annual income (moderately high)')
    if a['person_home_ownership'] in ('RENT','OTHER'): r.append('Renter / non-owner (higher default group)')
    if a['cb_person_default_on_file'] == 'Y': r.append('Prior default on credit record')
    if a['person_income'] < 35000: r.append('Low annual income')
    if a['loan_intent'] in ('DEBTCONSOLIDATION','MEDICAL','HOMEIMPROVEMENT'): r.append(f"Loan purpose ({LABEL[a['loan_intent']]}) has above-average default")
    if a['person_emp_length'] < 2: r.append('Short employment history')
    if a.get('loan_grade') in list('DEFG'): r.append(f"Lender grade {a['loan_grade']} (high-risk grade)")
    if a.get('loan_int_rate', 0) >= 14: r.append(f"Interest rate {a['loan_int_rate']}% (high-risk pricing)")
    if not r:
        r.append('No major risk flags found')
        if a['person_home_ownership'] in ('OWN','MORTGAGE'): r.append('Home owner / mortgage holder (lower default group)')
    return r

def predict(person_age, person_income, person_home_ownership, person_emp_length, loan_intent,
            loan_amnt, cb_person_default_on_file, cb_person_cred_hist_length, loan_grade=None, loan_int_rate=None):
    if person_income <= 0: raise ValueError('Income must be positive')
    if person_home_ownership not in HOME: raise ValueError(f'home must be one of {HOME}')
    if loan_intent not in INTENT: raise ValueError(f'intent must be one of {INTENT}')
    full = loan_grade is not None and loan_int_rate is not None
    a = dict(person_age=person_age, person_income=person_income, person_home_ownership=person_home_ownership,
             person_emp_length=person_emp_length, loan_intent=loan_intent, loan_amnt=loan_amnt,
             cb_person_default_on_file=cb_person_default_on_file, cb_person_cred_hist_length=cb_person_cred_hist_length,
             loan_percent_income=loan_amnt / person_income)
    if full: a.update(loan_grade=loan_grade, loan_int_rate=loan_int_rate)
    X = design(pd.DataFrame([a]), full).reindex(columns=_B['full_cols' if full else 'applicant_cols'], fill_value=0)
    p = min(max(float(_B['full' if full else 'applicant'].predict_proba(X)[0, 1]), .01), .99)   # never claim certainty
    b = band(p)
    return dict(probability=p, risk_band=b, decision=decision(b), model='full (with grade & rate)' if full else 'applicant-only',
                loan_to_income=a['loan_percent_income'], reasons=reasons(a))

def show(r):
    print(f"\nDefault probability : {r['probability']:.1%}\nRisk band           : {r['risk_band']}\nSuggested action    : {r['decision']}\nModel used          : {r['model']}\nKey factors:")
    for x in r['reasons']: print('  -', x)

def ask(q, cast=str, choices=None, default=None):
    while True:
        s = input(f"{q}{' '+str(choices) if choices else ''}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s and default is not None: return default
        try:
            v = cast(s.upper() if choices else s)
            if choices and v not in choices: raise ValueError
            return v
        except ValueError: print('  Invalid, try again.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, t in [('age',int),('income',float),('home',str),('emp',float),('intent',str),('amount',float),('prior',str),('hist',int),('grade',str),('rate',float)]:
        ap.add_argument('--'+k, type=t)
    g = ap.parse_args()
    if g.age is None:
        print('--- Loan default risk check ---')
        args = dict(person_age=ask('Age',int), person_income=ask('Annual income',float), person_home_ownership=ask('Home ownership',str,HOME),
                    person_emp_length=ask('Years employed',float), loan_intent=ask('Loan purpose',str,INTENT), loan_amnt=ask('Loan amount',float),
                    cb_person_default_on_file=ask('Any prior default?',str,['N','Y']), cb_person_cred_hist_length=ask('Credit history (years)',int))
    else:
        args = dict(person_age=g.age, person_income=g.income, person_home_ownership=g.home.upper(), person_emp_length=g.emp, loan_intent=g.intent.upper(),
                    loan_amnt=g.amount, cb_person_default_on_file=g.prior.upper(), cb_person_cred_hist_length=g.hist,
                    loan_grade=g.grade.upper() if g.grade else None, loan_int_rate=g.rate)
    show(predict(**args))
