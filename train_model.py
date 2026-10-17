"""Train the credit-risk prediction models and save them to model/credit_risk_model.joblib

Two models are trained on the cleaned Kaggle data:
  * applicant model - uses only what an applicant can tell you (age, income, home, job length,
                      purpose, loan amount, prior default, credit history). Used by default.
  * full model      - also uses lender-assigned loan grade and interest rate. Used when both are given.
"""
import pandas as pd, numpy as np, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

df = pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'data','credit_risk_cleaned.csv'))
df['loan_percent_income'] = df.loan_amnt / df.person_income        # recompute exactly from raw inputs
NUM = ['person_age','person_income','person_emp_length','loan_amnt','cb_person_cred_hist_length','loan_percent_income']
CAT = {'person_home_ownership':['MORTGAGE','OTHER','OWN','RENT'],
       'loan_intent':['DEBTCONSOLIDATION','EDUCATION','HOMEIMPROVEMENT','MEDICAL','PERSONAL','VENTURE'],
       'cb_person_default_on_file':['N','Y']}
GRADES = list('ABCDEFG')

def design(d, full):
    X = d[NUM + (['loan_int_rate'] if full else [])].copy()
    for c, levels in CAT.items():
        for l in levels: X[f'{c}_{l}'] = (d[c] == l).astype(int)
    if full:
        for g in GRADES: X[f'loan_grade_{g}'] = (d['loan_grade'] == g).astype(int)
    return X

if __name__ == '__main__':
    out, metrics = {'grades':GRADES,'cat':CAT,'num':NUM}, {}
    y = df.loan_status
    for name, full in [('applicant', False), ('full', True)]:
        X = design(df, full)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        base = GradientBoostingClassifier(n_estimators=300, learning_rate=.05, max_depth=4, subsample=.8, random_state=42)
        m = CalibratedClassifierCV(base, method='sigmoid', cv=3).fit(Xtr, ytr)   # calibrated probabilities
        p = m.predict_proba(Xte)[:, 1]; pr = (p >= .5).astype(int)
        metrics[name] = dict(AUC=roc_auc_score(yte,p), Accuracy=accuracy_score(yte,pr), Precision=precision_score(yte,pr),
                             Recall=recall_score(yte,pr), F1=f1_score(yte,pr), Brier=brier_score_loss(yte,p))
        metrics[name] = {k: round(v,3) for k, v in metrics[name].items()}
        out[name] = m.fit(X, y)           # refit on all data for the shipped model
        out[name+'_cols'] = list(X.columns)
        print(name, metrics[name])
    out['metrics'] = metrics
    joblib.dump(out, 'model/credit_risk_model.joblib')
    json.dump(metrics, open('model/model_metrics.json', 'w'), indent=1)
