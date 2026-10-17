# Bank Loan / Credit Risk Analysis

Analysis of default patterns in a 32,580-loan credit risk dataset (Kaggle *Credit Risk Dataset*) to find which borrower and loan factors are linked to default.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter

## Key findings
- 21.9% of loans default; defaulted loans hold ~24.8% of the money lent
- Loan-to-income above 30% lifts default from 22% to 69%
- Default rises from 10% (grade A) to 59% (grade D) and 98% (grade G)
- Lowest-income quintile defaults at 43% vs 9% for the highest; renters 31.6% vs owners 7.5%
- Prior default on file doubles risk (37.9% vs 18.4%); age adds almost no signal
- 6-flag scorecard: 4.4% default with 0 flags, 99% with all 6
- Random Forest AUC 0.93, Gradient Boosting 0.92, Logistic Regression 0.87

## Risk prediction model (new)
A calibrated Gradient Boosting model predicts the probability that a new applicant will default, from the details a user enters.

| Model | Inputs | AUC | Precision | Recall |
|---|---|---|---|---|
| Applicant model (default) | age, income, home ownership, years employed, loan purpose, loan amount, prior default, credit history | 0.90 | 0.90 | 0.58 |
| Full model (optional) | the above + lender grade and interest rate | 0.94 | 0.94 | 0.74 |

Scores are on a held-out 20% test set at a 0.5 cut-off. Output: default probability, a risk band (LOW < 15%, MEDIUM 15-40%, HIGH > 40%), a suggested action and the key risk factors.

```
python train_model.py        # retrains and saves model/credit_risk_model.joblib
python predict.py            # answer prompts in the terminal
python predict.py --age 23 --income 28000 --home RENT --emp 1 --intent MEDICAL --amount 12000 --prior Y --hist 2
streamlit run app.py         # web form
```
Use it in code: `from predict import predict; predict(23, 28000, 'RENT', 1, 'MEDICAL', 12000, 'Y', 2)`

## Folder structure
```
data/        original, cleaned and featured CSV files
notebooks/   Credit_Risk_Analysis.ipynb (executed)
charts/      18 charts used in the report
report/      Credit_Risk_Analysis_Report.docx
model/       saved models and metrics
train_model.py, predict.py, app.py   train, predict (terminal) and web form
analysis.py  regenerates charts and results.json
```

## Run it
```
pip install -r requirements.txt
python analysis.py
```

## Note
The dataset is a cleaned public sample widely regarded as simulated, and loan grade / interest rate are set by the lender. The project demonstrates analytical method rather than real bank statistics.
