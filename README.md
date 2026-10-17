# 🏦 Credit Risk Analysis & Default Prediction

An end-to-end data analytics project on **32,580 real loan records** (Kaggle *Credit Risk Dataset*). It finds which borrower and loan factors drive default, builds a simple risk scorecard, compares machine learning models, and includes an app that predicts the default risk of a new applicant.

**Tech:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit · Jupyter

---

## 📌 Business questions
1. What share of loans default, and how much money is at risk?
2. Which factors (grade, income, loan-to-income, housing, purpose, prior default) are linked to default?
3. Do risk factors compound when they appear together?
4. Can a model predict the default risk of a new applicant?

## 📊 Key findings
| Finding | Result |
|---|---|
| Overall default rate | **21.9%** (defaulted loans hold ~24.8% of the amount lent) |
| Loan-to-income above 30% | default jumps from **22% → 69%** |
| Loan grade | **10%** (grade A) → **59%** (D) → **98%** (G) |
| Income | lowest quintile **43%** vs highest **9%** |
| Housing | renters **31.6%** vs home owners **7.5%** |
| Prior default on record | **37.9%** vs 18.4% without |
| Age | almost no signal (20–23% across bands) |
| 6-flag risk scorecard | **4.4%** default with 0 flags → **99%** with all 6 |

![Risk flags](charts/15_risk_flags.png)

## 🔍 Methodology
1. **Data quality:** removed 165 duplicates and impossible values (ages 123/144, 123 years employment); filled missing employment length (median) and interest rate (median by loan grade); added missing-value flags.
2. **EDA:** default rate by category and numeric bands, distributions, boxplots, correlation matrix, interaction heatmaps.
3. **Risk scorecard:** one point each for grade D–G, loan ≥ 30% of income, renter, prior default, rate ≥ 14%, bottom-quartile income.
4. **Modelling:** Logistic Regression, Random Forest, Gradient Boosting on an 80/20 stratified split; permutation feature importance.

| Model | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.869 | 0.814 | 0.553 | 0.778 |
| Random Forest | 0.929 | 0.924 | 0.888 | 0.748 |
| Gradient Boosting | 0.922 | 0.925 | 0.934 | 0.707 |

## 🤖 Risk prediction app
A calibrated Gradient Boosting model turns user-entered details into a **default probability, risk band (Low / Medium / High), suggested action and key risk factors**.

| Model | Inputs | AUC |
|---|---|---|
| Applicant model (default) | age, income, home ownership, years employed, loan purpose, loan amount, prior default, credit history | 0.90 |
| Full model (optional) | the above + lender grade and interest rate | 0.94 |

```bash
streamlit run app.py          # web form
python predict.py             # terminal prompts
python predict.py --age 23 --income 28000 --home RENT --emp 1 --intent MEDICAL --amount 12000 --prior Y --hist 2
```
In code: `from predict import predict; predict(23, 28000, 'RENT', 1, 'MEDICAL', 12000, 'Y', 2)`

## 🚀 Run it yourself
```bash
git clone https://github.com/ashutosh120398-web/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python analysis.py        # regenerates charts and results
python train_model.py     # retrains the models
jupyter notebook notebooks/Credit_Risk_Analysis.ipynb
```

## 📁 Project structure
```
├── data/                 original, cleaned and featured CSV files
├── notebooks/            Credit_Risk_Analysis.ipynb (full analysis)
├── charts/               18 charts used in the report
├── report/               Credit_Risk_Analysis_Report.docx (22 pages)
├── model/                saved models and metrics
├── analysis.py           cleaning, EDA, scorecard, model comparison
├── train_model.py        trains the prediction models
├── predict.py            terminal predictor
├── app.py                Streamlit web app
└── requirements.txt
```

## 💡 Recommendations
- Cap loan-to-income around 30% or require extra security above it.
- Apply stronger underwriting to grades D and below.
- Treat a prior default as a mandatory review trigger.
- Use the multi-flag scorecard for triage: auto-approve 0–1 flags, review 2–3, decline or restructure 4+.
- Pick the model cut-off by weighing the cost of a missed default against a rejected good borrower.

## ⚠️ Limitations
- The dataset is a cleaned public sample widely regarded as simulated, so this project demonstrates analytical method, not real bank statistics. **Not for real lending decisions.**
- Loan grade and interest rate are set by the lender, so they partly reuse the lender's own risk view. This is why the applicant model excludes them.
- Findings show association, not causation.

## 👤 Author
**Ashutosh Kumar Singh** – B.Tech CSE (Data Science), aspiring Data Analyst  
GitHub: [ashutosh120398-web](https://github.com/ashutosh120398-web)

**Dataset:** Kaggle – Credit Risk Dataset.
