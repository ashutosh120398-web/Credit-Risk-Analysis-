"""Run with:  streamlit run app.py"""
import streamlit as st
from predict import predict, HOME, INTENT, LABEL

st.set_page_config(page_title='Loan Risk Predictor', page_icon='🏦', layout='centered')
st.title('🏦 Loan Default Risk Predictor')
st.caption('Enter the applicant and loan details. The model (trained on 32,000+ real loan records) estimates the chance of default.')

c1, c2 = st.columns(2)
age = c1.number_input('Age', 18, 90, 28)
income = c2.number_input('Annual income', 1000, 5_000_000, 50000, step=1000)
home = c1.selectbox('Home ownership', HOME, index=HOME.index('RENT'))
emp = c2.number_input('Years employed', 0.0, 50.0, 3.0, step=0.5)
intent = c1.selectbox('Loan purpose', INTENT, format_func=lambda x: LABEL[x])
amount = c2.number_input('Loan amount', 500, 100000, 10000, step=500)
prior = c1.selectbox('Prior default on record?', ['N', 'Y'], format_func=lambda x: 'Yes' if x == 'Y' else 'No')
hist = c2.number_input('Credit history (years)', 0, 50, 4)

with st.expander('Optional: lender-assigned grade and interest rate'):
    st.write('If both are filled in, the more accurate "full" model is used.')
    use = st.checkbox('I have a loan grade and interest rate')
    grade = st.selectbox('Loan grade (A = safest)', list('ABCDEFG'), disabled=not use)
    rate = st.number_input('Interest rate (%)', 1.0, 40.0, 11.0, step=0.1, disabled=not use)

if st.button('Predict risk', type='primary'):
    r = predict(age, income, home, emp, intent, amount, prior, hist, grade if use else None, rate if use else None)
    p, b = r['probability'], r['risk_band']
    colour = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}[b]
    st.markdown(f"### Risk: :{colour}[{b}]")
    st.metric('Estimated probability of default', f'{p:.1%}')
    st.progress(p)
    st.info(f"**Suggested action:** {r['decision']}")
    st.write(f"Loan amount is **{r['loan_to_income']:.0%}** of annual income.  \nModel used: *{r['model']}*")
    st.subheader('Key factors')
    for x in r['reasons']: st.write('•', x)
st.caption('Educational project. The training data is a public, simulated-style dataset, so this is not for real lending decisions.')
