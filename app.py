"""
app.py
------
Web app Streamlit buat prediksi credit score nasabah.
Jalankan: streamlit run app.py
"""

import streamlit as st

from inference import CreditScorePredictor, TEST_CASES

st.set_page_config(page_title="Credit Score Prediction", page_icon="💳",
                   layout="centered")


@st.cache_resource
def load_predictor():
    return CreditScorePredictor()


predictor = load_predictor()

OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
               "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
               "Accountant", "Musician", "Mechanic", "Writer", "Architect"]
CREDIT_MIX = ["Good", "Standard", "Bad"]
PAY_MIN = ["Yes", "No", "NM"]
PAY_BEHAVIOUR = ["High_spent_Large_value_payments", "High_spent_Medium_value_payments",
                 "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                 "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"]

st.title("💳 Credit Score Prediction")
st.caption("Menilai performa kredit nasabah (Good / Standard / Poor) "
           "menggunakan model XGBoost.")

# preset test case biar gampang demo
st.sidebar.header("Preset Test Case")
preset = st.sidebar.selectbox("Isi otomatis dengan contoh:",
                              ["— manual —"] + list(TEST_CASES.keys()))
d = TEST_CASES.get(preset, {})


def g(key, default):
    return d.get(key, default)


st.subheader("Data Demografi & Pekerjaan")
c1, c2 = st.columns(2)
age = c1.number_input("Age", 18, 100, int(g("Age", 30)))
occupation = c2.selectbox("Occupation", OCCUPATIONS,
                          index=OCCUPATIONS.index(g("Occupation", "Engineer"))
                          if g("Occupation", "Engineer") in OCCUPATIONS else 0)

st.subheader("Pendapatan")
c1, c2 = st.columns(2)
annual_income = c1.number_input("Annual Income", 0.0, value=float(g("Annual_Income", 50000.0)))
monthly_salary = c2.number_input("Monthly Inhand Salary", 0.0,
                                 value=float(g("Monthly_Inhand_Salary", 4000.0)))

st.subheader("Akun & Pinjaman")
c1, c2, c3 = st.columns(3)
num_bank = c1.number_input("Num Bank Accounts", 0, 20, int(g("Num_Bank_Accounts", 5)))
num_card = c2.number_input("Num Credit Card", 0, 20, int(g("Num_Credit_Card", 5)))
interest = c3.number_input("Interest Rate", 0, 50, int(g("Interest_Rate", 14)))
c1, c2, c3 = st.columns(3)
num_loan = c1.number_input("Num of Loan", 0, 20, int(g("Num_of_Loan", 3)))
num_loan_types = c2.number_input("Num Loan Types", 0, 12, int(g("Num_Loan_Types", 3)))
outstanding = c3.number_input("Outstanding Debt", 0.0, value=float(g("Outstanding_Debt", 1000.0)))

st.subheader("Riwayat Pembayaran")
c1, c2, c3 = st.columns(3)
delay = c1.number_input("Delay from due date", -10, 200, int(g("Delay_from_due_date", 15)))
num_delayed = c2.number_input("Num of Delayed Payment", 0, 100, int(g("Num_of_Delayed_Payment", 10)))
inquiries = c3.number_input("Num Credit Inquiries", 0, 100, int(g("Num_Credit_Inquiries", 4)))
c1, c2, c3 = st.columns(3)
changed_limit = c1.number_input("Changed Credit Limit", value=float(g("Changed_Credit_Limit", 8.0)))
util = c2.number_input("Credit Utilization Ratio", 0.0, 100.0, float(g("Credit_Utilization_Ratio", 32.0)))
history_months = c3.number_input("Credit History (months)", 0, 600, int(g("Credit_History_Months", 200)))

st.subheader("Lainnya")
c1, c2 = st.columns(2)
credit_mix = c1.selectbox("Credit Mix", CREDIT_MIX,
                          index=CREDIT_MIX.index(g("Credit_Mix", "Standard")))
pay_min = c2.selectbox("Payment of Min Amount", PAY_MIN,
                       index=PAY_MIN.index(g("Payment_of_Min_Amount", "No")))
c1, c2 = st.columns(2)
total_emi = c1.number_input("Total EMI per month", 0.0, value=float(g("Total_EMI_per_month", 100.0)))
invested = c2.number_input("Amount invested monthly", 0.0, value=float(g("Amount_invested_monthly", 200.0)))
balance = st.number_input("Monthly Balance", value=float(g("Monthly_Balance", 400.0)))
behaviour = st.selectbox("Payment Behaviour", PAY_BEHAVIOUR,
                         index=PAY_BEHAVIOUR.index(g("Payment_Behaviour", PAY_BEHAVIOUR[0]))
                         if g("Payment_Behaviour", PAY_BEHAVIOUR[0]) in PAY_BEHAVIOUR else 0)

if st.button("Prediksi Credit Score", type="primary", use_container_width=True):
    payload = {
        "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
        "Num_Credit_Card": num_card, "Interest_Rate": interest, "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay, "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": inquiries,
        "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding,
        "Credit_Utilization_Ratio": util, "Credit_History_Months": history_months,
        "Payment_of_Min_Amount": pay_min, "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested, "Payment_Behaviour": behaviour,
        "Monthly_Balance": balance, "Num_Loan_Types": num_loan_types,
    }
    label, proba = predictor.predict(payload)

    color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(label, "gray")
    st.markdown(f"### Hasil: :{color}[{label}]")
    if proba:
        st.write("Probabilitas per kelas:")
        for cls in ["Poor", "Standard", "Good"]:
            if cls in proba:
                st.progress(proba[cls], text=f"{cls} — {proba[cls]:.1%}")
