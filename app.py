import streamlit as st
import pandas as pd
from joblib import load

# =========================
# LOAD MODEL
# =========================
model = load("loan_model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Loan Risk System", layout="wide")

st.title("🏦 Loan Approval & Risk Analysis System")
st.markdown("### Predict loan approval with risk insights")

# =========================
# LAYOUT (2 COLUMNS)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    st.subheader("💰 Financial Details")

    app_income = st.number_input("Applicant Income", min_value=0)
    co_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=1)
    credit_history = st.selectbox("Credit History", [1, 0])

# =========================
# PREDICTION
# =========================

if st.button("🔍 Analyze Loan Risk"):

    # =========================
    # DATA PREPARATION
    # =========================
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Married': [1 if married == "Yes" else 0],
        'Education': [1 if education == "Graduate" else 0],
        'Self_Employed': [1 if self_employed == "Yes" else 0],
        'ApplicantIncome': [app_income],
        'CoapplicantIncome': [co_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history]
    })

    # =========================
    # FEATURE ENGINEERING
    # =========================
    input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
    input_data['LoanToIncome'] = input_data['LoanAmount'] / (input_data['TotalIncome'] + 1)
    input_data['EMI'] = input_data['LoanAmount'] / (input_data['Loan_Amount_Term'] + 1)
    input_data['EMItoIncome'] = input_data['EMI'] / (input_data['TotalIncome'] + 1)

    # =========================
    # ENCODING
    # =========================
    input_data['Property_Area_Semiurban'] = 1 if property_area == "Semiurban" else 0
    input_data['Property_Area_Urban'] = 1 if property_area == "Urban" else 0

    input_data['Dependents_1'] = 1 if dependents == "1" else 0
    input_data['Dependents_2'] = 1 if dependents == "2" else 0
    input_data['Dependents_3+'] = 1 if dependents == "3+" else 0

    # Match training columns
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # =========================
    # RESULTS DASHBOARD
    # =========================
    st.markdown("---")
    st.subheader("📊 Decision Dashboard")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Approval Probability", f"{probability:.2%}")

    with colB:
        st.metric("Total Income", f"{input_data['TotalIncome'][0]:,.0f}")

    with colC:
        st.metric("Loan Ratio", f"{input_data['LoanToIncome'][0]:.2f}")

    # =========================
    # DECISION LOGIC (REAL WORLD)
    # =========================
    st.subheader("🏦 Final Decision")

    if probability > 0.65:
        st.success("🟢 LOW RISK → APPROVE LOAN")
    elif probability > 0.45:
        st.warning("🟡 MEDIUM RISK → MANUAL REVIEW REQUIRED")
    else:
        st.error("🔴 HIGH RISK → REJECT LOAN")

    # =========================
    # RISK ANALYSIS
    # =========================
    st.subheader("⚠️ Risk Factors Analysis")

    if input_data['LoanToIncome'][0] > 0.4:
        st.warning("High loan-to-income ratio (financial strain risk)")

    if input_data['EMItoIncome'][0] > 0.08:
        st.warning("High EMI burden relative to income")

    if credit_history == 0:
        st.error("No credit history — high default risk")

    if self_employed == "Yes":
        st.info("Self-employed applicants may have unstable income")

    # =========================
    # BUSINESS SUMMARY
    # =========================
    st.subheader("💡 Business Summary")

    if probability > 0.65:
        st.write("Customer shows strong repayment ability with manageable financial risk.")
    elif probability > 0.45:
        st.write("Customer presents moderate risk. Further financial verification is recommended.")
    else:
        st.write("Customer is likely to default based on financial indicators.")