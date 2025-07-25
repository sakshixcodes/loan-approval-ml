import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model/loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# UI inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Approval"):
    # Encode inputs like training
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1.0 if credit_history == "Yes" else 0.0
    property_area_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_area_dict[property_area]

    # Prepare input array
    features = np.array([[gender, married, dependents, education, self_employed,
                          applicant_income, coapplicant_income, loan_amount,
                          loan_amount_term, credit_history, property_area]])

    # Predict
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ Loan will likely be Approved!")
    else:
        st.error("❌ Loan will likely be Rejected.")
