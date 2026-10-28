import streamlit as st
import pandas as pd
import joblib
import shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/loan_model.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-title { font-size: 38px; font-weight: 700; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #888888; font-size: 17px; margin-bottom: 30px; }
    .section-title { font-size: 22px; font-weight: 600; margin-top: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💰 AI Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning based loan prediction with Explainable AI</div>', unsafe_allow_html=True)

# -----------------------------
# Applicant & Loan Information (UI)
# -----------------------------
st.markdown('<div class="section-title">👤 Applicant Information</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col3:
    credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1)" if x == 1.0 else "Poor/Unavailable (0)")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)

st.markdown('<div class="section-title">🏦 Loan Information</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)

with col4:
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=150.0)
with col5:
    loan_term = st.number_input("Loan Amount Term (months)", min_value=1.0, value=360.0)

st.write("")
predict_button = st.button("🔍 Predict Loan Status", use_container_width=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_button:
    # Encode inputs
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    dependents_encoded = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    property_area_encoded = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    input_data = pd.DataFrame({
        "Gender": [gender_encoded], "Married": [married_encoded], "Dependents": [dependents_encoded],
        "Education": [education_encoded], "Self_Employed": [self_employed_encoded], 
        "ApplicantIncome": [applicant_income], "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount], "Loan_Amount_Term": [loan_term], 
        "Credit_History": [credit_history], "Property_Area": [property_area_encoded]
    })

    # ML Prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        if prediction == 1:
            st.success("### ✅ Loan Approved")
        else:
            st.error("### ❌ Loan Rejected")

    with result_col2:
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        st.metric("Prediction Probability", f"{probability * 100:.2f}%")

    # -----------------------------
    # Explainable AI (SHAP)
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]
    
    shap_df = pd.DataFrame({"Feature": input_data.columns, "SHAP Value": values})
    shap_df = shap_df.sort_values(by="SHAP Value", key=abs, ascending=False)
    
    # Show SHAP Table & Graph
    st.markdown("---")
    st.markdown('<div class="section-title">🧠 Technical Breakdown (SHAP)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Feature Impact Table")
        # Ye line table banayegi
        st.dataframe(shap_df, hide_index=True, use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Impact Graph")
        # Ye line graph banayegi
        st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])
        st.caption("Positive values push toward approval, negative values push toward rejection.")