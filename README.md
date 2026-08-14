# Explainable AI Based Loan Approval Prediction System

## 📌 Project Overview

This project uses Machine Learning to predict whether a loan application is likely to be approved or rejected.

The system also uses Explainable AI (XAI) with SHAP to explain which applicant features influenced the model's prediction.

## 🎯 Objectives

- Predict loan approval using Machine Learning.
- Analyze important factors affecting loan decisions.
- Provide understandable explanations for model predictions.
- Provide an interactive web interface using Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook

## 📊 Dataset

The project uses a loan application dataset containing information such as:

- Gender
- Married Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

## 🤖 Machine Learning Model

A Random Forest Classifier is used for loan approval prediction.

The trained model achieved approximately **82.93% accuracy** on the test dataset.

## 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain the model's predictions.

The system shows:

- Feature contributions
- Positive and negative feature influence
- SHAP feature importance
- Reasons behind individual predictions

## 🌐 Streamlit Application

The Streamlit application allows users to enter applicant information and receive:

1. Loan Approval/ Rejection prediction
2. Prediction probability
3. SHAP-based explanation of the prediction

## 📁 Project Structure

```text
Explainable-AI-Loan-Approval
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── train_u6lujuX_CVtuZ9i (1).csv
│
├── model/
│   └── loan_model.pkl
│
└── notebook/
    └── loan_prediction.ipynb