import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBoost model and scaler
model = joblib.load('xgb_model.jb')
scaler = joblib.load('scaler.jb')

st.title('Credit Card Default Risk Prediction')

st.markdown("""
Enter the customer's financial details below to assess the likelihood of defaulting on their next credit card payment.
""")

# Collect user input for prediction
limit_bal = st.number_input('Limit Balance (NT dollar)', min_value=0, value=20000)
sex = st.selectbox('Sex', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
education = st.selectbox('Education', [1, 2, 3, 4], format_func=lambda x: ['Graduate', 'University', 'High School', 'Others'][x-1])
marriage = st.selectbox('Marriage Status', [1, 2, 3], format_func=lambda x: ['Married', 'Single', 'Others'][x-1])
age = st.slider('Age', min_value=21, max_value=79, value=30)

pay = [st.number_input(f'PAY_{i}', value=0) for i in range(1, 7)]
bill_amt = [st.number_input(f'BILL_AMT{i}', value=0) for i in range(1, 7)]
pay_amt = [st.number_input(f'PAY_AMT{i}', value=0) for i in range(1, 7)]

# Prepare input data
input_data = np.array([[
    limit_bal, sex, education, marriage, age,
    *pay, *bill_amt, *pay_amt
]])

# Predict button
if st.button('Predict Default Risk'):
    # Apply scaler to input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_proba = model.predict_proba(input_data_scaled)[0][1]
    prediction = model.predict(input_data_scaled)[0]

    st.subheader(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
    st.write(f"Probability of Default: {prediction_proba:.2f}")
