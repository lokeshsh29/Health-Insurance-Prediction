import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# Load preprocessor and model
PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'
MODEL_PATH = 'artifacts/model.pkl'

def load_object(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

preprocessor = load_object(PREPROCESSOR_PATH)
model = load_object(MODEL_PATH)

st.set_page_config(page_title="Medical Charges Prediction", page_icon="💸", layout="centered")
st.title("Medical Charges Prediction")
st.markdown("Estimate your medical insurance charges based on your health and lifestyle information.")

# Sidebar info
st.sidebar.image("https://images.pexels.com/photos/2383010/pexels-photo-2383010.jpeg?auto=compress&w=400&h=400", width=120)
st.sidebar.markdown("**Project GitHub:** [Health-Insurance-Prediction](https://github.com/lokeshsh29/Health-Insurance-Prediction)")
st.sidebar.markdown("**By Lokesh Shekhar**")

# Form for user input
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=6, value=0)
    region = st.selectbox("Region", ["southwest", "southeast", "northeast", "northwest"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    submitted = st.form_submit_button("Predict my medical charges")

if submitted:
    # Prepare input for model
    input_dict = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'region': region,
        'smoker': smoker
    }

    input_df = pd.DataFrame([input_dict])
    # Transform input
    X = preprocessor.transform(input_df)
    # Predict
    prediction = model.predict(X)
    st.success(f"Your predicted medical charges: ₹{prediction[0]:,.2f}")
