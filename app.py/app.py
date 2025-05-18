import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.set_page_config(page_title="Air Quality Predictor", layout="centered")
st.title("Air Quality Index (AQI) Prediction")
st.markdown("Upload the data and get AQI predictions instantly.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Input Data Preview")
    st.write(data.head())

    # Load trained model
    try:
        model = joblib.load("air_quality_model.pkl")  # Ensure this exists
    except FileNotFoundError:
        st.error("Model file not found! Please add 'air_quality_model.pkl'.")
        st.stop()

    # Prediction
    if st.button("Predict AQI"):
        try:
            features = data.drop(columns=["AQI"], errors='ignore')  # Drop actual AQI if exists
            predictions = model.predict(features)
            data["Predicted_AQI"] = predictions
            st.subheader("Prediction Results")
            st.write(data)
            
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="predicted_aqi.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
