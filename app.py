import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("xgboost_model.pkl")  # Ensure this file is in your repo

st.title("Air Quality Index (AQI) Prediction")

# Collect inputs
pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=-10.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
wind = st.number_input("Wind Speed (m/s)", min_value=0.0)

if st.button("Predict AQI"):
    features = np.array([[pm25, pm10, no2, so2, co, o3, temp, humidity, wind]])
    aqi = model.predict(features)[0]
    st.success(f"Predicted AQI: {aqi:.2f}")
