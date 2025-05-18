import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Configure page
st.set_page_config(page_title="Air Quality Predictor", layout="centered")
st.title("Air Quality Index (AQI) Prediction")

# Load model function
def load_model():
    try:
        # Try to load your existing model (change the filename if needed)
        model = joblib.load("random_forest_model.pkl")
        st.success("Pre-trained model loaded successfully!")
        return model
    except FileNotFoundError:
        st.warning("No pre-trained model found. Please upload training data.")
        return None

# Main app function
def main():
    st.markdown("""
    ### How to use:
    1. Upload your air quality data (CSV)
    2. The app will predict AQI
    3. Download predictions
    """)
    
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload air quality data (CSV)", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(data.head())
        
        # Check if we need to train (no model exists)
        if model is None and 'AQI' in data.columns:
            if st.button("Train New Model"):
                with st.spinner("Training model..."):
                    X = data.drop(columns=['AQI'])
                    y = data['AQI']
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # Save the model
                    joblib.dump(model, "random_forest_model.pkl")
                    st.success("Model trained and saved!")
        
        # Prediction section
        if model is not None:
            if st.button("Predict AQI"):
                try:
                    features = data.drop(columns=['AQI'], errors='ignore')
                    predictions = model.predict(features)
                    data['Predicted_AQI'] = predictions
                    
                    st.subheader("Prediction Results")
                    st.write(data)
                    
                    # Download results
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        data=csv,
                        file_name="aqi_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
