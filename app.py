import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("🌍 Air Quality Index Prediction")
st.markdown("Predict AQI using machine learning algorithms with enhanced explainability and visualization.")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_air_quality_data.csv")
    return df.dropna()

df = load_data()
st.subheader("📊 Sample of Dataset")
st.dataframe(df.head())

# Data Visualization
with st.expander("🔍 Data Correlation Heatmap"):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Feature Selection
X = df.drop(['AQI'], axis=1)
y = df['AQI']

# Sidebar for model selection
st.sidebar.title("Model Selection & Parameters")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
else:
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1)
    model = XGBRegressor(learning_rate=learning_rate, random_state=42)

# Train Model
if st.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model Trained Successfully!")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

    # SHAP Explainability
    st.subheader("🔍 Feature Importance using SHAP")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    st.pyplot(fig)

# User Prediction
st.subheader("📌 Predict AQI from Input")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

if st.button("Predict AQI"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted AQI: {prediction:.2f}")
