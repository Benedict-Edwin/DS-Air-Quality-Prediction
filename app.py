import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import shap
import joblib

st.set_page_config(page_title="Air Quality Prediction", layout="wide")

st.title("Air Quality Index Prediction")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_air_quality_data.csv")
    return df

df = load_data()

# Show raw data toggle
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Show dataset info
if st.checkbox("Show data info"):
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

# Correlation heatmap - fix by selecting numeric columns only
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)
plt.clf()

# Prepare data
# Drop non-numeric or irrelevant columns if needed
X = df.drop(['AQI', 'Date', 'AQI_Bucket', 'City'], axis=1, errors='ignore')

# Encode categorical columns if any remain
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = df['AQI']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_option = st.selectbox("Choose regression model", ("Linear Regression", "XGBoost"))

if model_option == "Linear Regression":
    model = LinearRegression()
else:
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Train model button
if st.button("Train Model"):
    model.fit(X_train, y_train)
    st.success("Model trained successfully!")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # SHAP explanation for XGBoost
    if model_option == "XGBoost":
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        st.subheader("SHAP Summary Plot")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt)
        plt.clf()

# Single prediction inputs
st.sidebar.header("Make a Prediction")

input_data = {}
for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        input_data[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)
    else:
        # For categorical columns (already encoded), just slider for encoded range
        input_data[col] = st.sidebar.slider(col, min_value=int(min_val), max_value=int(max_val), value=int(mean_val))

# Predict button
if st.sidebar.button("Predict AQI"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"Predicted AQI: {prediction:.2f}")

