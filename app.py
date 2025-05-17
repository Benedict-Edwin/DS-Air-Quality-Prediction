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
import io

st.set_page_config(page_title="Air Quality Prediction", layout="wide")

st.title("Air Quality Index Prediction")

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_air_quality_data.csv")

df = load_data()

# Show raw data toggle
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Show dataset info toggle with buffer
if st.checkbox("Show data info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Correlation heatmap (numeric columns only)
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)
plt.clf()

# Prepare features and target
X = df.drop(columns=['AQI', 'Date', 'AQI_Bucket', 'City'], errors='ignore')

# Encode categorical columns if any remain
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = df['AQI']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_option = st.selectbox("Choose regression model", ("Linear Regression", "XGBoost"))

# Initialize model based on selection
def get_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    else:
        return XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Persist model in session state
if "model" not in st.session_state:
    st.session_state.model = None

# Train model button
if st.button("Train Model"):
    model = get_model(model_option)
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.success("Model trained successfully!")

    # Predict on test set
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

    # SHAP explanation (only for XGBoost)
    if model_option == "XGBoost":
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        st.subheader("SHAP Summary Plot")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt)
        plt.clf()

# Sidebar inputs for prediction
st.sidebar.header("Make a Prediction")

input_data = {}
for col in X.columns:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())

    if np.issubdtype(df[col].dtype, np.integer):
        input_data[col] = st.sidebar.slider(col, int(col_min), int(col_max), int(col_mean))
    else:
        input_data[col] = st.sidebar.slider(col, col_min, col_max, col_mean)

# Predict AQI for user input
if st.sidebar.button("Predict AQI"):
    if st.session_state.model is None:
        st.sidebar.error("Please train the model first!")
    else:
        try:
            input_df = pd.DataFrame([input_data])
            prediction = st.session_state.model.predict(input_df)[0]
            st.sidebar.success(f"Predicted AQI: {prediction:.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")
