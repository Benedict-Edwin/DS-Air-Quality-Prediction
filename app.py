import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Air Quality Prediction", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_air_quality_data.csv")

data = load_data()
st.title("Air Quality Prediction App")

# Show dataset
with st.expander("View Dataset"):
    st.dataframe(data)

# Feature selection
X = data.drop(columns=["AQI"])
y = data["AQI"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown(f"**Model Performance**  \nMean Squared Error: `{mse:.2f}`  \nR² Score: `{r2:.2f}`")

# User input for prediction
st.subheader("Predict AQI")
user_input = {}

cols = st.columns(len(X.columns))
for i, col in enumerate(X.columns):
    val = cols[i].number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.success(f"Predicted AQI: {prediction:.2f}")

# Plot actual vs predicted
st.subheader("Actual vs Predicted AQI")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax.set_xlabel("Actual AQI")
ax.set_ylabel("Predicted AQI")
st.pyplot(fig)