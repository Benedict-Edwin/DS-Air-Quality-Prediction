import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
@st.cache
def load_data():
    # Replace 'your_dataset.csv' with the actual path to your dataset
    df = pd.read_csv("synthetic_air_quality_data.csv")
    
    # Convert date column to datetime format if necessary
    if 'date_column' in df.columns:  # Replace 'date_column' with your actual date column name
        df['date_column'] = pd.to_datetime(df['date_column'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    return df

# Main function to run the app
def main():
    st.title("Air Quality Prediction App")
    
    # Load data
    df = load_data()
    
    # Display the dataframe
    st.dataframe(df)
    
    # Data Visualization
    with st.expander("🔍 Data Correlation Heatmap"):
        plt.figure(figsize=(12, 6))
        # Check if there are numeric columns to correlate
        if not df.select_dtypes(include=['number']).empty:
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.write("No numeric columns to display correlation.")

    # Show dataset info
    if st.checkbox("Show data info"):
        buffer = []
        df.info(buf=buffer)
        info_str = "\n".join(buffer)
        st.text(info_str)

# Run the app
if __name__ == "__main__":
    main()
