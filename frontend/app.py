import streamlit as st
import pandas as pd
import requests
import json
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "ames_housing.csv"

# UI Config
st.set_page_config(page_title="Ames Housing Price Predictor", page_icon="🏡", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff6b6b;
        border-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏡 Ames Housing Price Predictor")
st.markdown("Enter the characteristics of a house to predict its approximate sale price using an XGBoost model.")

@st.cache_data
def get_default_data():
    if not RAW_DATA_PATH.exists():
        return None
    df = pd.read_csv(RAW_DATA_PATH)
    defaults = {}
    for col in df.columns:
        if col in ['Id', 'SalePrice']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = str(df[col].mode()[0])
    return defaults

defaults = get_default_data()

if defaults:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    # Store user inputs
    user_inputs = defaults.copy()
    
    with col1:
        st.subheader("🎯 Property Characteristics")
        user_inputs['OverallQual'] = st.slider("Overall Quality (1-10)", 1, 10, int(defaults.get('OverallQual', 5)))
        user_inputs['YearBuilt'] = st.slider("Year Built", 1870, 2010, int(defaults.get('YearBuilt', 1970)))
        user_inputs['YearRemodAdd'] = st.slider("Year Remodeled", 1950, 2010, int(defaults.get('YearRemodAdd', 1990)))
        user_inputs['TotRmsAbvGrd'] = st.number_input("Total Rooms Above Ground", value=int(defaults.get('TotRmsAbvGrd', 6)))

    with col2:
        st.subheader("📏 Size & Dimensions")
        user_inputs['GrLivArea'] = st.number_input("Above Ground Living Area (sq ft)", value=int(defaults.get('GrLivArea', 1500)))
        user_inputs['TotalBsmtSF'] = st.number_input("Total Basement Area (sq ft)", value=int(defaults.get('TotalBsmtSF', 1000)))
        user_inputs['1stFlrSF'] = st.number_input("1st Floor Area (sq ft)", value=int(defaults.get('1stFlrSF', 1000)))
        user_inputs['GarageCars'] = st.slider("Garage Capacity (Cars)", 0, 5, int(defaults.get('GarageCars', 2)))
        user_inputs['FullBath'] = st.slider("Full Bathrooms", 0, 4, int(defaults.get('FullBath', 2)))

    st.markdown("---")
    left, middle, right = st.columns([1, 2, 1])
    with middle:
        if st.button("Calculate Predicted Price 🚀", use_container_width=True):
            with st.spinner("Running inference via ML pipeline..."):
                try:
                    res = requests.post("http://127.0.0.1:8000/predict", json={"data": user_inputs})
                    if res.status_code == 200:
                        price = res.json().get("predicted_price", 0)
                        st.success(f"## Implied Value: **${price:,.2f}**")
                        st.balloons()
                    else:
                        st.error(f"Error from API: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Start the FastAPI backend first using `uvicorn app.main:app --reload`.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
else:
    st.warning("Data not found. Please run the data ingestion and training pipeline first (run `python run_pipeline.py`).")
