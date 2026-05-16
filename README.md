# Ames Housing Price Prediction: End-to-End MLOps Pipeline 🏡

An ultra-modern, efficient machine learning pipeline and web application for predicting the sale price of houses in Ames, Iowa.

## 🚀 Key Features

*   **Model Validation & Training:** XGBoost model with optimized hyperparameters via Optuna.
*   **Experiment Tracking:** Uses MLflow to track performance metrics (RMSE, MAE, R2) and store versioned model artifacts.
*   **Production Serving API:** FastAPI implementation designed for high parallelism and minimal latency.
*   **Modern Frontend:** Streamlit application providing interactive sliders and inputs to dynamically predict house prices. 

## 🛠 Project Structure

```text
ames_housing_mlops/
├── data/
│   ├── raw/                 # Downloaded dataset
│   └── processed/           # Engineered features and splits
├── mlruns/                  # MLflow experiment tracking logs
├── models/                  # Saved models & preprocessor artifacts
├── src/                     # Core computational logic
├── app/                     # FastAPI Serving Application
└── frontend/                # Streamlit Web Application
```

## ⚙️ How to Run

### 1. Installation

Ensure you have a modern version of Python (3.9+). 
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline

Run the end-to-end master script. This will download the dataset asynchronously, process the features, run distributed hyperparameter searching, and output an optimized model pipeline.
```bash
python run_pipeline.py
```

### 3. Start Model Serving (FastAPI)

In a separate terminal, deploy the API backend:
```bash
uvicorn app.main:app --reload
```
You can view the interactive swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start the Application UI (Streamlit)

Launch the interactive UI to test real-time predictions:
```bash
streamlit run frontend/app.py
```
