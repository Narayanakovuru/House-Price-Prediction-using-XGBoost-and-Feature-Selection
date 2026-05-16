import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Dict, Any

import sys
from pathlib import Path
# Add base project dir to pythonpath to resolve src.config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DIR

app = FastAPI(title="Ames Housing Price Prediction API", version="1.0")

# Load model and preprocessor globally
logger = logging.getLogger("uvicorn.error")

try:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "best_xgb_model.json")
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model/preprocessor: {e}")

class Features(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
def predict_price(features: Features):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features.data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Fix column names for XGBoost
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.replace('[', '').replace(']', '').replace('<', '') for f in feature_names]
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        except:
            # Fallback
            X_processed_df = pd.DataFrame(X_processed)
            X_processed_df.columns = [f"f_{i}" for i in range(X_processed_df.shape[1])]
        
        # Predict (returns log price)
        log_price = model.predict(X_processed_df)[0]
        
        # Inverse log transform
        price = np.expm1(log_price)
        
        return {"predicted_price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
