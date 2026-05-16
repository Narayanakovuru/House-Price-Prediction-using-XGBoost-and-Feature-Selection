import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from src.config import PROCESSED_DATA_DIR, TARGET_COL, MODEL_DIR, MLRUNS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLRUNS_DIR.absolute().as_uri())
mlflow.set_experiment("Ames_Housing_XGBoost")

def train_best_model():
    logger.info("Loading training data...")
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")[TARGET_COL]
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")[TARGET_COL]

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        return rmse

    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) # 10 trials for speed in demonstration
    
    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    
    with mlflow.start_run(run_name="Best_XGBoost_Model"):
        mlflow.log_params(best_params)
        
        # Train final model
        model = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        logger.info(f"Model Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Log model
        mlflow.xgboost.log_model(model, "xgboost-model")
        
        # Save model locally for fast serving
        model.save_model(MODEL_DIR / "best_xgb_model.json")
        logger.info("Model training and logging completed.")

if __name__ == "__main__":
    train_best_model()
