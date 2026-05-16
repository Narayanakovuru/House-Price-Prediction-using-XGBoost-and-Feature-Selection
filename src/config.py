import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, MLRUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration for models and data
TARGET_COL = "SalePrice"
OPENML_NAME = "house_prices" 
