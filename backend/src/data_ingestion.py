import pandas as pd
from sklearn.datasets import fetch_openml
from src.config import RAW_DATA_DIR, OPENML_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data():
    logger.info(f"Fetching {OPENML_NAME} dataset from OpenML...")
    housing = fetch_openml(name=OPENML_NAME, as_frame=True)
    df = housing.frame
    
    # Save raw data
    raw_path = RAW_DATA_DIR / "ames_housing.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")
    return df

if __name__ == "__main__":
    ingest_data()
