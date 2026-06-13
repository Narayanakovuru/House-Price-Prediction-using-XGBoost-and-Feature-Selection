"""
Inject Drift and Run Monitoring
================================
1. Loads the downloaded data/raw/ames_housing.csv
2. Appends 300 highly drifted rows (extremely large living areas, low prices, etc.)
3. Saves it back to data/raw/ames_housing.csv
4. Runs preprocessing to generate new Parquet files
5. Runs the monitoring suite to compare the new drifted data against the original reference data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure root is on path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COL
from src.preprocessing import preprocess_data
from src.monitoring import run_all_monitoring

def inject_drift_and_monitor():
    raw_path = RAW_DATA_DIR / "ames_housing.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} does not exist. Please run run_pipeline.py first.")
        return

    # Load original data
    df = pd.read_csv(raw_path)
    print(f"Original dataset shape: {df.shape}")

    # Generate drifted rows
    num_drift_rows = 300
    print(f"Generating {num_drift_rows} highly drifted rows...")
    
    # Sample from existing rows to keep categories intact, but mutate numeric values
    drift_df = df.sample(num_drift_rows, replace=True, random_state=42).copy()
    
    # 1. Mutate numeric features to be extremely different (severe data drift)
    # Huge living areas (e.g., between 8,000 and 15,000 sq ft, normal is ~1,500)
    drift_df['GrLivArea'] = np.random.randint(8000, 15000, size=num_drift_rows)
    
    # Massive basement areas
    drift_df['TotalBsmtSF'] = np.random.randint(6000, 12000, size=num_drift_rows)
    
    # Huge 1st Floor Area
    drift_df['1stFlrSF'] = np.random.randint(6000, 12000, size=num_drift_rows)
    
    # Extremely low SalePrice (e.g., $10,000 to $30,000) for such massive houses!
    # This will create huge model performance degradation and target drift.
    drift_df['SalePrice'] = np.random.randint(10000, 30000, size=num_drift_rows)
    
    # Appending to original CSV
    new_df = pd.concat([df, drift_df], ignore_index=True)
    new_df.to_csv(raw_path, index=False)
    print(f"Drift injected! New dataset saved to {raw_path} with shape: {new_df.shape}")
    
    # Run preprocessing on the new drifted raw data
    print("Running preprocessing on mutated raw data...")
    preprocess_data()
    
    # Run monitoring
    print("Running monitoring suite with the new drifted Parquets...")
    reports = run_all_monitoring(simulate_drift=False)
    
    print("\n" + "="*50)
    print("Drift successfully simulated!")
    print("="*50)
    for name, path in reports.items():
        print(f"{name.upper()}: {path}")
    print("="*50)
    return reports

if __name__ == "__main__":
    inject_drift_and_monitor()
