"""
Ames Housing MLOps - Stage 4: Model & Data Monitoring
======================================================
Uses Evidently AI (v0.7+) to generate three monitoring reports:

  1. Data Drift      - Are incoming features shifting from training data?
  2. Model Quality   - Is prediction quality (RMSE, MAE, R2) degrading?
  3. Target Drift    - Is the SalePrice distribution changing over time?

Reports are saved as interactive HTML files in reports/ directory.

Usage:
  Normal:          python -m src.monitoring
  Simulate drift:  python -m src.monitoring --drift
"""

import sys
import logging
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_reference_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load training data as the baseline reference."""
    logger.info("Loading reference (training) data...")
    X = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")[TARGET_COL]
    
    # Assert types to satisfy static type checkers (Pyright/VSCode)
    assert isinstance(X, pd.DataFrame), "Expected X to be a pd.DataFrame"
    assert isinstance(y, pd.Series), "Expected y to be a pd.Series"
    return X, y


def load_current_data(simulate_drift: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load test data as simulated 'current production' data.
    In a real project this would be live inference data collected from your API.
    """
    logger.info("Loading current (test) data...")
    X = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")[TARGET_COL]

    # Assert types to satisfy static type checkers (Pyright/VSCode)
    assert isinstance(X, pd.DataFrame), "Expected X to be a pd.DataFrame"
    assert isinstance(y, pd.Series), "Expected y to be a pd.Series"

    if simulate_drift:
        logger.warning("Simulating artificial data drift for demonstration...")
        rng = np.random.default_rng(42)
        num_cols = X.select_dtypes(include=[np.number]).columns[:15]
        X = X.copy()
        X[num_cols] = X[num_cols] * 1.8 + rng.normal(0, 3, (len(X), len(num_cols)))

    return X, y


def get_predictions(X: pd.DataFrame) -> np.ndarray:
    """Run inference on a dataset using the saved XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "best_xgb_model.json")
    log_preds = model.predict(X)
    return np.expm1(log_preds)   # inverse log → actual house prices


# ──────────────────────────────────────────────
# Report 1: Data Drift
# ──────────────────────────────────────────────

def run_data_drift_report(X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> str:
    """
    Compare reference (training) vs current feature distributions.
    Flags columns that have statistically drifted using KS / chi-squared tests.
    """
    logger.info("Running Data Drift analysis...")

    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=X_ref, current_data=X_cur)

    path = REPORTS_DIR / f"data_drift_{_ts()}.html"
    report.save_html(str(path))
    logger.info(f"Data Drift report  --> {path.name}")
    return str(path)


# ──────────────────────────────────────────────
# Report 2: Model Performance / Regression Quality
# ──────────────────────────────────────────────

def run_model_quality_report(
    X_ref: pd.DataFrame, y_ref: pd.Series,
    X_cur: pd.DataFrame, y_cur: pd.Series,
) -> str:
    """
    Compare model prediction quality on reference vs current data.
    Shows RMSE, MAE, R2, residuals, and error distribution.
    """
    logger.info("Running Model Quality analysis...")

    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import RegressionPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

    # Build datasets Evidently expects: features + target + prediction columns
    ref_df = X_ref.copy()
    ref_df["target"]     = np.expm1(y_ref.values)
    ref_df["prediction"] = get_predictions(X_ref)

    cur_df = X_cur.copy()
    cur_df["target"]     = np.expm1(y_cur.values)
    cur_df["prediction"] = get_predictions(X_cur)

    col_map = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=X_ref.select_dtypes(include=[np.number]).columns.tolist(),
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_map)

    path = REPORTS_DIR / f"model_quality_{_ts()}.html"
    report.save_html(str(path))
    logger.info(f"Model Quality report --> {path.name}")
    return str(path)


# ──────────────────────────────────────────────
# Report 3: Target Drift
# ──────────────────────────────────────────────

def run_target_drift_report(y_ref: pd.Series, y_cur: pd.Series) -> str:
    """
    Detect if SalePrice distribution has shifted (target/label drift).
    This can indicate market changes that require model retraining.
    """
    logger.info("Running Target Drift analysis...")

    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import TargetDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

    ref_df = pd.DataFrame({"SalePrice": np.expm1(y_ref.values)})
    cur_df = pd.DataFrame({"SalePrice": np.expm1(y_cur.values)})

    col_map = ColumnMapping(target="SalePrice", prediction=None)

    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_map)

    path = REPORTS_DIR / f"target_drift_{_ts()}.html"
    report.save_html(str(path))
    logger.info(f"Target Drift report  --> {path.name}")
    return str(path)


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run_all_monitoring(simulate_drift: bool = False) -> dict:
    """
    Run all three monitoring reports.

    Args:
        simulate_drift: If True, introduces artificial feature drift so you can
                        see what a drift alert looks like in the HTML report.

    Returns:
        dict mapping report name -> absolute file path
    """
    sep = "=" * 55
    logger.info(sep)
    logger.info("  Starting Evidently AI Monitoring Suite (Stage 4)")
    if simulate_drift:
        logger.info("  [DRIFT SIMULATION MODE - artificial drift injected]")
    logger.info(sep)

    X_ref, y_ref = load_reference_data()
    X_cur, y_cur = load_current_data(simulate_drift=simulate_drift)

    reports = {}

    # Run Data Drift Report
    try:
        reports["data_drift"] = run_data_drift_report(X_ref, X_cur)
    except Exception as e:
        logger.error(f"data_drift report failed: {e}")

    # Run Model Quality Report
    try:
        reports["model_quality"] = run_model_quality_report(X_ref, y_ref, X_cur, y_cur)
    except Exception as e:
        logger.error(f"model_quality report failed: {e}")

    # Run Target Drift Report
    try:
        reports["target_drift"] = run_target_drift_report(y_ref, y_cur)
    except Exception as e:
        logger.error(f"target_drift report failed: {e}")

    logger.info(sep)
    logger.info("  All monitoring reports generated!")
    logger.info(f"  Saved to: {REPORTS_DIR}")
    for name, path in reports.items():
        logger.info(f"    {name}: {Path(path).name}")
    logger.info("  Open any .html file in your browser to view.")
    logger.info(sep)

    return reports


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently AI monitoring reports")
    parser.add_argument("--drift", action="store_true",
                        help="Simulate data drift for demonstration purposes")
    args = parser.parse_args()
    run_all_monitoring(simulate_drift=args.drift)
