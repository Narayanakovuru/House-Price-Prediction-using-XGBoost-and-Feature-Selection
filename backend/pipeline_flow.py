"""
Ames Housing MLOps Pipeline — Prefect Flow (All 4 Stages)
==========================================================
Stage 1: Data Versioning    (DVC — run manually after pipeline)
Stage 2: Experiment Tracking (MLflow + Optuna — inside train task)
Stage 3: Orchestration       (Prefect — this file)
Stage 4: Monitoring          (Evidently AI — monitoring task)

Run once:    python pipeline_flow.py
Dashboard:   prefect server start  (then open http://localhost:4200)
MLflow UI:   mlflow ui             (then open http://localhost:5000)
Schedule:    See the schedule= argument in ames_housing_pipeline()
"""

import sys
import logging
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

# ──────────────────────────────────────────────
# Tasks — each wraps one of your existing src/ functions
# ──────────────────────────────────────────────

@task(
    name="Data Ingestion",
    description="Fetches the Ames Housing dataset from OpenML and saves it as raw CSV.",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),  # Won't re-download if run within 24h
)
def ingest_task():
    logger = get_run_logger()
    logger.info("Starting data ingestion...")
    from src.data_ingestion import ingest_data
    ingest_data()
    logger.info("✅ Data ingestion complete.")


@task(
    name="Data Preprocessing",
    description="Cleans, encodes, and splits the data. Saves processed parquet files + preprocessor.",
    retries=2,
    retry_delay_seconds=15,
)
def preprocess_task():
    logger = get_run_logger()
    logger.info("Starting preprocessing...")
    from src.preprocessing import preprocess_data
    preprocess_data()
    logger.info("✅ Preprocessing complete.")


@task(
    name="Model Training (Optuna + XGBoost)",
    description="Runs Optuna hyperparameter search, trains best XGBoost model, logs to MLflow.",
    retries=1,
    retry_delay_seconds=10,
)
def train_task():
    logger = get_run_logger()
    logger.info("Starting model training with Optuna + MLflow...")
    from src.train import train_best_model
    train_best_model()
    logger.info("✅ Model training complete. Check MLflow UI at http://localhost:5000")


@task(
    name="Model & Data Monitoring (Evidently AI)",
    description="Stage 4: Runs Evidently AI to detect data drift, model degradation, and target drift. Saves HTML reports to reports/ directory.",
    retries=1,
    retry_delay_seconds=5,
)
def monitoring_task():
    logger = get_run_logger()
    logger.info("Starting Evidently AI monitoring...")
    from src.monitoring import run_all_monitoring
    reports = run_all_monitoring(simulate_drift=False)
    for name, path in reports.items():
        logger.info(f"  📊 {name}: {path}")
    logger.info("✅ Monitoring complete. Open HTML reports in reports/ directory.")


# ──────────────────────────────────────────────
# Flow — the pipeline that connects all tasks
# ──────────────────────────────────────────────

@flow(
    name="Ames Housing ML Pipeline",
    description="Full 4-stage MLOps pipeline: Ingest → Preprocess → Train → Monitor",
    # Uncomment to schedule (e.g. run every day at midnight):
    # schedule=CronSchedule(cron="0 0 * * *"),
)
def ames_housing_pipeline():
    logger = get_run_logger()
    logger.info("🚀 Pipeline started! (4 stages)")

    # Stage 1 note: DVC versioning run separately (dvc add / dvc push)
    # Stages 2, 3, 4 run here automatically:

    ingest_task()       # Stage 3 Task 1: fetch data
    preprocess_task()   # Stage 3 Task 2: clean & encode
    train_task()        # Stage 2+3:      train + log to MLflow
    monitoring_task()   # Stage 4:        Evidently drift & performance reports

    logger.info("🎉 All 4 stages completed successfully!")
    logger.info("📊 MLflow UI:      mlflow ui           → http://localhost:5000")
    logger.info("📋 Prefect UI:     prefect server start → http://localhost:4200")
    logger.info("📂 Drift reports:  open reports/*.html in your browser")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ames_housing_pipeline()
