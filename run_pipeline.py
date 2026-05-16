import logging
import sys
from pathlib import Path

# Ensure src module is discoverable
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.train import train_best_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run():
    logger.info("=== 1. Starting Data Ingestion ===")
    ingest_data()
    
    logger.info("=== 2. Starting Preprocessing ===")
    try:
        preprocess_data()
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return
    
    logger.info("=== 3. Starting Model Training ===")
    try:
        train_best_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    logger.info("=== Pipeline Completed Successfully! ===")

if __name__ == "__main__":
    run()
