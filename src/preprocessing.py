import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import joblib
import logging
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COL, MODEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    logger.info("Loading raw data...")
    df = pd.read_csv(RAW_DATA_DIR / "ames_housing.csv")
    
    X = df.drop(columns=[TARGET_COL, 'Id'], errors='ignore')
    # Apply log1p to target for more stable training (common for price)
    y = np.log1p(df[TARGET_COL])
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean feature names for XGBoost since it does not like characters like '[', ']', '<'
        feature_names = [f.replace('[', '').replace(']', '').replace('<', '') for f in feature_names]
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        feature_names = [f"f_{i}" for i in range(X_train_processed.shape[1])]
    
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Save processed data
    logger.info("Saving processed datasets...")
    X_train_processed_df.to_parquet(PROCESSED_DATA_DIR / "X_train.parquet", index=False)
    X_test_processed_df.to_parquet(PROCESSED_DATA_DIR / "X_test.parquet", index=False)
    
    # Save y target
    y_train_df = pd.DataFrame({TARGET_COL: y_train})
    y_test_df = pd.DataFrame({TARGET_COL: y_test})
    y_train_df.to_parquet(PROCESSED_DATA_DIR / "y_train.parquet", index=False)
    y_test_df.to_parquet(PROCESSED_DATA_DIR / "y_test.parquet", index=False)
    
    # Save preprocessor
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    
    # Save feature structure so frontend can use it generically
    # We will just save a small sample of X (raw) to help Streamlit infer types and limits
    X.sample(10, random_state=42).to_csv(MODEL_DIR / "feature_schema.csv", index=False)
    
    logger.info("Preprocessing complete.")
    
if __name__ == "__main__":
    preprocess_data()
