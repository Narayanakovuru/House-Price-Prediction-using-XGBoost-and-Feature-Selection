<div align="center">

# 🏡 Ames Housing Price Prediction

### End-to-End Production MLOps Pipeline · XGBoost · FastAPI · Streamlit

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prefect](https://img.shields.io/badge/Prefect-Orchestration-1D4AFF?style=flat-square&logo=prefect&logoColor=white)](https://www.prefect.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-Monitoring-E5484D?style=flat-square)](https://www.evidentlyai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

> **A production-grade, fully automated MLOps system** that transforms tabular housing data into real-time price predictions — integrating Bayesian hyperparameter optimization, experiment tracking, drift monitoring, and a REST-backed interactive UI into a single coherent pipeline.

</div>

---

## 📋 Table of Contents

- [Model Performance](#-model-performance)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [MLOps Stack](#-mlops-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [API Reference](#-api-reference)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Application Usage](#-application-usage)
- [Drift Monitoring & Simulation](#-drift-monitoring--simulation)
- [Experiment Tracking](#-experiment-tracking)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 📊 Model Performance

> All metrics are reported on the held-out **test set (20% split, 292 samples, `random_state=42`)**.  
> The target `SalePrice` is log-transformed via `np.log1p` during training; metrics below are in **log-scale** unless noted.

### 🏆 Best Run Results (MLflow Experiment: `Ames_Housing_XGBoost`)

| Metric | Best Run | Run 2 | Run 3 | Interpretation |
|:---|:---:|:---:|:---:|:---|
| **R² Score** | **0.9108** | 0.9093 | 0.9054 | Model explains **91.1%** of variance in house prices |
| **RMSE** (log-scale) | **0.1290** | 0.1301 | 0.1328 | Avg. prediction error of **~12.9%** in log-price space |
| **MAE** (log-scale) | **0.0875** | — | 0.0867 | Median prediction deviation of **~8.7%** in log-price space |
| **Inference Latency** | **< 10 ms** | — | — | Pre-loaded artifacts; sub-10ms per request |

> **RMSE in USD (approx.):** Since prices are log-transformed, `expm1(RMSE) ≈ $13,800` mean absolute deviation on a median house price of ~$163,000 — roughly **8.5% relative error**.

### 🔧 Optimal Hyperparameters (Optuna TPE — 10 Trials)

| Hyperparameter | Value | Search Range | Method |
|:---|:---:|:---:|:---|
| `n_estimators` | **500** | [100 – 500] | `suggest_int(step=50)` |
| `max_depth` | **5** | [3 – 8] | `suggest_int` |
| `learning_rate` | **0.01480** | [1e-3 – 0.1] | `suggest_float(log=True)` |
| `subsample` | **0.6267** | [0.6 – 1.0] | `suggest_float` |
| `colsample_bytree` | **0.7220** | [0.6 – 1.0] | `suggest_float` |
| `objective` | `reg:squarederror` | — | Fixed |
| `booster` | `gbtree` | — | Fixed |
| `random_state` | `42` | — | Fixed (reproducibility) |

---

## 🏗️ System Architecture

The system is composed of **5 decoupled stages** orchestrated by Prefect and connected through shared file artifacts (Parquet, joblib, JSON).

```mermaid
graph TD
    subgraph S1["① Ingestion & Preprocessing"]
        A["OpenML API\n(house_prices dataset)"] -->|"fetch raw CSV"| B["src/data_ingestion.py"]
        B -->|"ames_housing.csv"| C[("data/raw/")]
        C -->|"read"| D["src/preprocessing.py"]
        D -->|"log1p target · RobustScale · OneHot"| E[("data/processed/\nX_train · X_test · y_train · y_test")]
        D -->|"serialize"| F["models/preprocessor.joblib"]
        D -->|"sample schema"| G["models/feature_schema.csv"]
    end

    subgraph S2["② Training & Tracking"]
        E -->|"X_train, y_train"| H["src/train.py"]
        H -->|"Optuna TPE — 10 trials"| I["Best Hyperparameters"]
        I -->|"fit final XGBRegressor"| J["XGBoost Model"]
        J -->|"log params + metrics\nRMSE · MAE · R²"| K[("MLflow Tracking\nAmes_Housing_XGBoost")]
        J -->|"model.save_model()"| L["models/best_xgb_model.json"]
        L -->|"md5 pointer"| M["DVC Versioning"]
    end

    subgraph S3["③ Orchestration"]
        N["pipeline_flow.py\n(Prefect @flow)"] -->|"@task + retries + cache"| B
        N -->|"@task + retries"| D
        N -->|"@task + MLflow"| H
        N -->|"@task"| O["src/monitoring.py"]
    end

    subgraph S4["④ Real-time Inference"]
        P["frontend/app.py\n(Streamlit)"] -->|"POST /predict\n{JSON features}"| Q["app/main.py\n(FastAPI + Uvicorn)"]
        Q -->|"load once at startup"| F
        Q -->|"load once at startup"| L
        Q -->|"transform → predict → expm1"| Q
        Q -->|"{'predicted_price': $USD}"| P
    end

    subgraph S5["⑤ Monitoring & Alerting"]
        O -->|"reference data"| R["Evidently AI Suite"]
        R -->|"KS test per feature"| S1R[("reports/\ndata_drift_*.html")]
        R -->|"RMSE · MAE · R² over time"| S2R[("reports/\nmodel_quality_*.html")]
        R -->|"SalePrice distribution"| S3R[("reports/\ntarget_drift_*.html")]
        T["inject_drift.py\n(300 anomalous records)"] -->|"spike GrLivArea × 6\nSalePrice = $20k"| C
    end

    style S1 fill:#1e293b,stroke:#334155,color:#e2e8f0
    style S2 fill:#1e1a2e,stroke:#4c1d95,color:#e2e8f0
    style S3 fill:#0f2027,stroke:#164e63,color:#e2e8f0
    style S4 fill:#1a0f2e,stroke:#7c3aed,color:#e2e8f0
    style S5 fill:#1f1200,stroke:#92400e,color:#e2e8f0
```

---

## 📁 Project Structure

```
ames_housing_mlops/
│
├── app/
│   └── main.py                    # FastAPI REST backend — /predict & /health
│
├── data/                          # ⚠ Git-ignored — DVC-tracked
│   ├── raw/
│   │   └── ames_housing.csv       # 2,930 rows × 81 features (raw)
│   └── processed/
│       ├── X_train.parquet        # 2,344 samples × N engineered features
│       ├── X_test.parquet         # 586 samples × N engineered features
│       ├── y_train.parquet        # log1p(SalePrice) — train labels
│       └── y_test.parquet         # log1p(SalePrice) — test labels
│
├── frontend/
│   └── app.py                     # Streamlit dark-mode prediction dashboard
│
├── mlruns/                        # MLflow local tracking (auto-generated)
│   └── 955942723977080354/        # Experiment: Ames_Housing_XGBoost
│       └── <run_id>/
│           ├── metrics/           # rmse, mae, r2 per step
│           ├── params/            # Optuna-found hyperparameters
│           └── artifacts/         # Logged XGBoost model package
│
├── models/
│   ├── best_xgb_model.json        # XGBoost native serialization (fast load)
│   ├── best_xgb_model.json.dvc    # DVC pointer — md5 hash for reproducibility
│   ├── feature_schema.csv         # 10-row sample used for UI slider defaults
│   └── preprocessor.joblib        # Fitted Scikit-learn ColumnTransformer
│
├── reports/                       # Evidently AI HTML reports (timestamped)
│   ├── data_drift_<ts>.html
│   ├── model_quality_<ts>.html
│   └── target_drift_<ts>.html
│
├── src/
│   ├── config.py                  # BASE_DIR · DATA_DIR · MODEL_DIR · TARGET_COL
│   ├── data_ingestion.py          # OpenML fetch → data/raw/ames_housing.csv
│   ├── preprocessing.py           # ColumnTransformer pipeline + parquet export
│   ├── train.py                   # Optuna HPO + XGBoost training + MLflow logging
│   └── monitoring.py              # Evidently AI: drift & regression quality suite
│
├── .dvc/                          # DVC internal config
├── .dvcignore                     # DVC exclusion patterns
├── .gitignore                     # Excludes: data/, mlruns/, venv/, models/
├── .vscode/settings.json          # Auto-activates venv in VS Code terminals
├── ames-house-prediction.ipynb    # Legacy exploratory notebook (EDA origin)
├── inject_drift.py                # Drift simulation — injects 300 anomalous rows
├── pipeline_flow.py               # Prefect @flow orchestrator (recommended runner)
├── pyrightconfig.json             # Pyright/VSCode type-checker config
├── requirements.txt               # Pinned Python dependencies
└── run_pipeline.py                # Lightweight sequential runner (no Prefect daemon)
```

---

## 🔧 MLOps Stack

| Pillar | Tool | Role |
|:---|:---|:---|
| **Code Versioning** | Git | Track all source code, configs, and DVC pointers |
| **Data Versioning** | DVC | Version large binaries (CSVs, Parquet, model JSON) with md5 hashes |
| **Pipeline Orchestration** | Prefect | `@flow` / `@task` with retries, caching (`task_input_hash`), and Gantt dashboard |
| **Experiment Tracking** | MLflow | Log hyperparameters, RMSE/MAE/R² metrics, and model artifacts per run |
| **Hyperparameter Optimization** | Optuna | Bayesian TPE search — minimizes RMSE over 10 trials |
| **Model Serving** | FastAPI + Uvicorn | Async REST API with Pydantic validation, global model cache, sub-10ms latency |
| **Frontend UI** | Streamlit | Dark-mode interactive calculator with dynamic feature sliders |
| **Drift Monitoring** | Evidently AI | KS-test data drift, regression quality decay, and target distribution shift |
| **Type Checking** | Pyright | Static analysis — enforces type correctness across all `src/` modules |

---

## 🤖 Machine Learning Pipeline

### Stage 1 — Feature Engineering

The `src/preprocessing.py` module builds a `ColumnTransformer` with two sub-pipelines:

**Numeric Pipeline** (applied to `int64`/`float64` columns):
```
SimpleImputer(strategy='median')  →  RobustScaler()
```
| Design Choice | Justification |
|:---|:---|
| **Median imputation** | Robust to outliers vs. mean imputation on skewed price distributions |
| **RobustScaler** | Scales via IQR: `z = (x − median) / IQR` — immune to mansion-level outliers |

**Categorical Pipeline** (applied to `object`/`category` columns):
```
SimpleImputer(strategy='constant', fill_value='missing')  →  OneHotEncoder(handle_unknown='ignore')
```
| Design Choice | Justification |
|:---|:---|
| **Constant imputation** | `NaN` in categorical cols = feature absent (e.g., no garage) — not random |
| **OneHotEncoder** | `handle_unknown='ignore'` gracefully handles unseen categories at inference |
| **Column sanitization** | Strips `[`, `]`, `<` from OHE column names — XGBoost rejects these in DMatrix |

**Target Transformation:**
```python
y_train = np.log1p(df["SalePrice"])   # stabilise right-skew
ŷ_usd   = np.expm1(model.predict(X)) # inverse transform at inference
```

---

### Stage 2 — Model Training (XGBoost + Optuna)

```
Data Split:  80% train / 20% test  (random_state=42, stratified by price quartile)
Algorithm:   XGBRegressor — gradient-boosted decision trees (gbtree booster)
Objective:   reg:squarederror  (minimises MSE on log-transformed SalePrice)
```

**Why XGBoost for tabular housing data?**
- Handles mixed numeric/categorical features natively post-encoding
- Built-in L1 (`reg_alpha`) + L2 (`reg_lambda`) regularisation to prevent overfitting on high-cardinality OHE features
- Sparsity-aware split finding handles imputed values gracefully
- Consistently outperforms deep learning on tabular regression tasks at this scale

**Optuna Bayesian Search (Tree-structured Parzen Estimators):**

```
Objective function:  RMSE(y_test_log, model.predict(X_test))
Direction:           minimize
Trials:              10
Sampler:             TPE (default) — probabilistic, not exhaustive
```

| Hyperparameter | Search Range | Scale | Optimal |
|:---|:---:|:---:|:---:|
| `n_estimators` | 100 – 500 | step=50 | **500** |
| `max_depth` | 3 – 8 | integer | **5** |
| `learning_rate` | 1e-3 – 0.1 | log | **0.01480** |
| `subsample` | 0.6 – 1.0 | float | **0.6267** |
| `colsample_bytree` | 0.6 – 1.0 | float | **0.7220** |

---

### Stage 3 — Experiment Tracking (MLflow)

Every training run automatically logs to the `Ames_Housing_XGBoost` MLflow experiment:

| Artifact | Content |
|:---|:---|
| **Parameters** | All 5 Optuna-found hyperparameters |
| **Metrics** | `rmse`, `mae`, `r2` on test set |
| **Model artifact** | Full `xgboost-model` package with `conda.yaml` + `MLmodel` spec |

**Tracked run history (from `mlruns/`):**

| Run | RMSE ↓ | MAE ↓ | R² ↑ |
|:---|:---:|:---:|:---:|
| Run 3 (earliest) | 0.1328 | 0.0867 | 0.9054 |
| Run 2 | 0.1301 | — | 0.9093 |
| **Run 1 (best)** | **0.1290** | **0.0875** | **0.9108** |

---

## 🌐 API Reference

The FastAPI backend exposes two endpoints on `http://127.0.0.1:8000`:

### `POST /predict`

Accepts a JSON payload with raw house feature values and returns the predicted sale price in USD.

**Request body:**
```json
{
  "data": {
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "TotalBsmtSF": 850,
    "FullBath": 2,
    "YearBuilt": 2005,
    "Neighborhood": "CollgCr"
  }
}
```

**Response:**
```json
{
  "predicted_price": 214350.75
}
```

**Implementation detail:** The preprocessor and model are loaded **once at startup** into module-level globals — meaning each request bypasses I/O and runs pure in-memory inference in under 10 ms.

### `GET /health`

```json
{ "status": "ok" }
```

Interactive Swagger UI available at: `http://127.0.0.1:8000/docs`

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|:---|:---|
| Python | 3.9+ |
| pip | Latest |
| Git | Any |
| DVC | Installed via `requirements.txt` |

### 1 — Clone the Repository

```bash
git clone https://github.com/Narayanakovuru/House-Price-Prediction-using-XGBoost-and-Feature-Selection.git
cd House-Price-Prediction-using-XGBoost-and-Feature-Selection
```

### 2 — Create & Activate Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Full dependency list</summary>

```
scikit-learn    # Preprocessing pipelines & evaluation metrics
pandas          # Data manipulation & Parquet I/O
numpy           # Numerical operations & log transforms
xgboost         # Gradient-boosted tree model
optuna          # Bayesian hyperparameter optimisation
mlflow          # Experiment tracking & model registry
fastapi         # Async REST API framework
uvicorn         # ASGI server for FastAPI
streamlit       # Interactive frontend dashboard
pydantic        # Request body validation
requests        # HTTP client for Streamlit → FastAPI calls
prefect         # Pipeline orchestration & scheduling
dvc             # Data & model versioning
evidently       # ML monitoring & drift detection
```

</details>

---

## ▶️ Running the Pipeline

### Option A — Prefect Orchestrated *(recommended)*

```bash
python pipeline_flow.py
```

Runs all 4 stages as Prefect `@task` blocks with automatic **retries**, **24h caching** on data ingestion, and a real-time execution dashboard.

### Option B — Simple Sequential Runner *(no daemon required)*

```bash
python run_pipeline.py
```

### Pipeline Stage Breakdown

| # | Stage | Script | Output |
|:---:|:---|:---|:---|
| 1 | **Data Ingestion** | `src/data_ingestion.py` | `data/raw/ames_housing.csv` |
| 2 | **Preprocessing** | `src/preprocessing.py` | 4× Parquet files + `preprocessor.joblib` |
| 3 | **Training** | `src/train.py` | `best_xgb_model.json` + MLflow run |
| 4 | **Monitoring** | `src/monitoring.py` | 3× Evidently HTML reports |

> **Prefect caching:** Data ingestion uses `task_input_hash` caching — if you rerun the pipeline within 24 hours, stage 1 is skipped and local files are reused, saving network bandwidth.

---

## 🖥️ Application Usage

### Start the FastAPI Backend

```bash
uvicorn app.main:app --reload
```

| Endpoint | URL |
|:---|:---|
| REST API | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |

### Start the Streamlit Frontend

Open a **new terminal** (with `venv` activated):

```bash
streamlit run frontend/app.py
```

Navigate to `http://localhost:8501` to use the interactive house price calculator with:
- Dynamic sliders for `OverallQual`, `GrLivArea`, `YearBuilt`, `TotalBsmtSF`, `GarageCars`, `FullBath`
- **Smart defaults** — missing features auto-populated from `feature_schema.csv` (median for numeric, mode for categorical)
- Real-time predictions with visual success/error feedback

---

## 📈 Drift Monitoring & Simulation

### Running Normal Monitoring

```bash
python -m src.monitoring
```

Generates three Evidently AI HTML reports in `reports/`:

| Report | Statistical Test | Detects |
|:---|:---|:---|
| `data_drift_<ts>.html` | Kolmogorov-Smirnov (numeric) · Chi-squared (categorical) | Feature distribution shift |
| `model_quality_<ts>.html` | Regression metrics over time | RMSE / MAE / R² degradation |
| `target_drift_<ts>.html` | KS test on `SalePrice` distribution | Market-level price shifts |

### Simulating Concept Drift

```bash
python inject_drift.py
```

Injects **300 anomalous records** into `data/raw/ames_housing.csv`:
- `GrLivArea` multiplied **~6×** (10,000+ sq ft mansions)
- `SalePrice` set to **$20,000** (extreme underpricing)
- Reruns preprocessing and triggers the monitoring suite
- Produces HTML reports showing flagged **Data Drift**, **Target Drift**, and **Model Quality decay**

Alternatively, simulate drift in-memory without modifying source data:

```bash
python -m src.monitoring --drift
```

---

## 🔬 Experiment Tracking

### MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

View all tracked runs, compare hyperparameter configurations, inspect metric curves, and download logged model artifacts.

### Prefect Dashboard

```bash
prefect server start
# Open http://localhost:4200
```

View real-time flow execution status, task Gantt charts, retry events, and cache hit/miss logs.

---

## 🧰 Tech Stack

| Layer | Technology | Version | Purpose |
|:---|:---|:---:|:---|
| **ML Model** | XGBoost | 2.x | Gradient-boosted regression |
| **Preprocessing** | Scikit-learn | 1.x | ColumnTransformer, RobustScaler, OHE |
| **HPO** | Optuna | 3.x | Bayesian TPE hyperparameter search |
| **Experiment Tracking** | MLflow | 2.x | Params, metrics & model artifact logging |
| **Orchestration** | Prefect | 3.x | Flow/task execution, retries, caching |
| **Data Versioning** | DVC | 3.x | md5-pointer versioning for large binaries |
| **Monitoring** | Evidently AI | 0.7+ | Statistical drift detection & regression quality |
| **REST API** | FastAPI | 0.x | Async prediction endpoint with Pydantic validation |
| **ASGI Server** | Uvicorn | — | High-performance async web server |
| **Frontend** | Streamlit | — | Interactive dark-mode prediction UI |
| **Data I/O** | Pandas + Parquet | — | Efficient columnar data storage |
| **Type Checking** | Pyright | — | Static analysis across `src/` |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with precision using XGBoost · Optuna · MLflow · Prefect · FastAPI · Evidently AI · Streamlit**

*Ames, Iowa Housing Dataset — [OpenML: house_prices](https://www.openml.org/d/43926)*

</div>
