# ⚡ ReValue AI: ML-Powered Electronics Resale Price Valuation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Tests Passing](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)]()

> **ReValue AI** is a complete, production-grade machine learning system designed to estimate fair market resale prices for secondhand smartphones and laptops. Built from the ground up to cover Modules **M1 through M6** of the Machine Learning Lifecycle.

---

## 📑 Syllabus Module Coverage

| Module | Topic in Syllabus | ReValue AI Implementation | Key Files |
| :--- | :--- | :--- | :--- |
| **M1** | **The ML Lifecycle & System Architecture** | Complete end-to-end pipeline: raw data $\to$ preprocessing $\to$ training $\to$ model packaging $\to$ API serving $\to$ monitoring. Training-serving skew prevention. | `src/data_pipeline.py`, `src/features.py`, `src/api.py` |
| **M2** | **Supervised Learning — Linear Models at Depth** | OLS baseline, **Ridge ($L_2$)**, **Lasso ($L_1$)**, **ElasticNet**. StandardScaler, One-Hot Encoding, Ordinal Encoding, SimpleImputer. Linear coefficient interpretations. | `src/features.py`, `src/models_training.py` |
| **M3** | **Supervised Learning — Tree-Based Models** | **Random Forest** (bagging, feature subsampling, OOB score) and **HistGradientBoosting** (boosting, learning rate shrinkage). Feature importances (MDI, permutation). | `src/models_training.py` |
| **M4** | **Unsupervised Learning** | **K-Means clustering** (Lloyd's algorithm, K-Means++, Elbow method & Silhouette scores), **PCA** 2D/3D projections, and **Isolation Forest** anomaly/fraud detector. | `src/unsupervised.py` |
| **M5** | **Model Evaluation, Selection, Calibration** | 70/15/15 3-way split, 5-Fold Cross-Validation, MAE, RMSE, MAPE, $R^2$, Residual homoscedasticity diagnostics, empirical 90% prediction intervals, **Paired t-test** statistical significance. | `src/evaluation.py` |
| **M6** | **ML Engineering — Production to Serving** | **FastAPI** REST serving (`/predict`, `/predict/batch`), SQLite request logger, **Kolmogorov-Smirnov (KS) Data Drift detector**, and Champion-Challenger automated retraining loop. | `src/api.py`, `src/monitoring.py`, `src/retrain.py`, `src/dashboard.py` |

---

## 🏆 Model Benchmark Results

Evaluation on held-out **Test Set** (15% held-out):

| Model | 5-Fold CV MAE | Test MAE ($) | Test RMSE ($) | Test MAPE (%) | Test $R^2$ Score | Statistical Significance vs Ridge |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **$29.82** | **$27.44** | **$38.16** | **5.41%** | **0.9877** | **$p = 2.8 \times 10^{-49}$ (Beats Ridge by $53.50/dev)** |
| **Random Forest** | $45.99 | $41.19 | $56.24 | 8.12% | 0.9738 | $p = 3.1 \times 10^{-28}$ (Beats Ridge by $39.74/dev) |
| **Ridge Regression ($L_2$)** | $84.32 | $80.93 | $107.50 | 17.84% | 0.9048 | Baseline Regularized Model |
| **Linear Regression (OLS)** | $84.39 | $80.90 | $107.31 | 17.80% | 0.9052 | Baseline Ordinary Least Squares |
| **Lasso Regression ($L_1$)** | $84.68 | $82.02 | $108.31 | 18.25% | 0.9034 | Induces feature sparsity |
| **ElasticNet** | $125.62 | $117.30 | $153.62 | 26.50% | 0.8060 | Combined $L_1$ and $L_2$ penalties |

---

## 📂 Project Structure

```
├── data/
│   ├── raw/                      # Raw uncleaned dataset
│   ├── processed/                # Strict 70/15/15 splits (train.csv, val.csv, test.csv)
│   └── prediction_logs.db        # SQLite database logging production inference requests
├── models/                       # Packaged Scikit-Learn Pipelines & JSON metadata
│   ├── best_model_pipeline.joblib
│   ├── ridge_l2_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── gradient_boosting_pipeline.joblib
│   ├── unsupervised_pipeline.joblib
│   ├── models_benchmark.json
│   ├── unsupervised_summary.json
│   └── evaluation_diagnostics.json
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # M1: Data generation, validation, 3-way split
│   ├── features.py               # M2: Imputation, scaling, one-hot/ordinal encoding, skew prevention
│   ├── models_training.py        # M2, M3, M5: Linear & Tree model training + CV
│   ├── unsupervised.py           # M4: K-Means (elbow/silhouette), PCA, Isolation Forest
│   ├── evaluation.py             # M5: MAE/RMSE/R2, residuals, paired t-tests, intervals
│   ├── api.py                    # M1, M6: FastAPI REST API service
│   ├── monitoring.py             # M1, M6: Production logging, KS-test data drift detector
│   ├── retrain.py                # M6: Champion-Challenger automated retraining loop
│   └── dashboard.py              # Interactive multi-tab Streamlit dashboard
├── tests/
│   └── test_pipeline.py          # Unit & integration test suite (100% passing)
├── PROJECT_REVIEW_CHEAT_SHEET.md # Complete viva preparation guide for tomorrow's review
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
# Clone or navigate to the repository
cd "/Users/dhanuu/Desktop/ML PROJECT/ANTIGRAVITY IDE"

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
PYTHONPATH=. pytest tests/ -v
```
*(All 5 unit and integration tests will execute and pass)*

### 3. Launch FastAPI Server (REST API)
```bash
uvicorn src.api:app --reload --port 8000
```
- Open Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 4. Launch Streamlit Studio (Interactive Dashboard)
```bash
streamlit run src/dashboard.py
```
- Opens automatically in your browser at `http://localhost:8501`.
- Features 5 comprehensive tabs:
  1. **📱 Live Valuation Studio:** Enter specs and receive instant resale estimates, 90% confidence bands, future depreciation projections, and anomaly safety audit badges.
  2. **⚔️ Model Comparison Arena:** Full benchmark table, MAE/R² charts, parity plot, residual scatter, and hypothesis test metrics.
  3. **🔍 Interpretability:** Ridge standardized coefficients and Tree permutation importances.
  4. **🌐 Market Segments:** K-Means elbow & silhouette curve, 2D PCA scatter map, and Isolation Forest outliers.
  5. **⚙️ MLOps Lifecycle:** System architecture diagram, real-time inference request logs, and interactive KS-Test drift detector.

---

## 📡 REST API Reference

### `POST /predict`
Estimates resale value for a single device.
```json
{
  "device_type": "Smartphone",
  "brand": "Apple",
  "model_name": "iPhone 13",
  "os": "iOS",
  "original_price_usd": 999.0,
  "age_months": 18,
  "ram_gb": 6,
  "storage_gb": 128,
  "screen_size_inch": 6.1,
  "battery_health_pct": 86.0,
  "condition": "Good",
  "has_original_box": 1,
  "has_original_charger": 1,
  "has_screen_damage": 0,
  "has_body_scratches": 0
}
```

**Response:**
```json
{
  "predicted_resale_price_usd": 512.45,
  "confidence_interval_90_usd": [472.45, 552.45],
  "depreciation_rate_pct": 48.7,
  "market_tier": "Mid-Range Daily Driver",
  "is_anomaly": false,
  "anomaly_status": "Listing parameters are consistent with verified market listings.",
  "latency_ms": 1.84,
  "active_model": "Gradient_Boosting"
}
```

---

## 💡 Review Presentation Tip
Open both terminals:
- **Terminal 1:** `uvicorn src.api:app --reload --port 8000`
- **Terminal 2:** `streamlit run src/dashboard.py`

Walk your evaluators through the **5 tabs in Streamlit**, reference **PROJECT_REVIEW_CHEAT_SHEET.md** for technical defense, and showcase the **Swagger UI at `http://localhost:8000/docs`** to demonstrate a true production-grade system!
