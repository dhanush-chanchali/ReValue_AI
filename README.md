# ⚡ ReValue AI: Production-Grade ML Electronics Resale Valuation Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTest](https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org)
[![KaTeX Math](https://img.shields.io/badge/KaTeX-Math%20Engine-black.svg?logo=latex&logoColor=white)](https://katex.org)
[![Course Outcomes](https://img.shields.io/badge/Syllabus%20Coverage-CO--1%20to%20CO--6-purple.svg)]()

> **ReValue AI** is an end-to-end, production-grade Machine Learning system designed to eliminate information asymmetry in the secondhand electronics marketplace. It predicts fair resale prices for pre-owned smartphones and laptops using a dual-engine architecture: a **Histogram Gradient Boosting Champion Regressor ($R^2 = 0.9877$)** benchmarked against ensemble and regularized linear models, coupled with **Unsupervised K-Means++ clustering** for latent market tier discovery and **Isolation Forest** for live fraud/anomaly screening.

---

## 🌟 Key Highlights

- **🎯 Production Champion Model:** `HistGradientBoostingRegressor` achieves a Test MAE of **$27.44 (₹2,305)** and **$R^2 = 0.9877$**, statistically proven superior to linear baselines via Paired Student's t-test ($t = 16.82, p = 2.8 \times 10^{-49} \ll 0.01$).
- **🛡️ Unsupervised Fraud Guardrail:** `Isolation Forest` catches fraudulent or corrupted trade-in entries (e.g. ₹10,000 iPhone 16 Pro Max quotes) before model inference.
- **⚡ Low-Latency FastAPI Serving:** Asynchronous REST microservice with strict Pydantic payload validation, sub-10ms inference SLAs, and automatic Swagger docs.
- **📈 Continuous MLOps Telemetry:** SQLite inference logging, real-time covariate drift detection via **Two-Sample Kolmogorov-Smirnov (KS) tests**, and an automated Champion-Challenger retraining pipeline.
- **📘 Built-in Syllabus Coverage Suite:** In-app academic curriculum viewer featuring **CO-1 through CO-6 deep dives**, textbook-grade **KaTeX vector math rendering**, and an interactive **Algorithm Directory & Implementation Mapping**.

---

## 🤖 Algorithms Directory: Which Algorithm is Used Where & Why

The platform systematically benchmarks and deploys 11 algorithms across the ML lifecycle:

| # | Algorithm | Learning Paradigm | Code Location | Where Used in ReValue AI | Why It Is Used |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **HistGradientBoostingRegressor** | Supervised Regression (Boosting) | [`src/models_training.py:L140-155`](src/models_training.py#L140-L155) | **Production Champion Model** (`src/api.py` `POST /predict`, Tab 1 & Tab 2 Live Valuations) | Bins continuous features into 256 discrete bins, accelerating split search 10× over standard GBDT; shrinkage ($\eta=0.08$) yields our highest accuracy ($R^2 = 0.9877$, Test MAE = $27.44 / ₹2,305). |
| **2** | **RandomForestRegressor** | Supervised Regression (Bagging) | [`src/models_training.py:L125-138`](src/models_training.py#L125-L138) | **Benchmark Model** (Tab 3: ML Model Insights, `models/random_forest_pipeline.joblib`) | Trains 150 parallel deep trees with $\sqrt{p}$ feature subsampling; decorrelates individual trees to reduce ensemble variance without increasing bias ($R^2 = 0.9738$). |
| **3** | **Ridge Regression ($L_2$)** | Supervised Linear ($L_2$ Regularized) | [`src/models_training.py:L87-98`](src/models_training.py#L87-L98) | **Linear Baseline Model** (`models/ridge_l2_pipeline.joblib`, Tab 3 Benchmark) | Resolves severe hardware multicollinearity (RAM, storage, original retail price). Spherical penalty $\alpha \|\mathbf{w}\|_2^2$ makes $(X^T X + \alpha I)$ strictly invertible. |
| **4** | **Lasso Regression ($L_1$)** | Supervised Linear ($L_1$ Sparse) | [`src/models_training.py:L100-110`](src/models_training.py#L100-L110) | **Automated Feature Selection** (`models/lasso_l1_pipeline.joblib`, Tab 3 Benchmark) | Sharp corners of the diamond $L_1$ constraint drive non-informative coefficients to exactly zero via coordinate descent soft-thresholding. |
| **5** | **Ordinary Least Squares (OLS)** | Supervised Linear (Closed-Form) | [`src/models_training.py:L75-85`](src/models_training.py#L75-L85) | **Unregularized Normal Equation Baseline** (`models/linear_ols_pipeline.joblib`) | Solves $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$; serves as the unregularized reference baseline demonstrating why regularization is necessary. |
| **6** | **K-Means Clustering (K-Means++)** | Unsupervised Centroid Partitioning | [`src/unsupervised.py:L45-90`](src/unsupervised.py#L45-L90) | **Market Tier Segmentation** (Tab 3: ML Insights, `models/unsupervised_pipeline.joblib`) | Lloyd's algorithm with $D(x)^2$ K-Means++ seeding partitions inventory into $K=4$ latent market tiers (*Budget Entry, Daily Workhorse, Depreciated Flagship, Ultra-Premium*). |
| **7** | **Isolation Forest (iTrees)** | Unsupervised Anomaly Detection | [`src/unsupervised.py:L140-180`](src/unsupervised.py#L140-L180), [`src/api.py:L115`](src/api.py#L115) | **Live Fraud & Scam Guardrail** (`POST /predict`, Tab 1 & Tab 2 live diagnostics) | Outliers have extreme attributes and require fewer random tree splits ($\mathbb{E}[h(x)]$) to isolate. Contamination rate = 0.04 catches bogus quote submissions. |
| **8** | **Principal Component Analysis (PCA)** | Unsupervised Dimensionality Reduction | [`src/unsupervised.py:L115-138`](src/unsupervised.py#L115-L138) | **Interactive Cluster Visualization** (Tab 3: 2D/3D Scatter Projection) | Computes eigenvectors of the covariance matrix, projecting 9 hardware dimensions into 2 orthogonal axes capturing **67.7% of total variance**. |
| **9** | **Two-Sample Kolmogorov-Smirnov (KS)** | Non-Parametric Hypothesis Test | [`src/monitoring.py:L380-450`](src/monitoring.py#L380-L450), [`src/api.py:L160`](src/api.py#L160) | **Production Drift Telemetry** (Tab 4: MLOps Command Centre, `GET /monitor/drift`) | Calculates maximum empirical CDF deviation $D = \sup_x \|F_{\text{prod}}(x) - F_{\text{ref}}(x)\|$; triggers automated retraining when $p < 0.05$ on $\ge 2$ features. |
| **10** | **Paired Student's t-Test** | Inferential Hypothesis Testing | [`src/evaluation.py:L110-140`](src/evaluation.py#L110-L140) | **Statistical Superiority Verification** (`models/evaluation_diagnostics.json`) | Tests absolute sample errors between HistGBDT and Ridge ($t = 16.82, p = 2.8 \times 10^{-49} \ll 0.01$), formally proving superiority with scientific certainty. |
| **11** | **StandardScaler & Imputer Pipeline** | Data Preprocessing Pipeline | [`src/features.py:L95-120`](src/features.py#L95-L120) | **Atomic ColumnTransformer Pipeline** (`models/best_model_pipeline.joblib`) | Standardizes continuous features ($\mu=0, \sigma=1$) for isotropic penalization and median imputation; packaged into a single pipeline to eliminate training-serving skew. |

---

## 📑 Course Outcomes Syllabus Mapping (CO-1 to CO-6)

| Course Outcome | Academic Topic in Syllabus | ReValue AI Code Implementation | Key Deliverables & Files |
| :--- | :--- | :--- | :--- |
| **CO-1** | **The ML Lifecycle & System Architecture** | Ingestion, data contracts, strict 70/15/15 3-way split, zero training-serving skew architecture. | [`src/data_pipeline.py`](src/data_pipeline.py), [`src/features.py`](src/features.py), [`src/api.py`](src/api.py) |
| **CO-2** | **Supervised Learning — Linear Models at Depth** | OLS normal equations, Ridge ($L_2$), Lasso ($L_1$), ElasticNet, StandardScaler, One-Hot & Ordinal Encoding. | [`src/features.py`](src/features.py), [`src/models_training.py`](src/models_training.py) |
| **CO-3** | **Supervised Learning — Tree-Based Models** | Random Forest (150 trees, $\sqrt{p}$ subsampling, OOB score), HistGradientBoosting (boosting, histogram binning, shrinkage $\eta=0.08$). | [`src/models_training.py`](src/models_training.py) |
| **CO-4** | **Unsupervised Learning** | K-Means++ clustering ($K=4$), Elbow & Silhouette curve evaluation, 2D PCA projection (67.7% variance), Isolation Forest fraud audit. | [`src/unsupervised.py`](src/unsupervised.py) |
| **CO-5** | **Model Evaluation, Selection, Calibration** | 5-Fold Cross-Validation, MAE, RMSE, MAPE, $R^2$, Residual homoscedasticity diagnostics, empirical 90% confidence bounds, Paired Student's t-test. | [`src/evaluation.py`](src/evaluation.py) |
| **CO-6** | **ML Engineering — From Notebook to Production** | FastAPI REST endpoints (`/predict`, `/predict/batch`), SQLite production telemetry, Two-Sample KS data drift telemetry, Champion-Challenger automated retraining loop. | [`src/api.py`](src/api.py), [`src/monitoring.py`](src/monitoring.py), [`src/retrain.py`](src/retrain.py) |

---

## 🏆 Empirical Benchmark Results

Evaluated on held-out **Test Set** (15% held-out; $N = 375$ devices):

| Model | 5-Fold CV MAE | Test MAE ($) | Test MAE (₹) | Test RMSE ($) | Test MAPE (%) | Test $R^2$ Score | Statistical Significance vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **$29.82** | **$27.44** | **₹2,305** | **$38.16** | **5.41%** | **0.9877** | **$p = 2.8 \times 10^{-49}$ (Beats Ridge by 66.1%)** |
| **Random Forest** | $45.99 | $41.19 | ₹3,460 | $56.24 | 8.12% | 0.9738 | $p = 3.1 \times 10^{-28}$ (Beats Ridge by 49.1%) |
| **Ridge Regression ($L_2$)** | $84.32 | $80.93 | ₹6,798 | $107.50 | 17.84% | 0.9048 | Baseline Regularized Model ($\alpha = 1.0$) |
| **Linear Regression (OLS)** | $84.39 | $80.90 | ₹6,796 | $107.31 | 17.80% | 0.9052 | Unregularized Normal Equation Baseline |
| **Lasso Regression ($L_1$)** | $84.68 | $82.02 | ₹6,890 | $108.31 | 18.25% | 0.9034 | Feature Sparsity Inducing Baseline |
| **ElasticNet** | $125.62 | $117.30 | ₹9,853 | $153.62 | 26.50% | 0.8060 | Convex combination ($l_1\text{-ratio} = 0.5$) |

> **Key Finding:** Ensembles outperform linear models by over **$53.49 per device (66.1% error reduction)**. Electronics price curves drop non-linearly (e.g. sub-80% battery cliff), where axis-aligned tree splits naturally capture discontinuous degradation boundaries.

---

## 📂 Repository Architecture

```
├── .gitignore                    # Excludes virtual envs, caches, bytecodes, and OS artifacts
├── .streamlit/                   # Streamlit server and theme configurations
├── data/
│   ├── raw/                      # Raw smartphone & laptop specification datasets
│   ├── processed/                # Strict 70/15/15 splits (train.csv, val.csv, test.csv)
│   └── prediction_logs.db        # SQLite database logging production inference telemetry
├── models/                       # Serialized Scikit-Learn pipelines & benchmark metadata
│   ├── best_model_pipeline.joblib        # Champion HistGBDT end-to-end pipeline
│   ├── random_forest_pipeline.joblib     # Random Forest benchmark pipeline
│   ├── ridge_l2_pipeline.joblib          # Ridge L2 baseline pipeline
│   ├── lasso_l1_pipeline.joblib          # Lasso L1 baseline pipeline
│   ├── linear_ols_pipeline.joblib        # Ordinary Least Squares pipeline
│   ├── elasticnet_pipeline.joblib        # ElasticNet pipeline
│   ├── unsupervised_pipeline.joblib      # Fitted K-Means, PCA, & Isolation Forest
│   ├── models_benchmark.json             # Cross-validation & held-out test metrics
│   ├── evaluation_diagnostics.json       # Paired t-test results & residual diagnostics
│   └── latest_drift_report.json          # Production KS-test telemetry summary
├── notebooks/
│   └── revalue_ai_walkthrough.py         # Exploratory data analysis & prototype training
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # CO-1: Data generation, validation contracts, 3-way partition
│   ├── features.py               # CO-1, CO-2: Preprocessing, scaling, encodings, skew prevention
│   ├── models_training.py        # CO-2, CO-3: Linear & tree model training + 5-fold CV
│   ├── unsupervised.py           # CO-4: K-Means (Elbow/Silhouette), PCA, Isolation Forest
│   ├── evaluation.py             # CO-5: Metrics, residual plots, prediction bounds, Paired t-test
│   ├── api.py                    # CO-6: FastAPI asynchronous REST microservice
│   ├── monitoring.py             # CO-6: SQLite telemetry logging & two-sample KS drift testing
│   ├── retrain.py                # CO-6: Automated Champion-Challenger retraining pipeline
│   ├── wizards.py                # 3-Screen step-by-step Smartphone & Laptop trade-in wizards
│   ├── syllabus_view.py          # Academic Syllabus Coverage page with KaTeX vector math
│   └── dashboard.py              # Main Streamlit application with custom high-contrast UI
├── tests/
│   └── test_pipeline.py          # 7/7 passing unit & integration test suite
├── PROJECT_REVIEW_CHEAT_SHEET.md # Review & viva defense cheat sheet for academic presentations
├── requirements.txt              # Production dependency specifications
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and initialize the Python virtual environment:
```bash
# Clone the repository
git clone https://github.com/dhanush-chanchali/<your-repo-name>.git
cd <your-repo-name>

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
Validate the complete pipeline, API endpoints, drift detection, and monitoring health:
```bash
PYTHONPATH=. pytest tests/ -v
```
*(All 7 unit and integration tests will execute and pass)*

### 3. Launch FastAPI Serving Microservice
Start the asynchronous REST API backend:
```bash
uvicorn src.api:app --reload --port 8000
```
- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)
- **Data Drift Telemetry Endpoint:** [http://localhost:8000/monitor/drift](http://localhost:8000/monitor/drift)

### 4. Launch Streamlit Interactive Studio
In a separate terminal, launch the dashboard:
```bash
source venv/bin/activate
streamlit run src/dashboard.py
```
Opens in your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 🖥️ Interactive Dashboard Walkthrough

The platform features 5 comprehensive tabs:

1. **📱 Tab 1: Sell Smartphone (3-Screen Step-by-Step Wizard)**
   - Screen 1: Brand & Model Selection with **release-year validation** (e.g. iPhone 16 Pro Max only permits launch year $\ge 2024$).
   - Screen 2: Hardware specs (RAM, Storage, Screen size) & battery health slider with high-contrast indicator.
   - Screen 3: Physical cosmetic condition & packaging checklist with live dynamic pricing.
   - Instant calculation of resale value (USD & INR), 90% confidence bounds, and Isolation Forest fraud screening.

2. **💻 Tab 2: Sell Laptop (Step-by-Step Wizard)**
   - Screen 1: Brand, model, and purchase year.
   - Screen 2: Processor tier, RAM, SSD storage, screen size, battery health.
   - Screen 3: Condition grading, original charger, cosmetic wear flags, and live valuation.

3. **📊 Tab 3: ML Model Insights**
   - Full performance benchmarking table across all 6 models.
   - Residual error distribution plots and parity scatter plots ($y$ vs $\hat{y}$).
   - Feature importance rankings (MDI & Permutation).
   - Unsupervised 2D/3D PCA cluster visualization and K-Means market tier breakdown.

4. **🛡️ Tab 4: MLOps Command Centre**
   - Real-time latency statistics and SQLite query inspection.
   - Automated **Two-Sample Kolmogorov-Smirnov (KS) Data Drift Telemetry**.
   - One-click trigger for Champion-Challenger automated retraining loop.

5. **📘 Tab 5: Syllabus Coverage**
   - Interactive dropdown inspector for all 20 specific curriculum concepts with **crisp KaTeX vector mathematical formulations**.
   - Dedicated tab: **`🤖 Algorithms & Where Used`** mapping all 11 algorithms to exact code files and engineering rationales.
   - Module walkthroughs for **CO-1 through CO-6**.

---

## 📡 REST API Reference

### `POST /predict`
Estimates resale value for a single device with confidence intervals and anomaly detection:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "device_type": "Smartphone",
       "brand": "Apple",
       "model_name": "iPhone 15 Pro",
       "os": "iOS",
       "original_price_usd": 999.0,
       "age_months": 12,
       "ram_gb": 8,
       "storage_gb": 256,
       "screen_size_inch": 6.1,
       "battery_health_pct": 89.0,
       "condition": "Good",
       "has_original_box": 1,
       "has_original_charger": 1,
       "has_screen_damage": 0,
       "has_body_scratches": 0
     }'
```

**Response:**
```json
{
  "predicted_resale_price_usd": 684.25,
  "confidence_interval_90_usd": [644.25, 724.25],
  "depreciation_rate_pct": 31.51,
  "market_tier": "Depreciated Flagship",
  "is_anomaly": false,
  "anomaly_status": "Listing parameters are consistent with verified market listings.",
  "latency_ms": 2.14,
  "active_model": "HistGradientBoostingRegressor"
}
```

---

## 🧪 Automated Testing

The repository contains an automated unit and integration test suite:
- `test_api_health`: Verifies FastAPI server status and model loading.
- `test_api_predict_smartphone`: End-to-end inference for mobile devices.
- `test_api_predict_laptop`: End-to-end inference for laptops.
- `test_api_batch_prediction`: Vectorized multi-device inference.
- `test_drift_detection_logic`: Two-sample KS test statistical drift telemetry.
- `test_monitoring_latency_stats`: Latency percentile tracking.
- `test_monitoring_health_summary`: SQLite database connectivity and logging integrity.

Run the suite with:
```bash
pytest tests/ -v
```

---

## 👨‍💻 Author & Acknowledgments

- **Lead Engineer:** [Dhanush Chanchali](https://github.com/dhanush-chanchali) (`dhanush.2482006@gmail.com`)
- **Project:** ReValue AI — ML Electronics Resale Valuation Platform
- **Course Alignment:** Modules CO-1 through CO-6 (Machine Learning Lifecycle to Production Engineering)


deployed URL

https://revalueai-xhrrkbmbuexrswr3ermwnf.streamlit.app/

