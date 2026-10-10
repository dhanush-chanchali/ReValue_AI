"""
Module 6 (M1 & M6): FastAPI Serving Application
Exposes REST endpoints for online and batch resale price predictions,
model explainability, anomaly auditing, and real-time drift monitoring.
"""

import os
import sys
import time
import joblib
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring import log_prediction, get_recent_logs, detect_data_drift

app = FastAPI(
    title="ReValue AI - Prediction & Valuation Service",
    description="Production-grade ML prediction endpoint for smartphone and laptop resale prices.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and metadata
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

best_pipeline = None
unsupervised_pipeline = None
benchmark_metadata = {}
evaluation_metadata = {}
reference_train_df = None


def load_artifacts():
    global best_pipeline, unsupervised_pipeline, benchmark_metadata, evaluation_metadata, reference_train_df
    try:
        best_pipe_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
        if os.path.exists(best_pipe_path):
            best_pipeline = joblib.load(best_pipe_path)

        unsup_path = os.path.join(MODELS_DIR, "unsupervised_pipeline.joblib")
        if os.path.exists(unsup_path):
            unsupervised_pipeline = joblib.load(unsup_path)

        bench_path = os.path.join(MODELS_DIR, "models_benchmark.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r") as f:
                benchmark_metadata = json.load(f)

        eval_path = os.path.join(MODELS_DIR, "evaluation_diagnostics.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                evaluation_metadata = json.load(f)

        ref_train_path = os.path.join(DATA_DIR, "processed", "train.csv")
        if os.path.exists(ref_train_path):
            reference_train_df = pd.read_csv(ref_train_path)

    except Exception as e:
        print(f"Warning during artifact loading: {e}")


# Load artifacts immediately on module import
load_artifacts()


@app.on_event("startup")
def startup_event():
    load_artifacts()


# Pydantic Schemas with strict input validation (M6)
class DeviceInputRequest(BaseModel):
    device_type: str = Field(default="Smartphone")
    brand: str = Field(default="Apple")
    model_name: str = Field(default="iPhone 13")
    os: str = Field(default="iOS")
    original_price_usd: float = Field(default=999.0, gt=0)
    age_months: int = Field(default=18, ge=0, le=120)
    ram_gb: int = Field(default=6, ge=2, le=128)
    storage_gb: int = Field(default=128, ge=16, le=4096)
    screen_size_inch: float = Field(default=6.1, ge=4.0, le=20.0)
    battery_health_pct: float = Field(default=86.0, ge=30.0, le=100.0)
    condition: str = Field(default="Good")
    has_original_box: int = Field(default=1, ge=0, le=1)
    has_original_charger: int = Field(default=1, ge=0, le=1)
    has_screen_damage: int = Field(default=0, ge=0, le=1)
    has_body_scratches: int = Field(default=0, ge=0, le=1)

    @field_validator("condition")
    def validate_condition(cls, v):
        valid = ["Pristine", "Good", "Fair", "Heavily Used"]
        if v not in valid:
            raise ValueError(f"Condition must be one of {valid}")
        return v


class ValuationResponse(BaseModel):
    predicted_resale_price_usd: float
    confidence_interval_90_usd: List[float]
    depreciation_rate_pct: float
    market_tier: str
    is_anomaly: bool
    anomaly_status: str
    latency_ms: float
    active_model: str


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "ReValue AI Prediction API",
        "status": "online",
        "version": "1.0.0",
        "active_model": benchmark_metadata.get("best_model", "Gradient_Boosting"),
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    if best_pipeline is None:
        load_artifacts()
    return {
        "status": "healthy",
        "model_loaded": best_pipeline is not None,
        "unsupervised_loaded": unsupervised_pipeline is not None,
        "reference_data_loaded": reference_train_df is not None
    }


@app.post("/predict", response_model=ValuationResponse, tags=["Inference"])
def predict_device_resale(request: DeviceInputRequest):
    """
    End-to-End Prediction Endpoint (M1 Tracing):
    HTTP Request -> Pydantic Validation -> Feature Engineering & ColumnTransform ->
    Model Inference -> Prediction Interval Offset -> Anomaly Detection -> SQLite Logging.
    """
    if best_pipeline is None:
        load_artifacts()
        if best_pipeline is None:
            raise HTTPException(status_code=503, detail="Trained model pipeline not loaded.")

    start_time = time.time()
    input_dict = request.model_dump()
    df_single = pd.DataFrame([input_dict])

    # 1. Inference via trained scikit-learn pipeline (prevents train-serving skew)
    pred_val = float(best_pipeline.predict(df_single)[0])
    pred_price = round(max(25.0, pred_val), 2)

    # 2. Prediction Interval (M5)
    offset = 40.0
    if evaluation_metadata and "prediction_intervals" in evaluation_metadata:
        offset = abs(evaluation_metadata["prediction_intervals"].get("upper_bound_offset", 40.0))
    lower_bound = round(max(20.0, pred_price - offset), 2)
    upper_bound = round(min(request.original_price_usd, pred_price + offset), 2)

    # 3. Depreciation Percentage
    orig = request.original_price_usd
    deprec_pct = round(((orig - pred_price) / orig) * 100.0, 1)

    # 4. Unsupervised Segmentation & Anomaly Audit (M4)
    market_tier = "Standard Market Tier"
    is_anomaly = False
    anomaly_status = "Listing parameters are consistent with verified market listings."

    if unsupervised_pipeline is not None:
        try:
            unsup_features = [
                "original_price_usd", "age_months", "ram_gb", "storage_gb",
                "battery_health_pct", "screen_size_inch",
                "has_screen_damage", "has_body_scratches"
            ]
            feature_row = [input_dict[f] for f in unsup_features] + [pred_price]
            
            # Transform with unsupervised scaler
            imputed_row = unsupervised_pipeline["imputer"].transform([feature_row])
            scaled_row = unsupervised_pipeline["scaler"].transform(imputed_row)
            
            # Predict Cluster
            cluster_id = int(unsupervised_pipeline["kmeans"].predict(scaled_row)[0])
            market_tier = unsupervised_pipeline["cluster_to_tier"].get(cluster_id, "Market Segment")

            # Predict Anomaly via Isolation Forest
            iso_flag = int(unsupervised_pipeline["isolation_forest"].predict(scaled_row)[0])
            if iso_flag == -1:
                is_anomaly = True
                anomaly_status = "⚠️ Outlier / Suspicious Listing: Specs, condition, or price diverge significantly from standard market patterns."
        except Exception as e:
            print(f"Error in unsupervised audit: {e}")

    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    # 5. Production Request Logging for Drift Tracking (M1 & M6)
    try:
        log_prediction(
            input_data=input_dict,
            predicted_price=pred_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_anomaly=1 if is_anomaly else 0,
            latency_ms=latency_ms
        )
    except Exception as e:
        print(f"Logging error: {e}")

    return ValuationResponse(
        predicted_resale_price_usd=pred_price,
        confidence_interval_90_usd=[lower_bound, upper_bound],
        depreciation_rate_pct=deprec_pct,
        market_tier=market_tier,
        is_anomaly=is_anomaly,
        anomaly_status=anomaly_status,
        latency_ms=latency_ms,
        active_model=benchmark_metadata.get("best_model", "Random_Forest")
    )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch_devices(devices: List[DeviceInputRequest]):
    """Batch prediction endpoint for fleets or inventory audits."""
    if best_pipeline is None:
        load_artifacts()
    
    if not devices:
        return {"predictions": []}

    df_batch = pd.DataFrame([d.model_dump() for d in devices])
    preds = best_pipeline.predict(df_batch)
    
    results = []
    for d, p in zip(devices, preds):
        p_clean = round(max(25.0, float(p)), 2)
        results.append({
            "brand": d.brand,
            "model_name": d.model_name,
            "original_price_usd": d.original_price_usd,
            "predicted_resale_price_usd": p_clean,
            "depreciation_pct": round(((d.original_price_usd - p_clean) / d.original_price_usd) * 100.0, 1)
        })

    return {"count": len(results), "batch_predictions": results}


@app.get("/models/benchmark", tags=["Models"])
def get_models_benchmark():
    """Returns comparative metrics across Ridge, Lasso, Random Forest, Gradient Boosting (M2, M3, M5)."""
    return benchmark_metadata


@app.get("/evaluation/diagnostics", tags=["Evaluation"])
def get_evaluation_diagnostics():
    """Returns M5 statistical significance tests and learning curve data."""
    return evaluation_metadata


@app.get("/monitor/drift", tags=["Monitoring"])
def check_drift():
    """Evaluates covariate shift between reference training data and live logs (M6)."""
    logs = get_recent_logs(limit=200)
    if not logs or reference_train_df is None:
        return {"status": "No prediction logs or reference data available to compute drift"}
    
    current_df = pd.DataFrame(logs)
    report = detect_data_drift(reference_train_df, current_df)
    return report


@app.get("/monitor/logs", tags=["Monitoring"])
def view_logs(limit: int = 50):
    """Retrieves recent production inference logs."""
    return get_recent_logs(limit=limit)
