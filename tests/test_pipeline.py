"""
Unit and Integration Tests for ReValue AI ML Pipeline and FastAPI Service
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Workspace paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import app
from src.features import build_preprocessor, get_feature_names
from src.monitoring import detect_data_drift

client = TestClient(app)


def test_api_health():
    """Verify API health endpoint responds with 200 OK and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_predict_smartphone():
    """Verify end-to-end prediction for a smartphone."""
    payload = {
        "device_type": "Smartphone",
        "brand": "Apple",
        "model_name": "iPhone 13",
        "os": "iOS",
        "original_price_usd": 899.0,
        "age_months": 16,
        "ram_gb": 6,
        "storage_gb": 128,
        "screen_size_inch": 6.1,
        "battery_health_pct": 87.0,
        "condition": "Good",
        "has_original_box": 1,
        "has_original_charger": 1,
        "has_screen_damage": 0,
        "has_body_scratches": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_resale_price_usd" in res
    assert 200.0 <= res["predicted_resale_price_usd"] <= 899.0
    assert "confidence_interval_90_usd" in res
    assert len(res["confidence_interval_90_usd"]) == 2
    assert res["is_anomaly"] in [True, False]
    assert res["latency_ms"] >= 0.0


def test_api_predict_laptop():
    """Verify end-to-end prediction for a laptop."""
    payload = {
        "device_type": "Laptop",
        "brand": "Dell",
        "model_name": "XPS 15",
        "os": "Windows",
        "original_price_usd": 1599.0,
        "age_months": 22,
        "ram_gb": 16,
        "storage_gb": 512,
        "screen_size_inch": 15.6,
        "battery_health_pct": 82.0,
        "condition": "Fair",
        "has_original_box": 0,
        "has_original_charger": 1,
        "has_screen_damage": 0,
        "has_body_scratches": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_resale_price_usd" in res
    assert 200.0 <= res["predicted_resale_price_usd"] <= 1599.0


def test_api_batch_prediction():
    """Verify batch prediction endpoint."""
    batch_payload = [
        {
            "device_type": "Smartphone",
            "brand": "Samsung",
            "model_name": "Galaxy S22",
            "os": "Android",
            "original_price_usd": 799.0,
            "age_months": 20,
            "ram_gb": 8,
            "storage_gb": 128,
            "screen_size_inch": 6.1,
            "battery_health_pct": 84.0,
            "condition": "Good",
            "has_original_box": 1,
            "has_original_charger": 1,
            "has_screen_damage": 0,
            "has_body_scratches": 0
        },
        {
            "device_type": "Laptop",
            "brand": "Apple",
            "model_name": "MacBook Air M2",
            "os": "macOS",
            "original_price_usd": 1199.0,
            "age_months": 12,
            "ram_gb": 8,
            "storage_gb": 256,
            "screen_size_inch": 13.6,
            "battery_health_pct": 94.0,
            "condition": "Pristine",
            "has_original_box": 1,
            "has_original_charger": 1,
            "has_screen_damage": 0,
            "has_body_scratches": 0
        }
    ]
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["count"] == 2
    assert len(res["batch_predictions"]) == 2


def test_drift_detection_logic():
    """Verify KS-test + PSI drift detector catches simulated distribution shift."""
    ref_df = pd.DataFrame({
        "original_price_usd": np.random.normal(1000, 200, 100),
        "age_months": np.random.normal(15, 5, 100),
        "ram_gb": np.random.choice([8, 16], 100),
        "storage_gb": np.random.choice([256, 512], 100),
        "battery_health_pct": np.random.normal(90, 5, 100),
        "device_type": np.random.choice(["Smartphone", "Laptop"], 100),
        "brand": np.random.choice(["Apple", "Samsung"], 100),
        "condition": np.random.choice(["Good", "Fair"], 100),
    })

    # Heavily drifted distribution
    drifted_df = pd.DataFrame({
        "original_price_usd": np.random.normal(2500, 300, 100),
        "age_months": np.random.normal(50, 5, 100),
        "ram_gb": np.random.choice([32, 64], 100),
        "storage_gb": np.random.choice([1024, 2048], 100),
        "battery_health_pct": np.random.normal(60, 5, 100),
        "device_type": np.random.choice(["Laptop"], 100),
        "brand": np.random.choice(["Dell", "HP"], 100),
        "condition": np.random.choice(["Heavily Used", "Fair"], 100),
    })

    from src.monitoring import detect_data_drift
    report = detect_data_drift(ref_df, drifted_df)
    assert report["ks_features_drifted"] >= 2
    assert report["retrain_recommended"] is True
    assert report["health_score"] < 80
    assert "feature_details" in report


def test_monitoring_latency_stats():
    """Verify latency statistics function returns expected shape."""
    from src.monitoring import compute_latency_stats, log_prediction
    # Log a few synthetic predictions to populate DB
    for i in range(5):
        log_prediction(
            {"device_type": "Smartphone", "brand": "Apple", "model_name": "iPhone 13",
             "os": "iOS", "original_price_usd": 999.0, "age_months": 18,
             "ram_gb": 6, "storage_gb": 128, "screen_size_inch": 6.1,
             "battery_health_pct": 88.0, "condition": "Good",
             "has_original_box": 1, "has_original_charger": 1,
             "has_screen_damage": 0, "has_body_scratches": 0},
            predicted_price=520.0, lower_bound=480.0, upper_bound=560.0,
            is_anomaly=0, latency_ms=12.5 + i * 2.0,
        )
    stats = compute_latency_stats()
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    assert "sla_status" in stats


def test_monitoring_health_summary():
    """Verify model health summary returns expected keys."""
    from src.monitoring import get_model_health_summary
    summary = get_model_health_summary()
    assert "health_score" in summary
    assert "total_predictions" in summary
    assert "latency" in summary

