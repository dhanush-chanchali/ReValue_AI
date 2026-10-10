"""
Module 6 (M1 & M6): Production-Grade ML Monitoring & Observability Engine
═══════════════════════════════════════════════════════════════════════════
Advanced monitoring implementation covering:

• Structured prediction logging (SQLite with rich schema)
• Population Stability Index (PSI) for categorical drift
• Kolmogorov-Smirnov (KS) 2-sample test for numerical drift
• Jensen-Shannon Divergence as supplementary drift metric
• Rolling latency percentiles (p50, p95, p99) and SLA breach tracking
• Anomaly rate trending and alert thresholds
• Model performance degradation tracking (MAE rolling window)
• Hourly & daily request volume aggregation
• Champion model health score calculation
"""

import os
import sys
import json
import time
import datetime
import sqlite3
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prediction_logs.db")
DRIFT_REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "latest_drift_report.json")

# ─── Monitoring Thresholds (SLAs) ─────────────────────────────────────────────
SLA_LATENCY_P95_MS     = 250.0   # p95 latency must stay under 250 ms
SLA_ANOMALY_RATE_PCT   = 8.0     # Alert if > 8% requests are flagged anomalies
SLA_DRIFT_KS_THRESHOLD = 0.05    # KS p-value threshold for feature drift
SLA_PSI_THRESHOLD      = 0.20    # PSI >= 0.20 = major distribution shift
SLA_JS_THRESHOLD       = 0.15    # Jensen-Shannon >= 0.15 = moderate drift

# ─── Features monitored for drift ─────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "original_price_usd",
    "age_months",
    "ram_gb",
    "storage_gb",
    "battery_health_pct",
]
CATEGORICAL_FEATURES = [
    "device_type",
    "brand",
    "condition",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def init_monitoring_db() -> None:
    """
    Initializes the SQLite monitoring database with two tables:
      • inference_logs   – one row per prediction request
      • drift_checkpoints – one row per drift evaluation run
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Core prediction log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id              TEXT UNIQUE,
            timestamp               TEXT NOT NULL,
            hour_bucket             TEXT,
            day_bucket              TEXT,
            device_type             TEXT,
            brand                   TEXT,
            model_name              TEXT,
            os_type                 TEXT,
            original_price_usd      REAL,
            age_months              INTEGER,
            ram_gb                  INTEGER,
            storage_gb              INTEGER,
            battery_health_pct      REAL,
            screen_size_inch        REAL,
            condition               TEXT,
            has_original_box        INTEGER DEFAULT 0,
            has_original_charger    INTEGER DEFAULT 0,
            has_screen_damage       INTEGER DEFAULT 0,
            has_body_scratches      INTEGER DEFAULT 0,
            predicted_price_usd     REAL,
            lower_bound_usd         REAL,
            upper_bound_usd         REAL,
            confidence_width_usd    REAL,
            depreciation_pct        REAL,
            market_tier             TEXT,
            is_anomaly              INTEGER DEFAULT 0,
            active_model            TEXT,
            latency_ms              REAL
        )
    """)

    # Drift checkpoint table – records every drift evaluation run
    c.execute("""
        CREATE TABLE IF NOT EXISTS drift_checkpoints (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at        TEXT NOT NULL,
            window_size         INTEGER,
            features_drifted    INTEGER,
            psi_flags           INTEGER,
            js_flags            INTEGER,
            retrain_recommended INTEGER DEFAULT 0,
            health_score        REAL,
            summary             TEXT
        )
    """)

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def log_prediction(
    input_data: Dict[str, Any],
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    is_anomaly: int,
    latency_ms: float,
    active_model: str = "Gradient_Boosting",
    market_tier: str = "Unknown",
    depreciation_pct: float = 0.0,
) -> str:
    """
    Logs a single prediction request to the monitoring database.
    Automatically derives:
      – Unique request_id (deterministic hash of input + timestamp)
      – hour_bucket & day_bucket for time-series aggregations
      – confidence_width_usd for interval coverage tracking
    
    Returns the generated request_id for traceability.
    """
    init_monitoring_db()

    now = datetime.datetime.utcnow()
    ts  = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    hr  = now.strftime("%Y-%m-%dT%H:00:00Z")
    day = now.strftime("%Y-%m-%d")

    # Deterministic request ID: hash of inputs + fractional timestamp
    raw = f"{json.dumps(input_data, sort_keys=True)}{time.time()}"
    request_id = "req_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    confidence_width = round(abs(upper_bound - lower_bound), 2)
    orig_price = float(input_data.get("original_price_usd", 1.0))
    dep_pct    = depreciation_pct or round(
        max(0.0, (orig_price - predicted_price) / max(orig_price, 1.0)) * 100.0, 2
    )

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT OR IGNORE INTO inference_logs (
                request_id, timestamp, hour_bucket, day_bucket,
                device_type, brand, model_name, os_type,
                original_price_usd, age_months, ram_gb, storage_gb,
                battery_health_pct, screen_size_inch, condition,
                has_original_box, has_original_charger,
                has_screen_damage, has_body_scratches,
                predicted_price_usd, lower_bound_usd, upper_bound_usd,
                confidence_width_usd, depreciation_pct,
                market_tier, is_anomaly, active_model, latency_ms
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?
            )
        """, (
            request_id, ts, hr, day,
            input_data.get("device_type", "Unknown"),
            input_data.get("brand",       "Unknown"),
            input_data.get("model_name",  "Unknown"),
            input_data.get("os",          "Unknown"),
            float(input_data.get("original_price_usd",  0.0)),
            int(  input_data.get("age_months",           0)),
            int(  input_data.get("ram_gb",               0)),
            int(  input_data.get("storage_gb",           0)),
            float(input_data.get("battery_health_pct",   85.0)),
            float(input_data.get("screen_size_inch",     6.1)),
            input_data.get("condition", "Good"),
            int(  input_data.get("has_original_box",     0)),
            int(  input_data.get("has_original_charger", 1)),
            int(  input_data.get("has_screen_damage",    0)),
            int(  input_data.get("has_body_scratches",   0)),
            round(float(predicted_price), 2),
            round(float(lower_bound),     2),
            round(float(upper_bound),     2),
            confidence_width,
            dep_pct,
            market_tier,
            int(is_anomaly),
            active_model,
            round(float(latency_ms), 3),
        ))
        conn.commit()
    finally:
        conn.close()

    return request_id


# ══════════════════════════════════════════════════════════════════════════════
# LOG RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def get_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves the most recent inference logs ordered by newest first."""
    init_monitoring_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_logs_df() -> pd.DataFrame:
    """Returns all inference logs as a Pandas DataFrame for analytics."""
    init_monitoring_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM inference_logs ORDER BY id ASC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_log_count() -> int:
    """Returns total number of logged predictions."""
    init_monitoring_db()
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# LATENCY & THROUGHPUT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_latency_stats(limit: int = 500) -> Dict[str, Any]:
    """
    Returns rolling latency statistics and SLA compliance:
      p50 / p95 / p99 inference latency, mean, std, and breach flags.
    """
    logs = get_recent_logs(limit)
    if not logs:
        return {"status": "No data available"}

    latencies = np.array([r["latency_ms"] for r in logs if r.get("latency_ms") is not None])
    if len(latencies) == 0:
        return {"status": "No latency data"}

    p50  = float(np.percentile(latencies, 50))
    p95  = float(np.percentile(latencies, 95))
    p99  = float(np.percentile(latencies, 99))
    mean = float(np.mean(latencies))
    std  = float(np.std(latencies))

    sla_breach = p95 > SLA_LATENCY_P95_MS

    return {
        "n_requests": len(latencies),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "min_ms": round(float(latencies.min()), 2),
        "max_ms": round(float(latencies.max()), 2),
        "sla_p95_threshold_ms": SLA_LATENCY_P95_MS,
        "sla_breach": sla_breach,
        "sla_status": "🔴 SLA BREACH — p95 latency exceeds threshold" if sla_breach else "🟢 Within SLA"
    }


def compute_request_volume_trend() -> Dict[str, Any]:
    """Aggregates request counts by day and by hour for time-series charts."""
    df = get_all_logs_df()
    if df.empty or "day_bucket" not in df.columns:
        return {"daily": [], "hourly": []}

    daily  = df.groupby("day_bucket").size().reset_index(name="requests")
    hourly = df.groupby("hour_bucket").size().reset_index(name="requests")

    return {
        "daily":  daily.to_dict(orient="records"),
        "hourly": hourly.tail(24).to_dict(orient="records"),
        "total":  int(len(df)),
    }


def compute_prediction_distribution() -> Dict[str, Any]:
    """Returns statistical summary of predicted price distribution."""
    df = get_all_logs_df()
    if df.empty or "predicted_price_usd" not in df.columns:
        return {}

    preds = df["predicted_price_usd"].dropna().values
    return {
        "mean":    round(float(np.mean(preds)), 2),
        "median":  round(float(np.median(preds)), 2),
        "std":     round(float(np.std(preds)), 2),
        "p25":     round(float(np.percentile(preds, 25)), 2),
        "p75":     round(float(np.percentile(preds, 75)), 2),
        "min":     round(float(preds.min()), 2),
        "max":     round(float(preds.max()), 2),
        "anomaly_rate_pct": round(
            float(df["is_anomaly"].sum() / max(len(df), 1) * 100.0), 2
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION (NUMERICAL + CATEGORICAL + JSD)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_psi(ref_vals: np.ndarray, curr_vals: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for numerical distributions.
    PSI < 0.10  → No significant change (stable)
    PSI 0.10–0.20 → Moderate change (monitor)
    PSI ≥ 0.20  → Major shift (retraining recommended)
    """
    min_val = min(ref_vals.min(),  curr_vals.min())
    max_val = max(ref_vals.max(), curr_vals.max())
    edges   = np.linspace(min_val, max_val, bins + 1)

    ref_counts,  _ = np.histogram(ref_vals,  bins=edges)
    curr_counts, _ = np.histogram(curr_vals, bins=edges)

    ref_pct  = ref_counts  / max(ref_counts.sum(),  1) + 1e-8
    curr_pct = curr_counts / max(curr_counts.sum(), 1) + 1e-8

    psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    return round(psi, 4)


def _compute_jsd(ref_vals: np.ndarray, curr_vals: np.ndarray, bins: int = 20) -> float:
    """Jensen-Shannon Divergence (0–1 scale, lower = more similar distributions)."""
    min_val = min(ref_vals.min(),  curr_vals.min())
    max_val = max(ref_vals.max(), curr_vals.max())
    edges   = np.linspace(min_val, max_val, bins + 1)

    ref_counts,  _ = np.histogram(ref_vals,  bins=edges, density=True)
    curr_counts, _ = np.histogram(curr_vals, bins=edges, density=True)

    ref_counts  = ref_counts  + 1e-10
    curr_counts = curr_counts + 1e-10

    ref_norm  = ref_counts  / ref_counts.sum()
    curr_norm = curr_counts / curr_counts.sum()

    jsd = float(jensenshannon(ref_norm, curr_norm))
    return round(jsd, 4)


def _categorical_psi(ref_series: pd.Series, curr_series: pd.Series) -> float:
    """PSI for categorical features using category frequency comparison."""
    all_cats = set(ref_series.unique()) | set(curr_series.unique())
    ref_pct  = ref_series.value_counts(normalize=True).reindex(all_cats, fill_value=1e-8)
    curr_pct = curr_series.value_counts(normalize=True).reindex(all_cats, fill_value=1e-8)

    psi = float(np.sum((curr_pct.values - ref_pct.values) * np.log(curr_pct.values / ref_pct.values)))
    return round(psi, 4)


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    min_samples: int = 10,
) -> Dict[str, Any]:
    """
    Advanced multi-metric drift detection engine:
      • Kolmogorov-Smirnov (KS) test on numerical features
      • Population Stability Index (PSI) on numerical & categorical features
      • Jensen-Shannon Divergence on numerical features
      • Composite health score (0–100) for model readiness
      • Automatic retrain recommendation based on combined signal

    Returns a comprehensive drift report persisted to disk.
    """
    now_ts    = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    drift_report: Dict[str, Any] = {}
    ks_flags   = 0
    psi_flags  = 0
    jsd_flags  = 0

    # ── Numerical Features ──────────────────────────────────────────────────
    for col in NUMERICAL_FEATURES:
        if col not in reference_df.columns or col not in current_df.columns:
            drift_report[col] = {"status": "Column not found in both datasets"}
            continue

        ref_vals  = reference_df[col].dropna().values
        curr_vals = current_df[col].dropna().values

        if len(curr_vals) < min_samples:
            drift_report[col] = {
                "status": f"Insufficient samples ({len(curr_vals)} < {min_samples})",
                "drift_detected": False,
                "psi": None,
                "jsd": None,
            }
            continue

        # KS test
        ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
        ks_drifted = bool(p_val < SLA_DRIFT_KS_THRESHOLD)

        # PSI
        psi_val = _compute_psi(ref_vals, curr_vals)
        psi_flag = psi_val >= SLA_PSI_THRESHOLD

        # JSD
        jsd_val = _compute_jsd(ref_vals, curr_vals)
        jsd_flag = jsd_val >= SLA_JS_THRESHOLD

        # Severity
        any_drift = ks_drifted or psi_flag or jsd_flag
        if ks_drifted:
            ks_flags  += 1
        if psi_flag:
            psi_flags += 1
        if jsd_flag:
            jsd_flags += 1

        severity = "STABLE"
        if ks_drifted and psi_flag:
            severity = "CRITICAL"
        elif any_drift:
            severity = "WARNING"

        drift_report[col] = {
            "type": "numerical",
            "drift_detected": any_drift,
            "severity": severity,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(f"{p_val:.3e}"),
            "psi": psi_val,
            "psi_flag": psi_flag,
            "jsd": jsd_val,
            "jsd_flag": jsd_flag,
            "ref_mean": round(float(ref_vals.mean()), 3),
            "curr_mean": round(float(curr_vals.mean()), 3),
            "ref_std": round(float(ref_vals.std()), 3),
            "curr_std": round(float(curr_vals.std()), 3),
            "ref_n": len(ref_vals),
            "curr_n": len(curr_vals),
        }

    # ── Categorical Features ─────────────────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_series  = reference_df[col].dropna()
        curr_series = current_df[col].dropna()

        if len(curr_series) < min_samples:
            drift_report[col] = {
                "status": f"Insufficient samples ({len(curr_series)} < {min_samples})",
                "drift_detected": False,
            }
            continue

        psi_val  = _categorical_psi(ref_series, curr_series)
        psi_flag = psi_val >= SLA_PSI_THRESHOLD
        if psi_flag:
            psi_flags += 1

        ref_dist  = ref_series.value_counts(normalize=True).round(3).to_dict()
        curr_dist = curr_series.value_counts(normalize=True).round(3).to_dict()

        severity = "CRITICAL" if psi_val >= 0.25 else ("WARNING" if psi_flag else "STABLE")

        drift_report[col] = {
            "type": "categorical",
            "drift_detected": psi_flag,
            "severity": severity,
            "psi": psi_val,
            "psi_flag": psi_flag,
            "ref_distribution": ref_dist,
            "curr_distribution": curr_dist,
        }

    # ── Composite Health Score ────────────────────────────────────────────────
    # Score starts at 100 and deductions apply for each drift signal
    total_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    deductions = (ks_flags * 10) + (psi_flags * 12) + (jsd_flags * 8)
    health_score = round(max(0.0, min(100.0, 100.0 - deductions)), 1)

    retrain_recommended = bool(ks_flags >= 2 or psi_flags >= 2)
    retrain_urgency = (
        "🔴 CRITICAL – Retrain immediately"  if health_score < 50 else
        "🟡 WARNING  – Schedule retraining"  if health_score < 75 else
        "🟢 HEALTHY  – No action required"
    )

    # ── Anomaly Rate SLA ─────────────────────────────────────────────────────
    pred_dist = compute_prediction_distribution()
    anomaly_rate = pred_dist.get("anomaly_rate_pct", 0.0)
    anomaly_sla_breach = anomaly_rate > SLA_ANOMALY_RATE_PCT

    # ── Latency SLA ───────────────────────────────────────────────────────────
    latency_stats = compute_latency_stats()

    final_report = {
        "evaluated_at": now_ts,
        "window_samples": len(current_df),
        "health_score": health_score,
        "retrain_recommended": retrain_recommended,
        "retrain_urgency": retrain_urgency,
        "summary": (
            "⚠️  Covariate shift detected — multiple features have drifted from training distribution." 
            if retrain_recommended else
            "✅ Input distributions are statistically consistent with training data."
        ),
        "ks_features_drifted": ks_flags,
        "psi_features_flagged": psi_flags,
        "jsd_features_flagged": jsd_flags,
        "anomaly_rate_pct": anomaly_rate,
        "anomaly_sla_breach": anomaly_sla_breach,
        "latency_sla": latency_stats,
        "feature_details": drift_report,
    }

    # ── Persist Drift Report ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(final_report, f, indent=2)

    # ── Checkpoint in DB ──────────────────────────────────────────────────────
    init_monitoring_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO drift_checkpoints
            (evaluated_at, window_size, features_drifted, psi_flags, js_flags,
             retrain_recommended, health_score, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_ts,
        len(current_df),
        ks_flags,
        psi_flags,
        jsd_flags,
        int(retrain_recommended),
        health_score,
        final_report["summary"],
    ))
    conn.commit()
    conn.close()

    return final_report


def get_drift_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves history of drift evaluation checkpoints for trend analysis."""
    init_monitoring_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM drift_checkpoints ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_model_health_summary() -> Dict[str, Any]:
    """
    Returns a comprehensive model health snapshot including:
    - Latency SLA status
    - Anomaly rate vs threshold
    - Request volume
    - Latest drift evaluation summary
    """
    logs     = get_recent_logs(500)
    lat      = compute_latency_stats(500)
    pred_d   = compute_prediction_distribution()
    vol      = compute_request_volume_trend()

    # Load latest drift report if available
    drift_snap: Dict = {}
    if os.path.exists(DRIFT_REPORT_PATH):
        try:
            with open(DRIFT_REPORT_PATH, "r") as f:
                drift_snap = json.load(f)
        except Exception:
            pass

    anomaly_rate = pred_d.get("anomaly_rate_pct", 0.0)
    health_score = drift_snap.get("health_score", 100.0) if drift_snap else 100.0

    # Penalise health score for SLA breaches
    if lat.get("sla_breach"):
        health_score = max(0.0, health_score - 15.0)
    if anomaly_rate > SLA_ANOMALY_RATE_PCT:
        health_score = max(0.0, health_score - 10.0)

    return {
        "total_predictions": vol.get("total", len(logs)),
        "health_score": round(health_score, 1),
        "latency": lat,
        "prediction_distribution": pred_d,
        "request_volume": vol,
        "latest_drift": drift_snap,
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
