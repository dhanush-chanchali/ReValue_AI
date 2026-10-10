"""
Module 4 (M4): Unsupervised Learning Pipeline
Implements:
- K-Means Clustering (Lloyd's algorithm, K-Means++ init)
- Hyperparameter selection for K: Elbow Method (Inertia) & Silhouette Score
- Market Segmentation labels for devices
- Dimensionality Reduction: PCA (2D & 3D projections + explained variance)
- Anomaly Detection: Isolation Forest for detecting suspicious / fraudulent listings
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42


def run_unsupervised_analysis(data_dir: str, models_dir: str) -> Dict[str, Any]:
    """
    Executes K-Means segmentation, PCA dimensionality reduction,
    and Isolation Forest anomaly detection on processed device data.
    """
    os.makedirs(models_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "processed", "train.csv"))
    
    # Select key numerical and ordinal specs for clustering & anomaly detection
    feature_cols = [
        "original_price_usd",
        "age_months",
        "ram_gb",
        "storage_gb",
        "battery_health_pct",
        "screen_size_inch",
        "has_screen_damage",
        "has_body_scratches",
        "resale_price_usd"
    ]

    X = train_df[feature_cols].copy()

    # Preprocess: Impute missing + Standardize
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # -------------------------------------------------------------
    # 1. K-MEANS: Elbow Method & Silhouette Analysis (M4)
    # -------------------------------------------------------------
    k_range = list(range(2, 8))
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, km.labels_))
        silhouette_scores_list.append(round(score, 4))

    # Select optimal K based on silhouette score
    best_k_idx = int(np.argmax(silhouette_scores_list))
    best_k = k_range[best_k_idx]
    
    # Fit final K-Means with optimal K (typically 4 market tiers)
    optimal_kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=15, random_state=RANDOM_STATE)
    cluster_labels = optimal_kmeans.fit_predict(X_scaled)

    # Label clusters by average resale price tiers
    train_df["cluster"] = cluster_labels
    cluster_price_rank = train_df.groupby("cluster")["resale_price_usd"].mean().sort_values()
    
    tier_names = [
        "Budget Entry Device",
        "Mid-Range Daily Driver",
        "Depreciated Flagship",
        "Premium Flagship / Workstation"
    ]
    # Map cluster index to business tier
    cluster_to_tier = {}
    for rank_idx, cluster_id in enumerate(cluster_price_rank.index):
        tier_title = tier_names[min(rank_idx, len(tier_names) - 1)]
        cluster_to_tier[int(cluster_id)] = tier_title

    # -------------------------------------------------------------
    # 2. PCA: Dimensionality Reduction for Visualization (M4)
    # -------------------------------------------------------------
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # PCA feature loadings (which features drive PC1 and PC2)
    loadings_pc1 = {col: round(float(pca.components_[0, idx]), 3) for idx, col in enumerate(feature_cols)}
    loadings_pc2 = {col: round(float(pca.components_[1, idx]), 3) for idx, col in enumerate(feature_cols)}

    # -------------------------------------------------------------
    # 3. ISOLATION FOREST: Anomaly & Fraud Detection (M4)
    # -------------------------------------------------------------
    # Contamination set to 3% to catch suspicious / anomalous listings
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.03,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    anomaly_preds = iso_forest.fit_predict(X_scaled) # -1 is anomaly, 1 is normal
    anomaly_scores = iso_forest.score_samples(X_scaled)

    num_anomalies = int((anomaly_preds == -1).sum())

    # -------------------------------------------------------------
    # 4. Save Artifacts & Pipelines (M6)
    # -------------------------------------------------------------
    unsupervised_pipeline = {
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": optimal_kmeans,
        "pca": pca,
        "isolation_forest": iso_forest,
        "feature_cols": feature_cols,
        "cluster_to_tier": cluster_to_tier
    }
    joblib.dump(unsupervised_pipeline, os.path.join(models_dir, "unsupervised_pipeline.joblib"))

    # Sample projection points for fast Streamlit rendering
    sample_df = train_df.iloc[:400].copy()
    sample_scaled = X_scaled[:400]
    sample_pca = X_pca[:400]
    sample_anomalies = anomaly_preds[:400]

    visualization_data = []
    for i in range(len(sample_df)):
        visualization_data.append({
            "device_id": str(sample_df.iloc[i]["device_id"]),
            "device_type": str(sample_df.iloc[i]["device_type"]),
            "brand": str(sample_df.iloc[i]["brand"]),
            "model_name": str(sample_df.iloc[i]["model_name"]),
            "resale_price_usd": float(sample_df.iloc[i]["resale_price_usd"]),
            "original_price_usd": float(sample_df.iloc[i]["original_price_usd"]),
            "age_months": int(sample_df.iloc[i]["age_months"]),
            "battery_health_pct": float(sample_df.iloc[i]["battery_health_pct"]) if pd.notnull(sample_df.iloc[i]["battery_health_pct"]) else 85.0,
            "cluster": int(sample_df.iloc[i]["cluster"]),
            "tier_name": cluster_to_tier[int(sample_df.iloc[i]["cluster"])],
            "pca_x": round(float(sample_pca[i, 0]), 3),
            "pca_y": round(float(sample_pca[i, 1]), 3),
            "pca_z": round(float(sample_pca[i, 2]), 3),
            "is_anomaly": int(sample_anomalies[i] == -1)
        })

    summary_unsupervised = {
        "k_range": k_range,
        "inertias": [round(val, 2) for val in inertias],
        "silhouette_scores": silhouette_scores_list,
        "optimal_k": best_k,
        "cluster_to_tier": cluster_to_tier,
        "pca_explained_variance_ratio": explained_var,
        "loadings_pc1": loadings_pc1,
        "loadings_pc2": loadings_pc2,
        "anomaly_count_train": num_anomalies,
        "contamination_rate": 0.03,
        "sample_points": visualization_data
    }

    with open(os.path.join(models_dir, "unsupervised_summary.json"), "w") as f:
        json.dump(summary_unsupervised, f, indent=2)

    print("\n" + "="*70)
    print("✨ UNSUPERVISED LEARNING (M4) COMPLETED")
    print("="*70)
    print(f"  - Optimal K: {best_k} clusters (Max Silhouette Score: {max(silhouette_scores_list):.4f})")
    print(f"  - PCA Total Variance Explained (3 PCs): {sum(explained_var) * 100:.1f}%")
    print(f"  - Isolation Forest Detected: {num_anomalies} anomalies ({num_anomalies/len(train_df)*100:.1f}%)")
    print(f"  - Pipeline saved to: {os.path.join(models_dir, 'unsupervised_pipeline.joblib')}")

    return summary_unsupervised


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(workspace_root, "data")
    models_folder = os.path.join(workspace_root, "models")
    run_unsupervised_analysis(data_folder, models_folder)
