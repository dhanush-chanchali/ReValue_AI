"""
ReValue AI - End-to-End Walkthrough Script
Executes all steps sequentially:
M1: Data Pipeline & Split
M2: Feature Preprocessing
M3: Supervised Model Benchmark (Ridge vs RF vs GBDT)
M4: Unsupervised K-Means & Isolation Forest
M5: Evaluation Diagnostics & Paired Significance Test
M6: Sample Prediction & Drift Audit
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import generate_synthetic_devices_dataset, split_and_save_data
from src.models_training import train_and_compare_models
from src.unsupervised import run_unsupervised_analysis
from src.evaluation import run_full_evaluation_diagnostics
from src.monitoring import detect_data_drift

def main():
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(workspace, "data")
    models_dir = os.path.join(workspace, "models")

    print("\n" + "#"*70)
    print("  ⚡ REVALUE AI — COMPLETE END-TO-END ML LIFECYCLE WALKTHROUGH")
    print("#"*70)

    # Stage 1: M1 Data Pipeline
    print("\n>>> STAGE 1 (M1): Ingestion & 3-Way Leak-Free Splitting...")
    df = generate_synthetic_devices_dataset(n_samples=3500)
    split_paths = split_and_save_data(df, data_dir)
    print(f"Data saved to: {split_paths['train']} (Train 70%), {split_paths['val']} (Val 15%), {split_paths['test']} (Test 15%)")

    # Stage 2: M2, M3, M5 Supervised Training
    print("\n>>> STAGE 2 (M2, M3, M5): Training Linear Baselines & Tree Ensembles...")
    bench = train_and_compare_models(data_dir, models_dir)
    print(f"Champion Model Selected: {bench['best_model']}")

    # Stage 3: M4 Unsupervised Learning
    print("\n>>> STAGE 3 (M4): K-Means Segmentation & Isolation Forest Anomaly Audit...")
    unsup = run_unsupervised_analysis(data_dir, models_dir)
    print(f"Optimal Clusters: K = {unsup['optimal_k']} | Anomalies Flagged: {unsup['anomaly_count_train']}")

    # Stage 4: M5 Statistical Significance
    print("\n>>> STAGE 4 (M5): Statistical Hypothesis Testing & Prediction Intervals...")
    eval_res = run_full_evaluation_diagnostics(data_dir, models_dir)
    for comp, stats_data in eval_res["statistical_significance"].items():
        print(f"  * {comp}: Improvement = ${stats_data['mae_improvement_usd']:.2f} (p-value: {stats_data['p_value']})")

    # Stage 5: M6 Production Drift Verification
    print("\n>>> STAGE 5 (M6): Production Drift Detector Check...")
    ref_df = pd.read_csv(os.path.join(data_dir, "processed", "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "processed", "test.csv"))
    drift_rep = detect_data_drift(ref_df, test_df)
    print(f"Data Drift Check on Test Distribution: {drift_rep['summary']}")

    print("\n" + "="*70)
    print("✅ REVALUE AI WALKTHROUGH COMPLETE — READY FOR PROJECT REVIEW!")
    print("="*70)

if __name__ == "__main__":
    main()
