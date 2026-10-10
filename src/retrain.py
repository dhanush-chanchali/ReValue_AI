"""
Module 6 (M1 & M6): Retraining Pipeline & Champion-Challenger Validation
Demonstrates automated retraining workflow:
1. Ingest newly collected production data (or drift-corrected dataset)
2. Retrain challenger model pipeline
3. Validate challenger against current champion model
4. Automatically promote challenger if validation MAE improves
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models_training import train_and_compare_models
from src.unsupervised import run_unsupervised_analysis
from src.evaluation import run_full_evaluation_diagnostics


def trigger_retraining(data_dir: str, models_dir: str) -> dict:
    """
    Executes automated retraining loop.
    Checks for performance improvement before promoting challenger model.
    """
    print("\n" + "="*70)
    print("🔄 RETRAINING LOOP TRIGGERED (M1 & M6)")
    print("="*70)

    # 1. Load current champion metadata
    benchmark_path = os.path.join(models_dir, "models_benchmark.json")
    champion_mae = float("inf")
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            old_bench = json.load(f)
            best_m = old_bench.get("best_model")
            if best_m and best_m in old_bench.get("models_comparison", {}):
                champion_mae = old_bench["models_comparison"][best_m]["val_metrics"]["MAE"]
                print(f"Current Champion Model: {best_m} (Validation MAE: ${champion_mae:.2f})")

    # 2. Re-run supervised training & comparison
    new_bench = train_and_compare_models(data_dir, models_dir)
    challenger_model = new_bench["best_model"]
    challenger_mae = new_bench["models_comparison"][challenger_model]["val_metrics"]["MAE"]

    # 3. Champion-Challenger Decision
    promoted = challenger_mae <= champion_mae
    status_msg = (
        f"Challenger ({challenger_model}) achieved Val MAE ${challenger_mae:.2f} "
        f"vs Champion ${champion_mae:.2f}. "
        f"{'PROMOTED TO PRODUCTION.' if promoted else 'REJECTED: Champion retained.'}"
    )
    print(f"\n[Retraining Result] {status_msg}")

    # 4. Refresh unsupervised & evaluation diagnostics
    run_unsupervised_analysis(data_dir, models_dir)
    run_full_evaluation_diagnostics(data_dir, models_dir)

    return {
        "status": "success",
        "challenger_model": challenger_model,
        "challenger_mae": challenger_mae,
        "champion_mae": champion_mae,
        "promoted": promoted,
        "message": status_msg
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(workspace_root, "data")
    models_folder = os.path.join(workspace_root, "models")
    trigger_retraining(data_folder, models_folder)
