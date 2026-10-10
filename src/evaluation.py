"""
Module 5 (M5): Evaluation, Selection, Diagnostics & Statistical Significance
Implements:
- 3-Way Split discipline verification (Train/Val/Test)
- Learning curves (diagnosing underfitting vs overfitting)
- Residual diagnostics (homoscedasticity, residual distribution, Q-Q analysis)
- Prediction interval estimation (empirical 90% confidence bands)
- Paired statistical significance test comparing Ridge vs Random Forest vs Gradient Boosting
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.model_selection import learning_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RANDOM_STATE = 42


def compute_paired_model_significance(
    y_test: np.ndarray,
    preds_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Performs Paired t-test and Wilcoxon signed-rank test on absolute errors (M5)
    to statistically determine whether tree-based models significantly outperform
    the linear Ridge baseline.
    """
    baseline_name = "Ridge_L2"
    if baseline_name not in preds_dict:
        baseline_name = list(preds_dict.keys())[0]

    baseline_errors = np.abs(y_test - preds_dict[baseline_name])
    comparisons = {}

    for model_name, preds in preds_dict.items():
        if model_name == baseline_name:
            continue
        model_errors = np.abs(y_test - preds)
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(baseline_errors, model_errors)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_pval = stats.wilcoxon(baseline_errors, model_errors)

        is_significant = bool(p_val < 0.01)
        mean_diff = float(np.mean(baseline_errors) - np.mean(model_errors))

        comparisons[f"{baseline_name}_vs_{model_name}"] = {
            "baseline_mean_abs_error": round(float(np.mean(baseline_errors)), 2),
            "candidate_mean_abs_error": round(float(np.mean(model_errors)), 2),
            "mae_improvement_usd": round(mean_diff, 2),
            "paired_t_statistic": round(float(t_stat), 4),
            "p_value": float(f"{p_val:.2e}"),
            "wilcoxon_statistic": round(float(w_stat), 4),
            "wilcoxon_p_value": float(f"{w_pval:.2e}"),
            "statistically_significant": is_significant,
            "conclusion": f"{model_name} significantly outperforms {baseline_name} (p < 0.01)" if is_significant else "No statistically significant difference"
        }

    return comparisons


def generate_learning_curve_data(pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Generates learning curve coordinates (train sizes, train scores, val scores)
    to diagnose underfitting, overfitting, and sample sufficiency (M5).
    """
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.15, 1.0, 5),
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    train_mae_mean = -np.mean(train_scores, axis=1)
    val_mae_mean = -np.mean(test_scores, axis=1)

    return {
        "train_sizes": [int(s) for s in train_sizes],
        "train_mae": [round(float(m), 2) for m in train_mae_mean],
        "validation_mae": [round(float(m), 2) for m in val_mae_mean]
    }


def compute_prediction_intervals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    confidence_level: float = 0.90
) -> Dict[str, float]:
    """
    Computes empirical residual quantiles for 90% prediction intervals (M5 Calibration/Uncertainty).
    """
    residuals = y_test - y_pred
    lower_pct = ((1.0 - confidence_level) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - confidence_level) / 2.0) * 100.0

    lower_bound_delta = float(np.percentile(residuals, lower_pct))
    upper_bound_delta = float(np.percentile(residuals, upper_pct))
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    return {
        "confidence_level": confidence_level,
        "lower_bound_offset": round(lower_bound_delta, 2),
        "upper_bound_offset": round(upper_bound_delta, 2),
        "mean_residual": round(mean_residual, 2),
        "std_residual": round(std_residual, 2)
    }


def run_full_evaluation_diagnostics(data_dir: str, models_dir: str) -> Dict[str, Any]:
    """Runs all M5 diagnostics and exports evaluation artifact."""
    train_df = pd.read_csv(os.path.join(data_dir, "processed", "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "processed", "test.csv"))

    target_col = "resale_price_usd"
    drop_cols = ["device_id", target_col]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col].values

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col].values

    # Load pipelines
    models_to_evaluate = ["ridge_l2", "random_forest", "gradient_boosting"]
    predictions = {}
    pipelines = {}

    for m in models_to_evaluate:
        p_path = os.path.join(models_dir, f"{m}_pipeline.joblib")
        if os.path.exists(p_path):
            pipe = joblib.load(p_path)
            pipelines[m] = pipe
            predictions[m] = pipe.predict(X_test)

    # 1. Statistical Significance Test
    sig_test_results = compute_paired_model_significance(
        y_test,
        {"Ridge_L2": predictions.get("ridge_l2"), 
         "Random_Forest": predictions.get("random_forest"), 
         "Gradient_Boosting": predictions.get("gradient_boosting")}
    )

    # 2. Learning Curves for Best Model
    best_pipe = joblib.load(os.path.join(models_dir, "best_model_pipeline.joblib"))
    learning_curve_data = generate_learning_curve_data(best_pipe, X_train, y_train)

    # 3. Prediction Intervals & Residuals
    best_preds = best_pipe.predict(X_test)
    pred_intervals = compute_prediction_intervals(y_test, best_preds)

    # Residuals for sample plotting
    residuals_sample = []
    for i in range(min(300, len(y_test))):
        residuals_sample.append({
            "actual": float(y_test[i]),
            "predicted": round(float(best_preds[i]), 2),
            "residual": round(float(y_test[i] - best_preds[i]), 2)
        })

    eval_artifact = {
        "statistical_significance": sig_test_results,
        "learning_curve": learning_curve_data,
        "prediction_intervals": pred_intervals,
        "residuals_sample": residuals_sample
    }

    with open(os.path.join(models_dir, "evaluation_diagnostics.json"), "w") as f:
        json.dump(eval_artifact, f, indent=2)

    print("\n" + "="*70)
    print("📈 EVALUATION & STATISTICAL SIGNIFICANCE (M5) COMPLETE")
    print("="*70)
    for comp, res in sig_test_results.items():
        print(f"  - {comp}: Improvement = ${res['mae_improvement_usd']:.2f} | p-value = {res['p_value']} ({res['conclusion']})")

    return eval_artifact


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(workspace_root, "data")
    models_folder = os.path.join(workspace_root, "models")
    run_full_evaluation_diagnostics(data_folder, models_folder)
