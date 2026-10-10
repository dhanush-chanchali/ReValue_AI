"""
Module 3 (M2, M3, M5): Supervised Model Training, Comparison & Serialization
Implements:
- M2: Linear Regression, Ridge (L2), Lasso (L1), ElasticNet with coefficient analysis
- M3: Random Forest (Bagging, OOB score) & Gradient Boosting (Boosting, Hist-based)
- M5: 5-Fold Cross Validation, MAE, RMSE, MAPE, R2, Model Selection & Serialization
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from src.features import build_preprocessor, get_feature_names

RANDOM_STATE = 42


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAPE in percentage (%)"""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates all key regression metrics (M5)"""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 2),
        "R2_Score": round(r2, 4)
    }


def train_and_compare_models(data_dir: str, models_dir: str) -> Dict[str, Any]:
    """
    Trains Ridge, Lasso, Random Forest, and Gradient Boosting on train split.
    Evaluates via 5-Fold Cross-Validation, validates on Validation set,
    and performs final benchmark on held-out Test set.
    """
    os.makedirs(models_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "processed", "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "processed", "test.csv"))

    target_col = "resale_price_usd"
    drop_cols = ["device_id", target_col]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[target_col].values

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col].values

    # Define Candidate Models across M2 and M3
    candidate_models = {
        "Linear_OLS": LinearRegression(),
        "Ridge_L2": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso_L1": Lasso(alpha=1.5, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
        "Random_Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_depth=8,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )
    }

    comparison_results = {}
    fitted_pipelines = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "="*70)
    print("🚀 TRAINING & COMPARING SUPERVISED MODELS (M2, M3, M5)")
    print("="*70)

    for name, estimator in candidate_models.items():
        print(f"\nTraining pipeline for: {name} ...")
        # Build shared preprocessor pipeline to prevent training-serving skew
        preprocessor = build_preprocessor()
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        # 1. 5-Fold Cross Validation on Train Split (M5)
        cv_scores = cross_validate(
            full_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
            return_train_score=False
        )

        cv_mae = -float(np.mean(cv_scores["test_neg_mean_absolute_error"]))
        cv_rmse = -float(np.mean(cv_scores["test_neg_root_mean_squared_error"]))
        cv_r2 = float(np.mean(cv_scores["test_r2"]))

        # 2. Fit full training data
        full_pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = full_pipeline

        # 3. Validation set evaluation
        y_val_pred = full_pipeline.predict(X_val)
        val_metrics = evaluate_predictions(y_val, y_val_pred)

        # 4. Held-out Test set evaluation (Final unbiased benchmark)
        y_test_pred = full_pipeline.predict(X_test)
        test_metrics = evaluate_predictions(y_test, y_test_pred)

        print(f"  --> CV MAE: ${cv_mae:.2f} | Val MAE: ${val_metrics['MAE']:.2f} | Val R²: {val_metrics['R2_Score']:.4f} | Test MAE: ${test_metrics['MAE']:.2f} | Test R²: {test_metrics['R2_Score']:.4f}")

        comparison_results[name] = {
            "cv_mae": round(cv_mae, 2),
            "cv_rmse": round(cv_rmse, 2),
            "cv_r2": round(cv_r2, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

        # Save individual model pipeline artifact (M6)
        artifact_path = os.path.join(models_dir, f"{name.lower()}_pipeline.joblib")
        joblib.dump(full_pipeline, artifact_path)

    # 5. Model Selection: Select best model based on Validation MAE (M5)
    best_model_name = min(comparison_results, key=lambda k: comparison_results[k]["val_metrics"]["MAE"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_artifact_path = os.path.join(models_dir, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, best_artifact_path)

    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name} (Val MAE: ${comparison_results[best_model_name]['val_metrics']['MAE']:.2f})")

    # 6. Extract Feature Importances & Linear Coefficients (M2 & M3 Interpretability)
    fitted_prep = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_prep)

    # Extract Ridge Coefficients
    ridge_pipeline = fitted_pipelines["Ridge_L2"]
    ridge_coefs = ridge_pipeline.named_steps["model"].coef_
    ridge_importance = [
        {"feature": feat, "coefficient": round(float(coef), 3), "abs_impact": round(float(abs(coef)), 3)}
        for feat, coef in zip(feature_names, ridge_coefs)
    ]
    ridge_importance.sort(key=lambda x: x["abs_impact"], reverse=True)

    # Extract Random Forest Feature Importances (MDI Gini/Variance Impurity)
    rf_pipeline = fitted_pipelines["Random_Forest"]
    rf_importances = rf_pipeline.named_steps["model"].feature_importances_
    rf_importance_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in zip(feature_names, rf_importances)
    ]
    rf_importance_list.sort(key=lambda x: x["importance"], reverse=True)

    # Permutation importance on test set for Best Model
    perm_res = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_importance_list = [
        {"feature": col, "mean_importance": round(float(perm_res.importances_mean[idx]), 3)}
        for idx, col in enumerate(X_test.columns)
    ]
    perm_importance_list.sort(key=lambda x: x["mean_importance"], reverse=True)

    # Save summary metadata and explainability artifact
    summary_metadata = {
        "best_model": best_model_name,
        "models_comparison": comparison_results,
        "feature_names": feature_names,
        "ridge_coefficients": ridge_importance,
        "random_forest_mdi_importance": rf_importance_list,
        "best_model_permutation_importance": perm_importance_list,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    }

    with open(os.path.join(models_dir, "models_benchmark.json"), "w") as f:
        json.dump(summary_metadata, f, indent=2)

    print("\n✅ All models trained, compared, and packaged into 'models/' successfully!")
    return summary_metadata


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(workspace_root, "data")
    models_folder = os.path.join(workspace_root, "models")
    train_and_compare_models(data_folder, models_folder)
