"""
Module 2 (M2): Feature Engineering & Preprocessing Pipeline
Implements:
- Numerical scaling (StandardScaler vs MinMaxScaler)
- Categorical encoding (OneHotEncoder for nominal, OrdinalEncoder for ranked condition)
- Missing value imputation (SimpleImputer)
- Feature interactions (battery cliff, spec score, accessory completeness)
- Scikit-learn ColumnTransformer & Pipeline to prevent training-serving skew
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple


# Ordered condition hierarchy for ordinal encoding
CONDITION_ORDER = [["Heavily Used", "Fair", "Good", "Pristine"]]

# Feature column definitions
NUMERICAL_COLS = [
    "original_price_usd",
    "age_months",
    "ram_gb",
    "storage_gb",
    "screen_size_inch",
    "battery_health_pct"
]

BINARY_COLS = [
    "has_original_box",
    "has_original_charger",
    "has_screen_damage",
    "has_body_scratches"
]

NOMINAL_CATEGORICAL_COLS = [
    "device_type",
    "brand",
    "os"
]

ORDINAL_CATEGORICAL_COLS = [
    "condition"
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to engineer tabular electronics features.
    Ensures identical feature calculation during both training and production serving.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Spec Capacity Score: Combined memory and storage footprint
        if "ram_gb" in X_out.columns and "storage_gb" in X_out.columns:
            X_out["spec_power_score"] = (X_out["ram_gb"] * 2.5) + (X_out["storage_gb"] / 32.0)
        
        # 2. Battery Cliff Indicator (Sub-80% threshold where Apple & Android trade-in prices drop)
        if "battery_health_pct" in X_out.columns:
            # Handle potential NaNs safely before imputation
            battery = X_out["battery_health_pct"].fillna(85.0)
            X_out["is_battery_degraded"] = (battery < 80.0).astype(float)
        
        # 3. Accessory Completeness (Box + Charger completeness)
        if "has_original_box" in X_out.columns and "has_original_charger" in X_out.columns:
            X_out["accessory_completeness"] = (
                X_out["has_original_box"].astype(float) * 0.4 + 
                X_out["has_original_charger"].astype(float) * 0.6
            )
            
        # 4. Total Physical Wear Penalty
        if "has_screen_damage" in X_out.columns and "has_body_scratches" in X_out.columns:
            X_out["wear_severity"] = (
                X_out["has_screen_damage"].astype(float) * 2.5 + 
                X_out["has_body_scratches"].astype(float) * 1.0
            )

        return X_out


def build_preprocessor() -> Pipeline:
    """
    Constructs an end-to-end ColumnTransformer preprocessing pipeline.
    Implements:
    - Median Imputation + Standard Scaling for Continuous Numerics
    - One-Hot Encoding for Nominal categories (device_type, brand, os)
    - Ordinal Encoding for Condition rating (Heavily Used -> Pristine)
    - Passthrough for Binary flags
    """
    # 1. Numerical pipeline: Median Imputation + Standard Scaling
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Ordinal pipeline: Most Frequent Imputation + Ordered Integer Mapping
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=CONDITION_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler())
    ])

    # 3. Nominal categorical pipeline: One-hot encoding
    nom_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    # 4. Binary pipeline: Simple median imputer (0 or 1)
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # ColumnTransformer combining all pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("ord", ord_pipeline, ORDINAL_CATEGORICAL_COLS),
            ("nom", nom_pipeline, NOMINAL_CATEGORICAL_COLS),
            ("bin", bin_pipeline, BINARY_COLS)
        ],
        remainder="drop"
    )

    # Full feature engineering + transformation pipeline
    full_pipeline = Pipeline([
        ("engineer", DomainFeatureEngineer()),
        ("preprocess", preprocessor)
    ])

    return full_pipeline


def get_feature_names(fitted_preprocessor: Pipeline) -> List[str]:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer.
    Essential for linear coefficient and feature importance interpretation.
    """
    ct = fitted_preprocessor.named_steps["preprocess"]
    feature_names = []

    # Numerical feature names
    feature_names.extend(NUMERICAL_COLS)
    
    # Ordinal feature names
    feature_names.extend(ORDINAL_CATEGORICAL_COLS)

    # Nominal onehot feature names
    onehot_step = ct.named_transformers_["nom"].named_steps["onehot"]
    cat_names = onehot_step.get_feature_names_out(NOMINAL_CATEGORICAL_COLS).tolist()
    feature_names.extend(cat_names)

    # Binary feature names
    feature_names.extend(BINARY_COLS)

    return feature_names
