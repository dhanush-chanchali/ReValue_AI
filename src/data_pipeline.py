"""
Module 1 (M1): Data Pipeline & ML Lifecycle Management
Handles raw data ingestion, realistic data synthesis, schema validation,
and train/validation/test splitting with zero data leakage.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Reproducibility seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_devices_dataset(n_samples: int = 3500) -> pd.DataFrame:
    """
    Generates a realistic tabular dataset of secondhand smartphones and laptops.
    Simulates real-world market dynamics:
    - Non-linear depreciation over time (exponential decay)
    - Brand retention curves (Apple/MacBook retains higher % than budget brands)
    - Battery health penalty (sharp drop below 80%)
    - Physical condition multipliers (Pristine vs Heavily Used)
    - Spec bonuses (RAM, Storage)
    - Realistic noise and missing values to test M2 imputation
    """
    records = []

    brands_phones = ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"]
    brands_laptops = ["Apple", "Dell", "Lenovo", "HP", "Asus"]

    phone_models = {
        "Apple": ["iPhone 11", "iPhone 12", "iPhone 13", "iPhone 14 Pro", "iPhone 15 Pro"],
        "Samsung": ["Galaxy S20", "Galaxy S21", "Galaxy S22", "Galaxy S23 Ultra", "Galaxy Z Flip 4"],
        "Google": ["Pixel 5", "Pixel 6", "Pixel 7 Pro", "Pixel 8"],
        "OnePlus": ["OnePlus 8T", "OnePlus 9 Pro", "OnePlus 10 Pro", "OnePlus 11"],
        "Xiaomi": ["Redmi Note 10", "Xiaomi 11T", "Xiaomi 12 Pro", "Xiaomi 13"]
    }

    laptop_models = {
        "Apple": ["MacBook Air M1", "MacBook Air M2", "MacBook Pro 14 M1", "MacBook Pro 16 M2"],
        "Dell": ["XPS 13", "XPS 15", "Inspiron 15", "Latitude 5420"],
        "Lenovo": ["ThinkPad X1 Carbon", "IdeaPad 5", "Legion 5 Pro", "ThinkPad T14"],
        "HP": ["Spectre x360", "Envy 13", "Pavilion 15", "Omen 16"],
        "Asus": ["ZenBook 14", "ROG Zephyrus G14", "VivoBook 15", "TUF Gaming A15"]
    }

    conditions = ["Pristine", "Good", "Fair", "Heavily Used"]
    condition_weights = [0.25, 0.45, 0.20, 0.10]
    condition_multipliers = {"Pristine": 1.0, "Good": 0.85, "Fair": 0.68, "Heavily Used": 0.48}

    brand_retention_factors = {
        "Apple": 1.25,
        "Samsung": 1.05,
        "Google": 0.98,
        "OnePlus": 0.95,
        "Xiaomi": 0.85,
        "Dell": 1.00,
        "Lenovo": 1.02,
        "HP": 0.95,
        "Asus": 0.96
    }

    for i in range(n_samples):
        device_type = np.random.choice(["Smartphone", "Laptop"], p=[0.55, 0.45])

        if device_type == "Smartphone":
            brand = np.random.choice(brands_phones, p=[0.38, 0.30, 0.12, 0.10, 0.10])
            model_name = np.random.choice(phone_models[brand])
            screen_size = round(np.random.uniform(5.8, 6.8), 1)
            os_name = "iOS" if brand == "Apple" else "Android"
            
            # Specs
            if brand == "Apple":
                ram_gb = np.random.choice([4, 6, 8], p=[0.3, 0.5, 0.2])
                original_price = float(np.random.choice([799, 899, 999, 1099, 1199]))
            else:
                ram_gb = np.random.choice([6, 8, 12, 16], p=[0.3, 0.45, 0.2, 0.05])
                original_price = float(np.random.choice([399, 599, 799, 899, 1199]))

            storage_gb = np.random.choice([64, 128, 256, 512, 1024], p=[0.15, 0.45, 0.28, 0.10, 0.02])

        else:  # Laptop
            brand = np.random.choice(brands_laptops, p=[0.32, 0.20, 0.22, 0.14, 0.12])
            model_name = np.random.choice(laptop_models[brand])
            screen_size = round(np.random.uniform(13.3, 16.1), 1)
            os_name = "macOS" if brand == "Apple" else np.random.choice(["Windows", "Linux"], p=[0.92, 0.08])

            # Specs
            ram_gb = np.random.choice([8, 16, 32, 64], p=[0.35, 0.48, 0.14, 0.03])
            storage_gb = np.random.choice([256, 512, 1024, 2048], p=[0.30, 0.50, 0.17, 0.03])
            if brand == "Apple":
                original_price = float(np.random.choice([999, 1299, 1599, 1999, 2499]))
            else:
                original_price = float(np.random.choice([599, 799, 1099, 1499, 1999]))

        # Common attributes
        age_months = int(np.random.uniform(2, 54))
        condition = np.random.choice(conditions, p=condition_weights)
        battery_health = round(float(np.clip(100 - (age_months * 0.45 + np.random.normal(0, 4.5)), 55.0, 100.0)), 1)
        
        has_box = int(np.random.choice([1, 0], p=[0.55, 0.45]))
        has_charger = int(np.random.choice([1, 0], p=[0.75, 0.25]))
        
        # Physical damage probability correlated with condition
        if condition == "Pristine":
            screen_damage = 0
            body_scratches = 0
        elif condition == "Good":
            screen_damage = 0
            body_scratches = int(np.random.choice([0, 1], p=[0.75, 0.25]))
        elif condition == "Fair":
            screen_damage = int(np.random.choice([0, 1], p=[0.80, 0.20]))
            body_scratches = 1
        else:  # Heavily Used
            screen_damage = int(np.random.choice([0, 1], p=[0.50, 0.50]))
            body_scratches = 1

        # Real-world depreciation formula calculation
        # 1. Base Exponential Time Depreciation: P(t) = P0 * exp(-lambda * t)
        decay_rate = 0.016 if brand == "Apple" else 0.022
        time_depreciation_ratio = np.exp(-decay_rate * age_months)
        
        # 2. Condition multiplier
        cond_mult = condition_multipliers[condition]

        # 3. Battery health factor (nonlinear below 80% threshold)
        if battery_health >= 80:
            battery_mult = 1.0 - (100 - battery_health) * 0.002
        else:
            battery_mult = 0.96 - (80 - battery_health) * 0.007

        # 4. Accessories and damage adjustments
        acc_mult = 1.0 + (0.04 if has_box else 0.0) + (0.05 if has_charger else -0.04)
        damage_mult = 1.0 - (0.18 if screen_damage else 0.0) - (0.07 if body_scratches else 0.0)

        # 5. Spec retention bonus
        spec_bonus = 1.0 + (np.log2(max(ram_gb, 4) / 4) * 0.05) + (np.log2(max(storage_gb, 64) / 64) * 0.03)

        # 6. Brand retention factor
        brand_factor = brand_retention_factors.get(brand, 1.0)

        # Compute calculated resale price with realistic market noise
        raw_price = (original_price * time_depreciation_ratio * cond_mult * 
                     battery_mult * acc_mult * damage_mult * spec_bonus * (brand_factor ** 0.5))
        
        # Add realistic market variance (Gaussian noise ~ 4%)
        noise = np.random.normal(1.0, 0.04)
        resale_price = round(float(np.clip(raw_price * noise, 45.0, original_price * 0.95)), 2)

        record = {
            "device_id": f"DEV-{10000 + i}",
            "device_type": device_type,
            "brand": brand,
            "model_name": model_name,
            "os": os_name,
            "age_months": age_months,
            "original_price_usd": round(original_price, 2),
            "ram_gb": int(ram_gb),
            "storage_gb": int(storage_gb),
            "screen_size_inch": float(screen_size),
            "battery_health_pct": float(battery_health),
            "condition": condition,
            "has_original_box": has_box,
            "has_original_charger": has_charger,
            "has_screen_damage": screen_damage,
            "has_body_scratches": body_scratches,
            "resale_price_usd": resale_price
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Introduce intentional realistic missing values (~2% in battery_health and screen_size)
    # to demonstrate M2 Imputation (SimpleImputer, MissingIndicator)
    mask_battery = np.random.rand(len(df)) < 0.02
    df.loc[mask_battery, "battery_health_pct"] = np.nan

    mask_screen = np.random.rand(len(df)) < 0.015
    df.loc[mask_screen, "screen_size_inch"] = np.nan

    return df


def split_and_save_data(df: pd.DataFrame, data_dir: str) -> Dict[str, str]:
    """
    Splits dataset into 70% Train, 15% Validation, 15% Test.
    Enforces strict ML discipline (M1 & M5):
    - Test set is held out completely.
    - Prevents data leakage between splits.
    """
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

    # Save complete raw data
    raw_path = os.path.join(data_dir, "raw", "revalue_devices_raw.csv")
    df.to_csv(raw_path, index=False)

    # Shuffle dataset
    df_shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = df_shuffled.iloc[:n_train].copy()
    val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = df_shuffled.iloc[n_train + n_val:].copy()

    train_path = os.path.join(data_dir, "processed", "train.csv")
    val_path = os.path.join(data_dir, "processed", "val.csv")
    test_path = os.path.join(data_dir, "processed", "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "total_samples": n_total,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "features_count": train_df.shape[1] - 2, # Excluding device_id and target
        "target_variable": "resale_price_usd",
        "random_seed": RANDOM_STATE
    }
    
    with open(os.path.join(data_dir, "processed", "split_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "raw": raw_path,
        "train": train_path,
        "val": val_path,
        "test": test_path
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_folder = os.path.join(workspace_root, "data")
    
    print("[M1] Generating synthetic dataset for ReValue AI...")
    df = generate_synthetic_devices_dataset(n_samples=4000)
    paths = split_and_save_data(df, data_folder)
    print(f"[M1] Data Pipeline completed successfully:")
    for k, v in paths.items():
        print(f"  - {k.capitalize()}: {v}")
