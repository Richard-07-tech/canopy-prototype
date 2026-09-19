"""Train final baseline Random Forest model (v4: Optical + SAR + Topography on Expanded Forest-Masked Strict Dataset).

Evaluates all four model configurations side by side:
- v1 Baseline (Original 1,326, non-strict temporal matching, 7 Optical features)
- v1-strict (Original Nilgiris 97 samples, strict <= 30-day temporal fix only, 7 Optical features)
- v3 (Original 1,326, non-strict temporal matching, 10 Optical + SAR features)
- v4 (Expanded region 192 samples, WorldCover forest-filtered, NASADEM elevation/slope, strict <= 30-day matching, 12 Optical + SAR + Topo features)
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.train_model_v4")

OPTICAL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
SAR_FEATURES = ["VV", "VH", "VH_VV_ratio"]
TOPO_FEATURES = ["elevation", "slope"]
V4_FEATURES = OPTICAL_FEATURES + SAR_FEATURES + TOPO_FEATURES
TARGET_COL = "agb"

def run():
    setup_logging()

    # 1. Load v4 dataset
    data_path = PROJECT_ROOT / "data/processed/training_dataset_v4.parquet"
    if not data_path.exists():
        raise RuntimeError(f"v4 dataset not found at {data_path}")

    df_v4 = pd.read_parquet(data_path)
    logger.info("Loaded v4 dataset: %d rows, %d columns from %s", len(df_v4), len(df_v4.columns), data_path)

    # 2. Prepare X and y
    X = df_v4[V4_FEATURES].copy()
    y = df_v4[TARGET_COL].copy()

    # 3. Train/Test Split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("v4 Train set: %d samples | Test set: %d samples", len(X_train), len(X_test))

    # 4. Train RandomForestRegressor (Identical hyperparameters to v1)
    model_v4 = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training v4 RandomForestRegressor (n_estimators=200, max_features='sqrt')...")
    model_v4.fit(X_train, y_train)
    logger.info("Training complete.")

    # 5. Evaluate v4
    y_pred_v4 = model_v4.predict(X_test)
    r2_v4 = r2_score(y_test, y_pred_v4)
    mae_v4 = mean_absolute_error(y_test, y_pred_v4)

    # 6. Feature importances
    importances = model_v4.feature_importances_
    feat_imp = sorted(zip(V4_FEATURES, importances), key=lambda x: x[1], reverse=True)

    # 7. Save model and predictions
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "agb_model_final_v4.joblib"
    joblib.dump(model_v4, model_path)
    logger.info("Saved final v4 model to %s", model_path)

    preds_df = pd.DataFrame({
        "actual_agb": y_test.values,
        "predicted_agb": y_pred_v4,
        "residual": y_test.values - y_pred_v4,
    }, index=y_test.index)
    for col in ["shot_number", "latitude", "longitude", "gedi_date", "sentinel2_date", "sentinel1_date"]:
        if col in df_v4.columns:
            preds_df[col] = df_v4.loc[y_test.index, col].values

    preds_path = PROJECT_ROOT / "data/processed/test_predictions_final_v4.parquet"
    preds_df.to_parquet(preds_path, index=False)
    logger.info("Saved predictions to %s", preds_path)

    # 8. Compute v1-strict benchmark
    orig_path = PROJECT_ROOT / "data/processed/training_dataset.parquet"
    df_orig = pd.read_parquet(orig_path)
    strict_orig = df_orig[df_orig["sentinel2_day_difference"] <= 30].copy()
    X_str = strict_orig[OPTICAL_FEATURES]
    y_str = strict_orig[TARGET_COL]
    X_str_tr, X_str_te, y_str_tr, y_str_te = train_test_split(
        X_str, y_str, test_size=0.20, random_state=42
    )
    model_v1_strict = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    model_v1_strict.fit(X_str_tr, y_str_tr)
    pred_v1_strict = model_v1_strict.predict(X_str_te)
    r2_v1_strict = r2_score(y_str_te, pred_v1_strict)
    mae_v1_strict = mean_absolute_error(y_str_te, pred_v1_strict)

    # Historical values
    r2_v1 = 0.3580
    mae_v1 = 38.46
    r2_v3 = 0.3615
    mae_v3 = 38.39

    # Print Full Comparative Summary
    print("\n" + "=" * 80)
    print("FINAL MODELING ITERATION: 4-WAY COMPARATIVE SUMMARY")
    print("=" * 80)
    print(f"{'Model Version':<22} | {'Samples':<8} | {'Features':<9} | {'Test R²':<10} | {'Test MAE (Mg/ha)':<16} | {'Temporal Policy'}")
    print("-" * 80)
    print(f"{'v1 Baseline':<22} | {'1,326':<8} | {'7 (Opt)':<9} | {r2_v1:<10.4f} | {mae_v1:<16.2f} | {'Non-strict (up to 166d)'}")
    print(f"{'v1-strict':<22} | {len(strict_orig):<8} | {'7 (Opt)':<9} | {r2_v1_strict:<10.4f} | {mae_v1_strict:<16.2f} | {'Strict (<=30d, orig region)'}")
    print(f"{'v3 (Optical+SAR)':<22} | {'1,326':<8} | {'10 (SAR)':<9} | {r2_v3:<10.4f} | {mae_v3:<16.2f} | {'Non-strict (up to 166d)'}")
    print(f"{'v4 (Final Combined)':<22} | {len(df_v4):<8} | {len(V4_FEATURES):<9} | {r2_v4:<10.4f} | {mae_v4:<16.2f} | {'Strict (<=30d, expanded)'}")
    print("=" * 80)
    print("\nv4 Feature Importances:")
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        tag = "[Topo]" if feat in TOPO_FEATURES else ("[SAR] " if feat in SAR_FEATURES else "[Opt] ")
        bar = "#" * int(imp * 50)
        print(f"  {rank:2d}. {tag} {feat:<12s} {imp:.4f}  {bar}")
    print("=" * 80)

if __name__ == "__main__":
    run()
