"""Train baseline Random Forest model with optical (Sentinel-2) + SAR (Sentinel-1) features.

Retrains the baseline Random Forest model using the exact same hyperparameters,
random seed (42), and train/test split (80/20) as the v1 optical baseline
to isolate the direct effect of adding Sentinel-1 SAR features (VV, VH, VH/VV ratio).
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

from common import PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.train_model_v3")

OPTICAL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
SAR_FEATURES = ["VV", "VH", "VH_VV_ratio"]
EXPANDED_FEATURES = OPTICAL_FEATURES + SAR_FEATURES
TARGET_COL = "agb"


def run() -> dict:
    setup_logging()

    # 1. Load dataset
    data_path = PROJECT_ROOT / "data/processed/training_dataset.parquet"
    if not data_path.exists():
        raise RuntimeError(f"Training dataset not found at {data_path}")

    df = pd.read_parquet(data_path)
    logger.info("Loaded training dataset: %d rows, %d columns from %s", len(df), len(df.columns), data_path)

    # 2. Select features and target
    X = df[EXPANDED_FEATURES].copy()
    y = df[TARGET_COL].copy()

    logger.info("Features (%d): %s", len(EXPANDED_FEATURES), EXPANDED_FEATURES)
    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)

    # 3. Train/test split (80/20, identical seed to v1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("Train set: %d samples | Test set: %d samples", len(X_train), len(X_test))

    # 4. Baseline Random Forest (identical hyperparameters to v1)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training RandomForestRegressor (n_estimators=200, max_features='sqrt') ...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 5. Evaluate on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("Test R² = %.4f | Test MAE = %.2f Mg/ha", r2, mae)

    # 6. Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(EXPANDED_FEATURES, importances), key=lambda x: x[1], reverse=True)

    # 7. Save model and predictions
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "agb_model_sar_v3.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    preds_df = pd.DataFrame({
        "actual_agb": y_test.values,
        "predicted_agb": y_pred,
        "residual": y_test.values - y_pred,
    }, index=y_test.index)
    for col in ["shot_number", "latitude", "longitude", "gedi_date", "sentinel2_date", "sentinel1_date"]:
        if col in df.columns:
            preds_df[col] = df.loc[y_test.index, col].values

    preds_path = PROJECT_ROOT / "data/processed/test_predictions_sar_v3.parquet"
    preds_df.to_parquet(preds_path, index=False)
    logger.info("Saved predictions to %s", preds_path)

    # Print summary
    print("\n" + "=" * 65)
    print("MODEL RETRAINING SUMMARY: OPTICAL + SAR EXPANDED FEATURE SET")
    print("=" * 65)
    print(f"Dataset              : {data_path}")
    print(f"Total Samples        : {len(df)}")
    print(f"Train / Test Split   : {len(X_train)} / {len(X_test)}")
    print(f"Model                : RandomForestRegressor (n_estimators=200, max_features='sqrt')")
    print()
    print(f"New Feature List ({len(EXPANDED_FEATURES)}) : {', '.join(EXPANDED_FEATURES)}")
    print(f"Test R² Score        : {r2:.4f}")
    print(f"Test MAE             : {mae:.2f} Mg/ha")
    print()
    print("Comparison vs v1 Baseline (Optical Only):")
    print(f"  v1 Baseline R²     : 0.3580")
    print(f"  v3 (Optical+SAR) R²: {r2:.4f}  (Delta: {r2 - 0.3580:+.4f})")
    print(f"  v1 Baseline MAE    : 38.46 Mg/ha")
    print(f"  v3 (Optical+SAR) MAE: {mae:.2f} Mg/ha  (Delta: {mae - 38.46:+.2f} Mg/ha)")
    print()
    print("Feature Importances (sorted):")
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        is_sar = " [SAR]" if feat in SAR_FEATURES else "      "
        bar = "#" * int(imp * 50)
        print(f"  {rank:2d}. {feat:<12s}{is_sar}  {imp:.4f}  {bar}")
    print("=" * 65 + "\n")

    return {
        "features": EXPANDED_FEATURES,
        "r2": r2,
        "mae": mae,
        "feature_importances": feat_imp,
        "model_path": model_path,
        "predictions_path": preds_path,
    }


if __name__ == "__main__":
    run()
