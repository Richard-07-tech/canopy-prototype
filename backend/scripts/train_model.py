"""Train a baseline AGB (biomass) prediction model using Random Forest.

Uses Sentinel-2 spectral bands (B2, B3, B4, B8, B11, B12, NDVI) as features
and GEDI L4A AGB as the target variable.
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

logger = logging.getLogger("biomass_pipeline.train_model")

FEATURE_COLS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
TARGET_COL = "agb"


def run() -> None:
    setup_logging()

    # 1a. Load dataset
    data_path = PROJECT_ROOT / "data/processed/training_dataset.parquet"
    if not data_path.exists():
        raise RuntimeError(f"Training dataset not found at {data_path}")

    df = pd.read_parquet(data_path)
    logger.info("Loaded training dataset: %d rows, %d columns from %s", len(df), len(df.columns), data_path)

    # 1b. Select features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    logger.info("Features: %s | Target: %s", FEATURE_COLS, TARGET_COL)
    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)

    # 1c. Train/test split (80/20, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("Train set: %d samples | Test set: %d samples", len(X_train), len(X_test))

    # 1d. Train Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training RandomForestRegressor (n_estimators=200) ...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 1e. Evaluate on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("Test R² = %.4f | Test MAE = %.2f Mg/ha", r2, mae)

    # 1f. Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

    # 2. Save model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "agb_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # 3. Save test predictions
    test_preds_df = pd.DataFrame({
        "actual_agb": y_test.values,
        "predicted_agb": y_pred,
        "residual": y_test.values - y_pred,
    }, index=y_test.index)
    # Include original metadata for inspection
    for col in ["shot_number", "latitude", "longitude", "gedi_date", "sentinel2_date"]:
        if col in df.columns:
            test_preds_df[col] = df.loc[y_test.index, col].values

    preds_path = PROJECT_ROOT / "data/processed/test_predictions.parquet"
    test_preds_df.to_parquet(preds_path, index=False)
    logger.info("Test predictions saved to %s", preds_path)

    # 4. Print summary
    print("\n" + "=" * 65)
    print("MODEL TRAINING SUMMARY")
    print("=" * 65)
    print(f"Dataset              : {data_path}")
    print(f"Total Samples        : {len(df)}")
    print(f"Train / Test Split   : {len(X_train)} / {len(X_test)}")
    print(f"Model                : RandomForestRegressor (n_estimators=200)")
    print()
    print(f"Test R² Score        : {r2:.4f}")
    print(f"Test MAE             : {mae:.2f} Mg/ha")
    print()
    print("Feature Importances (sorted):")
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank}. {feat:<6s}  {imp:.4f}  {bar}")
    print()
    print(f"Top 3 Features       : {feat_imp[0][0]}, {feat_imp[1][0]}, {feat_imp[2][0]}")
    print()
    print(f"Saved Model          : {model_path}")
    print(f"Saved Predictions    : {preds_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run()
