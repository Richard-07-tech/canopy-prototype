"""Train improved AGB prediction model (v2) with derived vegetation indices
and light hyperparameter tuning via GridSearchCV.

Adds 5 derived features computed from existing bands:
  - EVI (Enhanced Vegetation Index)
  - SAVI (Soil Adjusted Vegetation Index)
  - NBR-like ratio (B8-B12)/(B8+B12)
  - NIR/Red ratio (B8/B4)
  - SWIR1/NIR ratio (B11/B8)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from common import PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.train_model_v2")

ORIGINAL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
DERIVED_FEATURES = ["EVI", "SAVI", "NBR", "NIR_Red_ratio", "SWIR1_NIR_ratio"]
ALL_FEATURES = ORIGINAL_FEATURES + DERIVED_FEATURES
TARGET_COL = "agb"

# Baseline results for comparison
BASELINE_R2 = 0.3580
BASELINE_MAE = 38.46


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived vegetation indices from existing Sentinel-2 bands."""
    df = df.copy()

    B2 = df["B2"].astype(float)
    B4 = df["B4"].astype(float)
    B8 = df["B8"].astype(float)
    B11 = df["B11"].astype(float)
    B12 = df["B12"].astype(float)

    # EVI: 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
    evi_denom = B8 + 6.0 * B4 - 7.5 * B2 + 1.0
    df["EVI"] = np.where(np.abs(evi_denom) > 1e-6, 2.5 * (B8 - B4) / evi_denom, 0.0)

    # SAVI: ((B8 - B4) / (B8 + B4 + 0.5)) * 1.5
    savi_denom = B8 + B4 + 0.5
    df["SAVI"] = np.where(np.abs(savi_denom) > 1e-6, ((B8 - B4) / savi_denom) * 1.5, 0.0)

    # NBR-like: (B8 - B12) / (B8 + B12)
    nbr_denom = B8 + B12
    df["NBR"] = np.where(np.abs(nbr_denom) > 1e-6, (B8 - B12) / nbr_denom, 0.0)

    # NIR/Red ratio: B8 / B4
    df["NIR_Red_ratio"] = np.where(B4 > 1e-6, B8 / B4, 0.0)

    # SWIR1/NIR ratio: B11 / B8
    df["SWIR1_NIR_ratio"] = np.where(B8 > 1e-6, B11 / B8, 0.0)

    # Replace any inf/nan from edge cases
    for col in DERIVED_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def run() -> None:
    setup_logging()

    # 1a. Load dataset
    data_path = PROJECT_ROOT / "data/processed/training_dataset.parquet"
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    # 1b. Compute derived features
    df = add_derived_features(df)
    logger.info("Added %d derived features: %s", len(DERIVED_FEATURES), DERIVED_FEATURES)

    # 1c. Select features and target
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    logger.info("Feature matrix: %s (%d features) | Target: %s", X.shape, len(ALL_FEATURES), TARGET_COL)

    # 1d. Train/test split (same as v1 for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 1e. GridSearchCV for light hyperparameter tuning
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3],
    }
    base_rf = RandomForestRegressor(
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Running GridSearchCV (5-fold) over %d combos ...",
                len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_leaf"]))

    grid = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_r2 = grid.best_score_

    logger.info("Best params: %s (CV R2=%.4f)", best_params, best_cv_r2)

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("Test R2=%.4f | MAE=%.2f Mg/ha", r2, mae)

    # Feature importances
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)

    # Decide if v2 is meaningfully better
    r2_improvement = r2 - BASELINE_R2
    mae_improvement = BASELINE_MAE - mae
    is_better = r2_improvement >= 0.03

    # Save model if better
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "agb_model_v2.joblib"

    if is_better:
        joblib.dump(best_model, model_path)
        logger.info("v2 is better -- saved to %s", model_path)
    else:
        logger.info("v2 improvement is marginal (dR2=%.4f) -- not saving as primary model", r2_improvement)

    # Save test predictions regardless
    test_preds_df = pd.DataFrame({
        "actual_agb": y_test.values,
        "predicted_agb": y_pred,
        "residual": y_test.values - y_pred,
    }, index=y_test.index)
    for col in ["shot_number", "latitude", "longitude", "gedi_date", "sentinel2_date"]:
        if col in df.columns:
            test_preds_df[col] = df.loc[y_test.index, col].values

    preds_path = PROJECT_ROOT / "data/processed/test_predictions_v2.parquet"
    test_preds_df.to_parquet(preds_path, index=False)

    # Print summary
    print("\n" + "=" * 65)
    print("MODEL COMPARISON: v1 (baseline) vs v2 (improved)")
    print("=" * 65)
    print(f"{'Metric':<28s} {'v1 (baseline)':>14s} {'v2 (improved)':>14s}")
    print("-" * 65)
    print(f"{'Features':<28s} {'7':>14s} {'12':>14s}")
    print(f"{'Best Params':<28s} {'defaults':>14s} {str(best_params):>14s}")
    print(f"{'CV R2 (5-fold, train)':<28s} {'N/A':>14s} {best_cv_r2:>14.4f}")
    print(f"{'Test R2':<28s} {BASELINE_R2:>14.4f} {r2:>14.4f}")
    print(f"{'Test MAE (Mg/ha)':<28s} {BASELINE_MAE:>14.2f} {mae:>14.2f}")
    print(f"{'R2 improvement':<28s} {'':>14s} {r2_improvement:>+14.4f}")
    print(f"{'MAE improvement':<28s} {'':>14s} {mae_improvement:>+14.2f}")
    print()
    print("Feature Importances (v2, sorted):")
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank:>2d}. {feat:<18s}  {imp:.4f}  {bar}")
    print()

    if is_better:
        print(">>> VERDICT: v2 IS BETTER. Use models/agb_model_v2.joblib.")
        print(f"    Saved model  : {model_path}")
    else:
        print(">>> VERDICT: v2 improvement is marginal or negligible.")
        print(f"    R2 delta = {r2_improvement:+.4f} (threshold: >= +0.03)")
        print("    The baseline model remains a reasonable choice.")
        if r2_improvement > 0:
            # Save anyway for reference, user can decide
            joblib.dump(best_model, model_path)
            print(f"    Saved v2 anyway for reference: {model_path}")

    print(f"    Test predictions: {preds_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run()
