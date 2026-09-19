"""Precompute Aboveground Biomass (AGB) grid across the study region.

Generates a ~250m resolution grid across [76.20, 11.20, 76.90, 11.90],
extracts the 12 features from the latest cloud-free Sentinel-2/1 + NASADEM + ESA WorldCover,
predicts AGB using models/agb_model_final_v4.joblib,
masks non-forest cells (nulling out AGB/carbon/CO2),
and saves the result to data/processed/agb_grid.parquet.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import rasterio

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config, init_earth_engine, PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.precompute_grid")

FEATURE_COLS = [
    "B2", "B3", "B4", "B8", "B11", "B12", "NDVI",
    "VV", "VH", "VH_VV_ratio",
    "elevation", "slope"
]

def build_feature_stack(cfg: dict, ee_region: ee.Geometry) -> ee.Image:
    """Build multi-band Earth Engine composite image containing all 12 features + WorldCover."""
    # 1. Sentinel-2 Surface Reflectance (cloud-masked using Cloud Score Plus)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(ee_region) \
        .filterDate("2024-11-01", "2024-12-31")
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(ee_region)
    linked = ee.Join.saveFirst("cs").apply(
        primary=s2, secondary=cs,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index")
    )

    def mask_and_scale(img):
        img = ee.Image(img)
        c = ee.Image(img.get("cs")).select("cs")
        clear = c.gte(0.6)
        refl = img.select(["B2", "B3", "B4", "B8", "B11", "B12"]).multiply(10000.0).updateMask(clear)
        ndvi = refl.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return refl.addBands(ndvi)

    s2_comp = ee.ImageCollection(linked).map(mask_and_scale).median()

    # 2. Sentinel-1 SAR composite (VV, VH, VH/VV ratio)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(ee_region) \
        .filterDate("2024-11-01", "2024-12-31") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    s1_med = s1.select(["VV", "VH"]).median()
    # Linear ratio: 10^((VH - VV)/10)
    vh_vv_ratio = ee.Image(10.0).pow(s1_med.select("VH").subtract(s1_med.select("VV")).divide(10.0)).rename("VH_VV_ratio")
    s1_comp = s1_med.addBands(vh_vv_ratio)

    # 3. NASADEM elevation & slope
    dem = ee.Image(cfg["datasets"]["auxiliary"]["dem"]).select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")

    # 4. ESA WorldCover (Class 10 is tree cover)
    wc = ee.ImageCollection(cfg["datasets"]["auxiliary"]["landcover"]).first().select("Map").rename("worldcover")

    # Combine into 13-band image
    stack = ee.Image.cat([
        s2_comp.select(["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]),
        s1_comp.select(["VV", "VH", "VH_VV_ratio"]),
        dem,
        slope,
        wc
    ]).clip(ee_region)

    return stack


def run(scale_m: float = 250.0) -> Path:
    start_time = time.time()
    setup_logging()
    cfg = load_config(PROJECT_ROOT / "config/config.yaml")
    init_earth_engine(cfg)

    exp_bbox = cfg["study_region"]["bbox"]  # [76.20, 11.20, 76.90, 11.90]
    ee_region = ee.Geometry.BBox(*exp_bbox)
    logger.info("Generating precomputed grid for region %s at ~%dm resolution...", exp_bbox, scale_m)

    cache_dir = PROJECT_ROOT / "data/raw/auxiliary"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tif_path = cache_dir / f"agb_grid_{int(scale_m)}m_stack.tif"

    if not tif_path.exists():
        # 1. Build EE composite
        stack = build_feature_stack(cfg, ee_region)
        logger.info("Constructed Earth Engine 13-band image stack: %s", stack.bandNames().getInfo())

        # 2. Download raster GeoTIFF
        logger.info("Requesting download URL from Earth Engine (scale=%dm)...", scale_m)
        url = stack.getDownloadURL({
            "scale": scale_m,
            "crs": "EPSG:4326",
            "region": ee_region,
            "format": "GEO_TIFF"
        })
        logger.info("Downloading raster stack to %s ...", tif_path)
        urllib.request.urlretrieve(url, tif_path)
        logger.info("Downloaded GeoTIFF (%.2f MB).", tif_path.stat().st_size / (1024 * 1024))
    else:
        logger.info("Found cached GeoTIFF at %s (%.2f MB). Reusing raster.", tif_path, tif_path.stat().st_size / (1024 * 1024))

    # 3. Read raster and coordinates
    with rasterio.open(tif_path) as ds:
        arr = ds.read()  # shape: (13, height, width)
        height, width = ds.shape
        transform = ds.transform
        logger.info("Raster grid dimensions: %d rows x %d cols (%d total cells)", height, width, height * width)

        # Generate lat/lon for each cell center
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel(), offset="center")
        lons = np.array(xs)
        lats = np.array(ys)

    # 4. Prepare feature matrix X (12 features) and WorldCover mask
    X_raw = arr[:12].reshape(12, -1).T
    wc = arr[12].reshape(-1)

    valid_sensor_mask = np.all(np.isfinite(X_raw), axis=1)
    is_forest = valid_sensor_mask & (wc == 10)
    forest_count = int(np.sum(is_forest))
    non_forest_count = len(is_forest) - forest_count
    logger.info("Valid cells: %d | Forest cells (Class 10): %d (%.1f%%) | Non-forest: %d",
                int(np.sum(valid_sensor_mask)), forest_count, 100.0 * forest_count / len(is_forest), non_forest_count)

    # 5. Load model and predict
    model_path = PROJECT_ROOT / "models/agb_model_final_v4.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logger.info("Loading locked-in v4 model from %s...", model_path)
    model = joblib.load(model_path)

    agb_preds = np.full(len(X_raw), np.nan, dtype=np.float32)
    if forest_count > 0:
        logger.info("Running inference on %d forest cells...", forest_count)
        # Wrap in DataFrame with exact feature names to prevent sklearn UserWarning
        X_forest_df = pd.DataFrame(X_raw[is_forest], columns=FEATURE_COLS)
        agb_preds[is_forest] = model.predict(X_forest_df).astype(np.float32)

    # 6. Carbon, CO2-equivalent, and Canopy Density calculations
    # Carbon = AGB * 0.47, CO2-equivalent = Carbon * 3.67
    carbon_preds = np.where(is_forest, agb_preds * 0.47, np.nan).astype(np.float32)
    co2_preds = np.where(is_forest, carbon_preds * 3.67, np.nan).astype(np.float32)

    # Canopy density proxy via Sentinel-2 NDVI (band index 6 in FEATURE_COLS)
    # Higher NDVI within forest-classified cells reflects denser canopy cover.
    ndvi_raw = X_raw[:, 6]
    ndvi_canopy = np.where(is_forest, ndvi_raw, np.nan).astype(np.float32)
    canopy_density_pct = np.where(
        is_forest,
        np.round(np.clip(ndvi_raw, 0.0, 1.0) * 100.0, 1),
        np.nan
    ).astype(np.float32)

    # 7. Assemble DataFrame
    grid_df = pd.DataFrame({
        "lat": lats.astype(np.float32),
        "lon": lons.astype(np.float32),
        "agb_mg_ha": agb_preds,
        "carbon_tons_per_ha": carbon_preds,
        "co2_equivalent_tons_per_ha": co2_preds,
        "ndvi_canopy_proxy": ndvi_canopy,
        "canopy_density_pct": canopy_density_pct,
        "is_forest": is_forest.astype(bool),
    })

    out_parquet = PROJECT_ROOT / "data/processed/agb_grid.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(out_parquet, index=False)

    runtime_s = time.time() - start_time
    logger.info("Successfully saved %d grid cells to %s (runtime: %.1fs).", len(grid_df), out_parquet, runtime_s)

    # Summary report
    print("\n" + "=" * 65)
    print("PRECOMPUTE AGB GRID EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Bounding Box           : {exp_bbox}")
    print(f"Grid Resolution        : ~{int(scale_m)}m")
    print(f"Grid Dimensions        : {height} x {width}")
    print(f"Total Grid Cells       : {len(grid_df):,}")
    print(f"Forest Cells (Class 10): {forest_count:,} ({100.0 * forest_count / len(grid_df):.1f}%)")
    print(f"Non-Forest Cells       : {non_forest_count:,}")
    print(f"Precompute Runtime     : {runtime_s:.2f} seconds")
    print(f"Output File            : {out_parquet}")
    print(f"File Size              : {out_parquet.stat().st_size / (1024 * 1024):.2f} MB")
    print("-" * 65)
    if forest_count > 0:
        forest_s = grid_df.loc[grid_df["is_forest"], "agb_mg_ha"]
        print("Forest AGB Distribution (Mg/ha):")
        print(f"  Mean: {forest_s.mean():.2f} | Std: {forest_s.std():.2f} | Min: {forest_s.min():.2f} | Median: {forest_s.median():.2f} | Max: {forest_s.max():.2f}")
        c_s = grid_df.loc[grid_df["is_forest"], "carbon_tons_per_ha"]
        print(f"Forest Carbon Mean         : {c_s.mean():.2f} t/ha")
        co2_s = grid_df.loc[grid_df["is_forest"], "co2_equivalent_tons_per_ha"]
        print(f"Forest CO2-equivalent Mean : {co2_s.mean():.2f} t/ha")
        canopy_s = grid_df.loc[grid_df["is_forest"], "ndvi_canopy_proxy"]
        print(f"Forest Canopy Density (NDVI): Mean {canopy_s.mean():.4f} | Min {canopy_s.min():.4f} | Max {canopy_s.max():.4f}")
        canopy_pct_s = grid_df.loc[grid_df["is_forest"], "canopy_density_pct"]
        print(f"Forest Canopy Density (%)   : Mean {canopy_pct_s.mean():.1f}% | Min {canopy_pct_s.min():.1f}% | Max {canopy_pct_s.max():.1f}%")
        corr = float(np.corrcoef(canopy_s.values, forest_s.values)[0, 1])
        print(f"Canopy-AGB Pearson Corr    : +{corr:.4f}")
    print("=" * 65)

    return out_parquet


if __name__ == "__main__":
    run()
