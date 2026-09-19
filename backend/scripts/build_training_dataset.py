"""Build ML Training Dataset by matching GEDI AGB points to Sentinel-2 imagery.

For each GEDI footprint point (~25m radius):
1. Finds the closest-in-time available Sentinel-2 acquisition date with valid pixel coverage.
2. Uses rasterio windowed extraction (25m radius window) to compute pixel means
   for bands: B2, B3, B4, B8, B11, B12, and NDVI.
3. Drops rows falling outside coverage or containing nodata/NaN values (logging drop counts).
4. Saves final dataset to data/processed/training_dataset.parquet.
"""
from __future__ import annotations

import logging
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from common import load_config, PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.build_training")

# Standard band list and mapping to output column names
S2_BANDS = [
    ("B02", "B2"),
    ("B03", "B3"),
    ("B04", "B4"),
    ("B08", "B8"),
    ("B11", "B11"),
    ("B12", "B12"),
    ("NDVI", "NDVI"),
]


class InMemoryDayRasterCache:
    """Fast in-memory cache for Sentinel-2 rasters per acquisition date.
    
    Loads raw band arrays into RAM once per date, enabling fast window
    slicing and pixel averaging across footprint points.
    """

    def __init__(self, s2_idx: dict[str, dict[str, Path]], radius_m: float = 25.0):
        self.s2_idx = s2_idx
        self.radius_m = radius_m
        self._cache: dict[str, dict[str, Any]] = {}

    def get_day_cache(self, date_str: str) -> dict[str, Any] | None:
        if date_str in self._cache:
            return self._cache[date_str]

        day_bands = self.s2_idx.get(date_str)
        if not day_bands:
            return None

        # Find reference tif to get transform and spatial dimensions
        ref_ds = None
        for stem, _ in S2_BANDS:
            tif = day_bands.get(stem)
            if tif and Path(tif).exists():
                ref_ds = rasterio.open(tif)
                break

        if ref_ds is None:
            return None

        transform = ref_ds.transform
        crs = ref_ds.crs
        width = ref_ds.width
        height = ref_ds.height
        ref_ds.close()

        bands_data: dict[str, np.ndarray] = {}
        for stem, out_col in S2_BANDS:
            tif = day_bands.get(stem)
            if not tif or not Path(tif).exists():
                return None
            with rasterio.open(tif) as ds:
                bands_data[out_col] = ds.read(1, masked=False)

        day_entry = {
            "transform": transform,
            "crs": crs,
            "width": width,
            "height": height,
            "bands": bands_data,
        }
        self._cache[date_str] = day_entry
        return day_entry

    def sample_all_bands(
        self, date_str: str, lon: float, lat: float
    ) -> dict[str, float] | None:
        day_entry = self.get_day_cache(date_str)
        if day_entry is None:
            return None

        try:
            transform = day_entry["transform"]
            width = day_entry["width"]
            height = day_entry["height"]
            bands = day_entry["bands"]

            rc, cc = rasterio.transform.rowcol(transform, lon, lat)
            rmin, rmax = max(0, rc - 2), min(height, rc + 3)
            cmin, cmax = max(0, cc - 2), min(width, cc + 3)

            if rmin >= rmax or cmin >= cmax:
                return None

            band_results: dict[str, float] = {}
            for _, out_col in S2_BANDS:
                raw_arr = bands[out_col]
                sub = raw_arr[rmin:rmax, cmin:cmax].ravel()
                
                # Filter nodata: NDVI can be negative, reflectance bands > 0
                if out_col == "NDVI":
                    valid = np.isfinite(sub)
                else:
                    valid = np.isfinite(sub) & (sub > 0)
                    
                vals = sub[valid]
                if vals.size == 0:
                    return None
                band_results[out_col] = float(np.mean(vals))

            return band_results
        except Exception:
            return None


def _index_s2_imagery(raw_root: Path) -> dict[str, dict[str, Path]]:
    """Index available S2 GeoTIFFs: date_str -> {band_name: path}"""
    idx: dict[str, dict[str, Path]] = {}
    if not raw_root.exists():
        return idx
    for tif in raw_root.glob("*/*/*.tif"):
        day = tif.parent.name
        try:
            dt.datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        idx.setdefault(day, {})[tif.stem] = tif
    return idx


def run(cfg: dict | None = None) -> Path:
    setup_logging()
    cfg = cfg or load_config(PROJECT_ROOT / "config/config.yaml")

    gedi_path = PROJECT_ROOT / "data/raw/gedi/gedi_samples.parquet"
    if not gedi_path.exists():
        raise RuntimeError(f"GEDI samples file not found at {gedi_path}")

    gdf = pd.read_parquet(gedi_path)
    if gdf.empty:
        raise RuntimeError("GEDI samples table is empty.")

    logger.info("Loaded %d GEDI samples from %s", len(gdf), gedi_path)

    # Index available Sentinel-2 imagery
    s2_raw_root = PROJECT_ROOT / "data/raw/sentinel2"
    s2_idx = _index_s2_imagery(s2_raw_root)
    s2_dates = list(s2_idx.keys())
    logger.info("Found %d Sentinel-2 acquisition dates in local raw storage: %s", len(s2_dates), sorted(s2_dates))

    sampler = InMemoryDayRasterCache(s2_idx, radius_m=25.0)

    rows: list[dict[str, Any]] = []
    dropped_counts = {
        "no_s2_date_matched": 0,
        "missing_or_nodata_band_values": 0,
        "outside_coverage": 0,
    }

    for i, row in enumerate(gdf.itertuples(index=False)):
        gedi_date_str = str(row.acquisition_date)
        lon, lat = float(row.longitude), float(row.latitude)
        agb_val = float(row.gedi_agb)
        g_d = dt.datetime.strptime(gedi_date_str, "%Y-%m-%d").date()

        # Sort candidate dates by closest calendar day difference
        sorted_s2_dates = sorted(
            s2_dates,
            key=lambda cand: abs((dt.datetime.strptime(cand, "%Y-%m-%d").date() - g_d).days)
        )

        best_date = None
        best_diff = None
        band_vals = None

        for cand_date in sorted_s2_dates:
            cand_diff = abs((dt.datetime.strptime(cand_date, "%Y-%m-%d").date() - g_d).days)
            vals = sampler.sample_all_bands(cand_date, lon, lat)
            if vals is not None:
                best_date = cand_date
                best_diff = cand_diff
                band_vals = vals
                break

        if band_vals is None or best_date is None:
            dropped_counts["missing_or_nodata_band_values"] += 1
            continue

        # Build feature row
        out_row = {
            "shot_number": getattr(row, "shot_number", None),
            "latitude": lat,
            "longitude": lon,
            "agb": agb_val,
            "gedi_quality": getattr(row, "gedi_quality", None),
            "gedi_sensitivity": getattr(row, "sensitivity", None),
            "gedi_date": gedi_date_str,
            "sentinel2_date": best_date,
            "sentinel2_day_difference": best_diff,
        }
        out_row.update(band_vals)
        rows.append(out_row)

    total_input = len(gdf)
    final_count = len(rows)
    total_dropped = total_input - final_count

    logger.info("Processing complete. Total input: %d | Retained: %d | Dropped: %d",
                total_input, final_count, total_dropped)
    for reason, count in dropped_counts.items():
        if count > 0:
            logger.info("  - Dropped due to '%s': %d rows", reason, count)

    out_df = pd.DataFrame(rows)

    # Extract Sentinel-1 SAR features (VV, VH, VH_VV_ratio)
    try:
        from extract_s1_features import extract_s1_features
        logger.info("Extracting Sentinel-1 SAR features for %d retained samples ...", len(out_df))
        s1_dict = extract_s1_features(out_df)
        if s1_dict:
            s1_df = pd.DataFrame.from_dict(s1_dict, orient="index")
            s1_df.index.name = "shot_number"
            s1_df = s1_df.reset_index()
            out_df["shot_number"] = out_df["shot_number"].astype(str)
            out_df = pd.merge(out_df, s1_df, on="shot_number", how="left")
            logger.info("Joined %d S1 features successfully.", len(s1_df))
    except Exception as e:
        logger.warning("Sentinel-1 extraction failed or skipped: %s", e)

    # Reorder columns logically
    first_cols = ["shot_number", "latitude", "longitude", "agb", "gedi_quality",
                  "gedi_sensitivity", "gedi_date", "sentinel2_date", "sentinel2_day_difference",
                  "sentinel1_date", "sentinel1_day_difference"]
    band_cols = [col_name for _, col_name in S2_BANDS] + ["VV", "VH", "VH_VV_ratio"]
    all_cols = [c for c in first_cols if c in out_df.columns] + [c for c in band_cols if c in out_df.columns]
    out_df = out_df[all_cols]

    # Save outputs
    out_dir_main = PROJECT_ROOT / "data/processed"
    out_dir_main.mkdir(parents=True, exist_ok=True)
    out_path_main = out_dir_main / "training_dataset.parquet"
    out_df.to_parquet(out_path_main, index=False)

    out_dir_sub = PROJECT_ROOT / "data/processed/training"
    out_dir_sub.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_dir_sub / "biomass_training_dataset.parquet", index=False)

    # 4. Print Summary
    print("\n" + "=" * 65)
    print("FINAL TRAINING DATASET BUILD SUMMARY")
    print("=" * 65)
    print(f"Output File Path       : {out_path_main}")
    print(f"Total GEDI Input Rows  : {total_input}")
    print(f"Final Retained Rows    : {final_count}")
    print(f"Total Dropped Rows     : {total_dropped}")
    print("\nBreakdown of Dropped Rows:")
    for reason, cnt in dropped_counts.items():
        print(f"  - {reason}: {cnt}")
    print("\nFinal Dataset Columns:")
    for c in out_df.columns:
        print(f"  - {c}")
    print("=" * 65 + "\n")

    return out_path_main


if __name__ == "__main__":
    run()
