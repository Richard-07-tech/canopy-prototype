"""Extract Sentinel-1 SAR features (VV, VH, VH/VV ratio) for GEDI footprints

Matches each GEDI footprint to the closest Sentinel-1 acquisition in time
within the matching window (max 30 days) and computes footprint-aware
spatial averaging (25m diameter / 12.5m radius circular buffer at 10m scale).
"""
from __future__ import annotations

import logging
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import ee

from common import load_config, init_earth_engine, PROJECT_ROOT, setup_logging

logger = logging.getLogger("biomass_pipeline.extract_s1")

# Candidate scenes per GEDI date
CANDIDATES_BY_DATE = {
    "2024-10-28": [
        ("2024-10-29", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241029T004902_20241029T004927_056310_06E58C_9097"),
        ("2024-10-24", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241024T004054_20241024T004119_056237_06E2A7_EDB7"),
        ("2024-11-05", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241105T004054_20241105T004119_056412_06E99A_912D"),
    ],
    "2024-11-18": [
        ("2024-11-17", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241117T004054_20241117T004119_056587_06F094_45D6"),
        ("2024-11-22", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241122T004901_20241122T004926_056660_06F382_E9FD"),
        ("2024-11-10", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20241110T004901_20241110T004926_056485_06EC7A_F326"),
    ],
    "2024-07-05": [
        ("2024-07-08", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240708T004054_20240708T004119_054662_06A790_6EC4"),
        ("2024-07-01", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240701T004901_20240701T004926_054560_06A401_3449"),
        ("2024-07-13", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240713T004901_20240713T004926_054735_06AA12_1596"),
    ]
}


def extract_s1_features(df: pd.DataFrame) -> dict[str, dict]:
    """Extract VV, VH for each GEDI point using footprint spatial averaging."""
    init_earth_engine(load_config(PROJECT_ROOT / "config/config.yaml"))
    
    results = {}
    for gedi_date_str, candidate_list in CANDIDATES_BY_DATE.items():
        sub = df[df["gedi_date"] == gedi_date_str] if "gedi_date" in df.columns else df[df["acquisition_date"] == gedi_date_str]
        g_d = dt.datetime.strptime(gedi_date_str, "%Y-%m-%d").date()
        unassigned = set(sub["shot_number"].astype(str))
        
        logger.info("Extracting S1 for GEDI date %s (%d points) ...", gedi_date_str, len(sub))
        for s_date_str, scene_id in candidate_list:
            if not unassigned:
                break
            rem_sub = sub[sub["shot_number"].astype(str).isin(unassigned)]
            cand_diff = abs((dt.datetime.strptime(s_date_str, "%Y-%m-%d").date() - g_d).days)
            logger.info("  Querying %s (%s, diff: %dd) for %d points ...",
                        scene_id.split("/")[-1], s_date_str, cand_diff, len(rem_sub))
            
            feats = [
                ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(12.5),
                           {"shot_number": str(r.shot_number)})
                for r in rem_sub.itertuples()
            ]
            fc = ee.FeatureCollection(feats)
            img = ee.Image(scene_id).select(["VV", "VH"])
            reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
            info = reduced.getInfo()

            for f in info.get("features", []):
                props = f["properties"]
                sn = props["shot_number"]
                vv = props.get("VV")
                vh = props.get("VH")
                if vv is not None and vh is not None and np.isfinite(vv) and np.isfinite(vh):
                    # Linear power ratio: 10^((VH - VV)/10)
                    vh_vv_linear = 10.0 ** ((float(vh) - float(vv)) / 10.0)
                    results[sn] = {
                        "sentinel1_date": s_date_str,
                        "sentinel1_day_difference": cand_diff,
                        "VV": float(vv),
                        "VH": float(vh),
                        "VH_VV_ratio": float(vh_vv_linear),
                    }
                    unassigned.remove(sn)

    logger.info("S1 extraction complete: %d / %d points matched.", len(results), len(df))
    return results


def run():
    setup_logging()
    data_path = PROJECT_ROOT / "data/processed/training_dataset.parquet"
    df = pd.read_parquet(data_path)
    logger.info("Loaded training dataset: %d rows from %s", len(df), data_path)

    s1_dict = extract_s1_features(df)
    if len(s1_dict) != len(df):
        raise RuntimeError(f"Missing S1 data for {len(df) - len(s1_dict)} points!")

    # Merge into DataFrame
    s1_df = pd.DataFrame.from_dict(s1_dict, orient="index")
    s1_df.index.name = "shot_number"
    s1_df = s1_df.reset_index()
    # Ensure shot_number types match
    df["shot_number"] = df["shot_number"].astype(str)
    merged_df = pd.merge(df, s1_df, on="shot_number", how="inner")

    # Column ordering
    metadata_cols = [
        "shot_number", "latitude", "longitude", "agb", "gedi_quality",
        "gedi_sensitivity", "gedi_date", "sentinel2_date", "sentinel2_day_difference",
        "sentinel1_date", "sentinel1_day_difference"
    ]
    feature_cols = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "VV", "VH", "VH_VV_ratio"]
    ordered_cols = [c for c in metadata_cols if c in merged_df.columns] + [c for c in feature_cols if c in merged_df.columns]
    merged_df = merged_df[ordered_cols]

    # Save to primary and secondary paths
    merged_df.to_parquet(data_path, index=False)
    sub_path = PROJECT_ROOT / "data/processed/training/biomass_training_dataset.parquet"
    merged_df.to_parquet(sub_path, index=False)
    logger.info("Saved updated dataset with SAR features to %s and %s", data_path, sub_path)
    print("\nUpdated Dataset Columns:")
    for col in merged_df.columns:
        print(f"  - {col}")
    print("\nDataset Summary Stats for SAR features:")
    print(merged_df[["VV", "VH", "VH_VV_ratio"]].describe().to_string())


if __name__ == "__main__":
    run()
