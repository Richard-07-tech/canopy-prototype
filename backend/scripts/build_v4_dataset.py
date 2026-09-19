"""Extract all features for the final v4 dataset (Expanded + Forest-filtered + Topo + Strict S2/S1).

Filters applied:
1. Region: Expanded Nilgiris/Western Ghats [76.20, 11.20, 76.90, 11.90]
2. GEDI L4A Quality: l4_quality_flag==1, degrade_flag==0, sensitivity>=0.9, agbd<=600, solar_elevation<0
3. Forest Mask: ESA WorldCover v200 Class 10 (Tree Cover)
4. Temporal Match: Strict <= 30-day Sentinel-2 (cloud-screened) and Sentinel-1
5. Topography: NASADEM elevation + slope (footprint-averaged)
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
import ee
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config, init_earth_engine, PROJECT_ROOT, setup_logging
from download_sentinel2 import _masked_collection

logger = logging.getLogger("biomass_pipeline.build_v4")

def run():
    setup_logging()
    cfg = load_config(PROJECT_ROOT / "config/config.yaml")
    init_earth_engine(cfg)

    exp_bbox = cfg["study_region"]["bbox"]  # [76.20, 11.20, 76.90, 11.90]
    ee_region = ee.Geometry.BBox(*exp_bbox)

    # 1. Load GEDI footprints
    logger.info("Loading GEDI footprints from expanded region...")
    product = cfg["datasets"]["gedi"]["product"]
    idx_id = f"{product}_INDEX"
    idx = ee.FeatureCollection(idx_id).filterBounds(ee_region) \
        .filter(ee.Filter.gte('time_start', '2024-01-01')) \
        .filter(ee.Filter.lte('time_start', '2024-12-31'))
    tids = idx.aggregate_array('table_id').getInfo()
    cols = [ee.FeatureCollection(t).filterBounds(ee_region) for t in tids]
    col = cols[0]
    for c in cols[1:]:
        col = col.merge(c)

    gedi_filter = ee.Filter.And([
        ee.Filter.eq("l4_quality_flag", 1),
        ee.Filter.eq("degrade_flag", 0),
        ee.Filter.gte("sensitivity", 0.9),
        ee.Filter.lte("agbd", 600.0),
        ee.Filter.lt("solar_elevation", 0)
    ])
    filtered_gedi = col.filter(gedi_filter).select([
        "shot_number", "agbd", "sensitivity", "l4_quality_flag", "lat_lowestmode", "lon_lowestmode",
        "delta_time"
    ])

    total_gedi = filtered_gedi.size().getInfo()
    logger.info("Total GEDI footprints in expanded bbox passing quality filters: %d", total_gedi)

    feats = filtered_gedi.toList(total_gedi)
    rows = []
    batch_size = 2000
    for lo in range(0, total_gedi, batch_size):
        hi = min(lo + batch_size, total_gedi)
        sub_chunk = ee.FeatureCollection(feats.slice(lo, hi)).getInfo()
        for ft in sub_chunk.get("features", []):
            props = ft.get("properties", {})
            geom = ft.get("geometry", {})
            if geom and "coordinates" in geom:
                props["longitude"] = geom["coordinates"][0]
                props["latitude"] = geom["coordinates"][1]
            rows.append(props)

    df = pd.DataFrame(rows)
    epoch = pd.Timestamp("2018-01-01", tz="UTC")
    df["acquisition_time"] = epoch + pd.to_timedelta(df["delta_time"], unit="s")
    df["acquisition_date"] = pd.to_datetime(df["acquisition_time"]).dt.date.astype("string")
    df = df.rename(columns={"agbd": "agb", "l4_quality_flag": "gedi_quality", "sensitivity": "gedi_sensitivity",
                            "acquisition_date": "gedi_date"})

    # 2. ESA WorldCover Forest Mask (Class 10)
    logger.info("Filtering with ESA WorldCover v200...")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc_results = {}
    for lo in range(0, len(df), 1500):
        hi = min(lo + 1500, len(df))
        sub_df = df.iloc[lo:hi]
        fc_sub = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]), {"shot_number": str(r.shot_number)})
            for r in sub_df.itertuples()
        ])
        reduced_wc = wc.reduceRegions(collection=fc_sub, reducer=ee.Reducer.first(), scale=10).getInfo()
        for ftr in reduced_wc.get("features", []):
            sn = ftr["properties"]["shot_number"]
            wc_results[sn] = ftr["properties"].get("first")

    df["worldcover"] = df["shot_number"].astype(str).map(wc_results)
    forest_df = df[df["worldcover"] == 10].copy()
    logger.info("Forest footprints retained: %d / %d", len(forest_df), len(df))

    # 3. Topography (NASADEM elevation + slope)
    logger.info("Extracting NASADEM elevation and slope with 12.5m footprint buffer...")
    dem = ee.Image(cfg["datasets"]["auxiliary"]["dem"]).select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    topo = dem.addBands(slope)

    topo_results = {}
    for lo in range(0, len(forest_df), 1500):
        hi = min(lo + 1500, len(forest_df))
        sub_df = forest_df.iloc[lo:hi]
        fc_sub = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(12.5), {"shot_number": str(r.shot_number)})
            for r in sub_df.itertuples()
        ])
        reduced_topo = topo.reduceRegions(collection=fc_sub, reducer=ee.Reducer.mean(), scale=30).getInfo()
        for ftr in reduced_topo.get("features", []):
            sn = ftr["properties"]["shot_number"]
            topo_results[sn] = {
                "elevation": ftr["properties"].get("elevation"),
                "slope": ftr["properties"].get("slope")
            }

    forest_df["elevation"] = [topo_results[str(sn)]["elevation"] for sn in forest_df["shot_number"]]
    forest_df["slope"] = [topo_results[str(sn)]["slope"] for sn in forest_df["shot_number"]]

    # 4. Strict <= 30-day Sentinel-2 Matching
    logger.info("Matching Sentinel-2 within strict <= 30 days...")
    s2_col = _masked_collection(cfg, ee_region)
    s2_bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]

    matched_rows = []
    for g_date_str in sorted(forest_df["gedi_date"].unique()):
        g_sub = forest_df[forest_df["gedi_date"] == g_date_str]
        g_d = dt.datetime.strptime(g_date_str, "%Y-%m-%d").date()
        start_w = (g_d - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        end_w = (g_d + dt.timedelta(days=30) + dt.timedelta(days=1)).strftime("%Y-%m-%d")

        cand_col = s2_col.filterDate(start_w, end_w)
        times = cand_col.aggregate_array("system:time_start").getInfo() or []
        day_diffs = []
        seen_days = set()
        for ms in times:
            d_str = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            if d_str not in seen_days:
                seen_days.add(d_str)
                diff = abs((dt.datetime.strptime(d_str, "%Y-%m-%d").date() - g_d).days)
                day_diffs.append((diff, d_str))
        day_diffs.sort()

        unmatched = set(g_sub["shot_number"].astype(str))
        g_dict = {str(r.shot_number): r for r in g_sub.itertuples()}

        for diff, d_str in day_diffs:
            if not unmatched:
                break
            rem_pts = [g_dict[sn] for sn in unmatched]
            day_end = (dt.datetime.strptime(d_str, "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")
            day_img = cand_col.filterDate(d_str, day_end).mosaic().select(s2_bands)

            for lo in range(0, len(rem_pts), 500):
                chunk = rem_pts[lo:lo+500]
                fc_chunk = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(12.5), {"shot_number": str(r.shot_number)})
                    for r in chunk
                ])
                reduced = day_img.reduceRegions(collection=fc_chunk, reducer=ee.Reducer.mean(), scale=10).getInfo()
                for feat in reduced.get("features", []):
                    props = feat["properties"]
                    sn = props["shot_number"]
                    if all(props.get(b) is not None and np.isfinite(props[b]) for b in s2_bands):
                        base_r = g_dict[sn]
                        row_entry = {
                            "shot_number": sn,
                            "latitude": base_r.latitude,
                            "longitude": base_r.longitude,
                            "agb": base_r.agb,
                            "gedi_quality": base_r.gedi_quality,
                            "gedi_sensitivity": base_r.gedi_sensitivity,
                            "gedi_date": g_date_str,
                            "sentinel2_date": d_str,
                            "sentinel2_day_difference": diff,
                            "elevation": base_r.elevation,
                            "slope": base_r.slope,
                        }
                        for b in s2_bands:
                            row_entry[b] = float(props[b])
                        matched_rows.append(row_entry)
                        unmatched.discard(sn)

    v4_df = pd.DataFrame(matched_rows)
    logger.info("Sentinel-2 matched: %d points surviving all filters.", len(v4_df))

    # 5. Extract Sentinel-1 SAR features (VV, VH, VH/VV ratio) for these surviving points
    logger.info("Extracting Sentinel-1 SAR features for %d points...", len(v4_df))
    from extract_s1_features import CANDIDATES_BY_DATE
    s1_results = {}

    for gedi_date_str in sorted(v4_df["gedi_date"].unique()):
        sub = v4_df[v4_df["gedi_date"] == gedi_date_str]
        g_d = dt.datetime.strptime(gedi_date_str, "%Y-%m-%d").date()
        unassigned = set(sub["shot_number"].astype(str))

        # Check candidate scenes
        cand_list = CANDIDATES_BY_DATE.get(gedi_date_str)
        if not cand_list:
            # Query S1 collection dynamically within 30 days
            s1_col = ee.ImageCollection(cfg["datasets"]["sentinel1"]["product"]) \
                .filterBounds(ee_region) \
                .filterDate((g_d - dt.timedelta(days=30)).strftime("%Y-%m-%d"),
                            (g_d + dt.timedelta(days=31)).strftime("%Y-%m-%d")) \
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            s1_ids = s1_col.aggregate_array("system:id").getInfo() or []
            cand_list = []
            for sid in s1_ids:
                # parse date from ID
                parts = sid.split("_")
                for p in parts:
                    if len(p) == 15 and p.startswith("2024"):
                        s_d_str = f"{p[0:4]}-{p[4:6]}-{p[6:8]}"
                        cand_list.append((s_d_str, sid))
                        break
            cand_list.sort(key=lambda x: abs((dt.datetime.strptime(x[0], "%Y-%m-%d").date() - g_d).days))

        for s_date_str, scene_id in cand_list:
            if not unassigned:
                break
            rem_sub = sub[sub["shot_number"].astype(str).isin(unassigned)]
            cand_diff = abs((dt.datetime.strptime(s_date_str, "%Y-%m-%d").date() - g_d).days)
            if cand_diff > 30:
                continue

            feats = [
                ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(12.5),
                           {"shot_number": str(r.shot_number)})
                for r in rem_sub.itertuples()
            ]
            fc = ee.FeatureCollection(feats)
            img = ee.Image(scene_id).select(["VV", "VH"])
            reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10).getInfo()

            for f in reduced.get("features", []):
                props = f["properties"]
                sn = props["shot_number"]
                vv = props.get("VV")
                vh = props.get("VH")
                if vv is not None and vh is not None and np.isfinite(vv) and np.isfinite(vh):
                    vh_vv_linear = 10.0 ** ((float(vh) - float(vv)) / 10.0)
                    s1_results[sn] = {
                        "sentinel1_date": s_date_str,
                        "sentinel1_day_difference": cand_diff,
                        "VV": float(vv),
                        "VH": float(vh),
                        "VH_VV_ratio": float(vh_vv_linear),
                    }
                    unassigned.discard(sn)

    logger.info("S1 matched for %d / %d points.", len(s1_results), len(v4_df))

    # Merge S1 features
    s1_df = pd.DataFrame.from_dict(s1_results, orient="index").reset_index().rename(columns={"index": "shot_number"})
    v4_df = pd.merge(v4_df, s1_df, on="shot_number", how="inner")
    logger.info("Final v4 dataset ready with %d rows and %d columns.", len(v4_df), len(v4_df.columns))

    # Reorder columns
    cols_order = [
        "shot_number", "latitude", "longitude", "agb", "gedi_quality", "gedi_sensitivity",
        "gedi_date", "sentinel2_date", "sentinel2_day_difference", "sentinel1_date", "sentinel1_day_difference",
        "B2", "B3", "B4", "B8", "B11", "B12", "NDVI",
        "VV", "VH", "VH_VV_ratio",
        "elevation", "slope"
    ]
    v4_df = v4_df[[c for c in cols_order if c in v4_df.columns]]

    # Save to data/processed/training_dataset_v4.parquet and update training_dataset.parquet
    out_v4 = PROJECT_ROOT / "data/processed/training_dataset_v4.parquet"
    v4_df.to_parquet(out_v4, index=False)
    logger.info("Saved v4 dataset to %s", out_v4)

    return v4_df

if __name__ == "__main__":
    run()
