"""Compute exact footprint count surviving all filters in the expanded region.

Filters:
1. GEDI L4A Quality Filters (l4_quality_flag==1, degrade_flag==0, sensitivity>=0.9, agbd<=600, solar_elevation<0)
2. ESA WorldCover v200: Tree Cover only (class 10)
3. Sentinel-2: Strict <= 30-day clear-sky match (Cloud Score+ cs >= 0.6)
"""
import datetime as dt
import logging
from pathlib import Path
import ee
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config, init_earth_engine, PROJECT_ROOT, setup_logging

def get_expanded_counts():
    setup_logging()
    logger = logging.getLogger("compute_expanded_counts")
    cfg = load_config(PROJECT_ROOT / "config/config.yaml")
    init_earth_engine(cfg)

    exp_bbox = cfg["study_region"]["bbox"] # [76.20, 11.20, 76.90, 11.90]
    ee_region = ee.Geometry.BBox(*exp_bbox)

    # 1. Load filtered GEDI collection without per-step size() calls
    f = cfg["datasets"]["gedi"]["filters"]
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

    # Apply quality filters
    gedi_filter = ee.Filter.And([
        ee.Filter.eq("l4_quality_flag", 1),
        ee.Filter.eq("degrade_flag", 0),
        ee.Filter.gte("sensitivity", 0.9),
        ee.Filter.lte("agbd", 600.0),
        ee.Filter.lt("solar_elevation", 0)
    ])
    filtered_gedi = col.filter(gedi_filter).select([
        "shot_number", "agbd", "sensitivity", "lat_lowestmode", "lon_lowestmode",
        "delta_time"
    ])

    total_gedi = filtered_gedi.size().getInfo()
    logger.info("Total GEDI footprints passing quality filters in expanded bbox: %d", total_gedi)

    # Download GEDI metadata/coordinates to pandas
    feats = filtered_gedi.toList(total_gedi)
    batch_size = 2000
    rows = []
    for lo in range(0, total_gedi, batch_size):
        hi = min(lo + batch_size, total_gedi)
        logger.info("Fetching GEDI points %d..%d...", lo, hi)
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
    df = df.rename(columns={"agbd": "gedi_agb"})
    logger.info("Fetched %d GEDI points across dates:\n%s", len(df), df["acquisition_date"].value_counts())

    # 2. Query ESA WorldCover for all points
    logger.info("Querying ESA WorldCover v200 for %d points...", len(df))
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    
    # Chunk WorldCover reduceRegions
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
    wc_breakdown = df["worldcover"].value_counts(dropna=False).to_dict()
    logger.info("ESA WorldCover breakdown on expanded GEDI points: %s", wc_breakdown)

    forest_df = df[df["worldcover"] == 10].copy()
    logger.info("Points surviving WorldCover Forest Filter (Class 10): %d / %d (%.2f%%)",
                len(forest_df), len(df), 100.0 * len(forest_df) / len(df))

    # 3. Check Sentinel-2 coverage strictly <= 30 days
    logger.info("Checking Sentinel-2 coverage within <= 30 days for %d forest points...", len(forest_df))
    from download_sentinel2 import _masked_collection
    s2_col = _masked_collection(cfg, ee_region)

    # For each unique GEDI date, find available S2 clear days within 30 days
    gedi_dates = sorted(forest_df["acquisition_date"].unique())
    logger.info("Forest GEDI dates: %s", gedi_dates)

    # Check S2 clear coverage per point
    # We test sampling B8/NDVI from the closest clear day within 30 days
    surviving_points = []
    
    for g_date_str in gedi_dates:
        g_sub = forest_df[forest_df["acquisition_date"] == g_date_str]
        g_d = dt.datetime.strptime(g_date_str, "%Y-%m-%d").date()
        logger.info("Evaluating S2 matches for GEDI date %s (%d forest points)...", g_date_str, len(g_sub))
        
        # Candidate S2 dates within 30 days
        start_w = (g_d - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        end_w = (g_d + dt.timedelta(days=30) + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        
        cand_col = s2_col.filterDate(start_w, end_w)
        cand_size = cand_col.size().getInfo()
        logger.info("  Found %d candidate S2 granules within +-30 days", cand_size)
        if cand_size == 0:
            continue
            
        # Get list of days sorted by proximity to g_d
        times = cand_col.aggregate_array("system:time_start").getInfo()
        day_diffs = []
        seen_days = set()
        for ms in times:
            d_str = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            if d_str not in seen_days:
                seen_days.add(d_str)
                diff = abs((dt.datetime.strptime(d_str, "%Y-%m-%d").date() - g_d).days)
                day_diffs.append((diff, d_str))
        day_diffs.sort()
        logger.info("  S2 dates sorted by day difference: %s", [(d, diff) for diff, d in day_diffs][:10])

        # Test valid pixel sampling across candidate days for these points
        # In GEE, we can mosaic the masked collection prioritizing closest dates
        # Or test reduceRegions on the composite
        unmatched_pts = set(g_sub["shot_number"].astype(str))
        matched_dict = {}

        for diff, d_str in day_diffs:
            if not unmatched_pts:
                break
            rem_sub = g_sub[g_sub["shot_number"].astype(str).isin(unmatched_pts)]
            day_img = cand_col.filterDate(d_str, (dt.datetime.strptime(d_str, "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")).mosaic()
            
            # Reduce over remaining points in chunks of 500
            for lo in range(0, len(rem_sub), 500):
                chunk_rem = rem_sub.iloc[lo:lo+500]
                fc_rem = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(12.5), {"shot_number": str(r.shot_number)})
                    for r in chunk_rem.itertuples()
                ])
                reduced = day_img.select(["B2", "B4", "B8", "NDVI"]).reduceRegions(
                    collection=fc_rem, reducer=ee.Reducer.mean(), scale=10
                ).getInfo()
                for feat in reduced.get("features", []):
                    props = feat["properties"]
                    sn = props["shot_number"]
                    if props.get("B8") is not None and np.isfinite(props["B8"]) and props.get("NDVI") is not None:
                        matched_dict[sn] = {
                            "s2_date": d_str,
                            "s2_day_diff": diff
                        }
                        unmatched_pts.discard(sn)

        logger.info("  Date %s: %d / %d forest points matched S2 within <= 30 days.",
                    g_date_str, len(matched_dict), len(g_sub))
        for sn, m in matched_dict.items():
            surviving_points.append(sn)

    print("\n" + "=" * 60)
    print("SURVIVAL REPORT FOR EXPANDED REGION")
    print("=" * 60)
    print(f"Total GEDI footprints (quality-filtered) : {len(df)}")
    print(f"ESA WorldCover breakdown before filter   : {wc_breakdown}")
    print(f"Surviving WorldCover Forest (Class 10)   : {len(forest_df)}")
    print(f"Surviving Forest + Strict <= 30-day S2   : {len(surviving_points)}")
    print(f"Comparison vs Original v1 (1,326)        : {len(surviving_points)} vs 1,326")
    print("=" * 60)
    return len(surviving_points)

if __name__ == "__main__":
    get_expanded_counts()
