"""GEDI L4A footprint retrieval module.

Dataset (verified against the Earth Engine Data Catalog):
    LARSE/GEDI/GEDI04_A_002  — GEDI Level 4A footprint-level aboveground
    biomass density (AGBD), Version 2.1, from 2019-04-18 onward.
    FeatureCollection of ~25 m footprints; key fields include agbd, agbd_se,
    l4_quality_flag, degrade_flag, sensitivity, beam, lat_lowestmode,
    lon_lowestmode, shot_number, solar_elevation, selected_algorithm.

Quality filtering rationale (per the GEDI L4A ATBD / user guide):
    - l4_quality_flag == 1   : most useful biomass predictions
    - degrade_flag   == 0    : no degraded pointing/positioning information
    - sensitivity    >= 0.9  : beam can penetrate the canopy (0.9–0.95 typical)
    - solar_elevation < 0    : nighttime shots have higher waveform fidelity

Nothing is silently discarded: every filter reports a removal count, and the
per-filter numbers are stored in metadata/gedi_metadata.json.
"""
from __future__ import annotations

import json
import logging
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from common import (PipelineError, EmptyQueryError, date_range, dataset_enabled,
                    load_region_geometry, ee_geometry, write_manifest, PROJECT_ROOT)

logger = logging.getLogger("biomass_pipeline.gedi")

# Columns we keep for ML + provenance (names verified on the catalog page).
KEEP_COLUMNS = [
    "shot_number", "agbd", "agbd_se", "l4_quality_flag", "l2_quality_flag",
    "degrade_flag", "sensitivity", "beam", "channel", "selected_algorithm",
    "solar_elevation", "elev_lowestmode", "lat_lowestmode", "lon_lowestmode",
    "predictor_limit_flag", "response_limit_flag", "surface_flag",
    "delta_time",  # seconds since 2018-01-01 -> acquisition timestamp
]

# Earth Engine caps a single getInfo()/table download; chunk the export.
EXPORT_BATCH = 3000


def load_gedi_collection(cfg: dict[str, Any], ee_region):
    """Load GEDI FeatureCollection, supporting both direct FCs and IndexedFolders (LARSE/GEDI/GEDI04_A_002)."""
    import ee
    product = cfg["datasets"]["gedi"]["product"]
    start_s, end_s = date_range(cfg)
    try:
        col = ee.FeatureCollection(product).filterBounds(ee_region).filterDate(start_s, end_s)
        # Test if size() works directly
        _ = col.size().getInfo()
        return col
    except Exception:
        pass

    idx_id = product if product.endswith("_INDEX") else f"{product}_INDEX"
    idx = ee.FeatureCollection(idx_id).filterBounds(ee_region) \
        .filter(ee.Filter.gte('time_start', start_s)) \
        .filter(ee.Filter.lte('time_start', end_s))
    tids = idx.aggregate_array('table_id').getInfo()
    if not tids:
        return ee.FeatureCollection([])
    cols = [ee.FeatureCollection(t).filterBounds(ee_region) for t in tids]
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c)
    return merged


def _filtered_collection(cfg: dict[str, Any], ee_region) -> tuple[Any, dict[str, int]]:
    """Build the filtered EE FeatureCollection and count removals per filter.

    Returns (collection, counts) where counts has raw/removed/retained numbers.
    Server-side filters are applied BEFORE any download — no global pulls.
    """
    import ee

    f = cfg["datasets"]["gedi"]["filters"]
    col = load_gedi_collection(cfg, ee_region).select(KEEP_COLUMNS)

    counts: dict[str, Any] = {}
    counts["raw_in_region_time"] = int(col.size().getInfo())

    # Apply filters one at a time and record how many each step removes, so
    # the QC trail shows not just HOW MANY were dropped but WHY.
    steps: list[tuple[str, Any]] = []
    if f.get("l4_quality_flag", True):
        steps.append(("l4_quality_flag==1", ee.Filter.eq("l4_quality_flag", 1)))
    if f.get("degrade_flag") is not None:
        steps.append((f"degrade_flag=={int(f['degrade_flag'])}",
                      ee.Filter.eq("degrade_flag", int(f["degrade_flag"]))))
    if f.get("min_sensitivity") is not None:
        steps.append((f"sensitivity>={float(f['min_sensitivity'])}",
                      ee.Filter.gte("sensitivity", float(f["min_sensitivity"]))))
    if f.get("max_agbd") is not None:
        steps.append((f"agbd<={float(f['max_agbd'])}",
                      ee.Filter.lte("agbd", float(f["max_agbd"]))))
    if f.get("solar_elevation_lt0", True):
        steps.append(("solar_elevation<0", ee.Filter.lt("solar_elevation", 0)))

    per_filter: dict[str, int] = {}
    prev = counts["raw_in_region_time"]
    for name, filt in steps:
        col = col.filter(filt)
        after = int(col.size().getInfo())
        per_filter[name] = prev - after   # removed by THIS filter
        prev = after

    counts["retained"] = prev
    counts["removed_total"] = counts["raw_in_region_time"] - prev
    counts["removed_per_filter"] = per_filter
    logger.info("GEDI: %d raw, %d retained, %d removed. Per-filter: %s",
                counts["raw_in_region_time"], counts["retained"],
                counts["removed_total"], per_filter)
    return col, counts


def _download_all(col) -> pd.DataFrame:
    """Download a (possibly large) FeatureCollection in batches to a DataFrame."""
    import ee  # noqa: F401  (needed by ee.List slicing semantics)

    n = int(col.size().getInfo())
    if n == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    feats = col.toList(col.size())
    for lo in range(0, n, EXPORT_BATCH):
        hi = min(lo + EXPORT_BATCH, n)
        chunk = ee.FeatureCollection(feats.slice(lo, hi)).getInfo()
        for feat in chunk.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Point" and "coordinates" in geom:
                props.setdefault("lon_lowestmode", geom["coordinates"][0])
                props.setdefault("lat_lowestmode", geom["coordinates"][1])
            rows.append(props)
        logger.info("GEDI download progress: %d/%d", hi, n)
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any]) -> Path | None:
    if not dataset_enabled(cfg, "gedi"):
        logger.info("GEDI dataset disabled in config — skipping.")
        return None

    from common import init_earth_engine
    init_earth_engine(cfg)

    region_geom, region_desc = load_region_geometry(cfg)
    ee_region = ee_geometry(region_geom)
    start, end = date_range(cfg)

    raw_dir = PROJECT_ROOT / "data/raw/gedi"
    meta_dir = PROJECT_ROOT / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "gedi_samples.parquet"

    if out_path.exists():
        logger.info("Reusing existing %s (idempotent run). Delete it to re-download.", out_path)
        df = pd.read_parquet(out_path)
        return out_path

    col, counts = _filtered_collection(cfg, ee_region)
    if counts["retained"] == 0:
        raise EmptyQueryError(
            f"No GEDI footprints pass filters in '{cfg['study_region']['name']}' "
            f"for {start}..{end} (raw in region: {counts['raw_in_region_time']}). "
            "Widen the date range or relax filters.")

    df = _download_all(col)
    if df.empty:
        raise PipelineError("GEDI export returned no rows despite a non-empty collection.")

    # --- tidy + acquisition date (delta_time = seconds since 2018-01-01) ---
    if "acquisition_time" not in df.columns:
        df["acquisition_time"] = pd.NaT
    if "delta_time" in df.columns:
        epoch = pd.Timestamp("2018-01-01", tz="UTC")
        df["acquisition_time"] = epoch + pd.to_timedelta(df["delta_time"], unit="s")
    # Earth Engine stamps each feature with system:time_start; prefer it when present.
    if "system:time_start" in df.columns:
        df["acquisition_time"] = pd.to_datetime(df["system:time_start"], unit="ms", utc=True)
    df["acquisition_date"] = pd.to_datetime(df["acquisition_time"]).dt.date.astype("string")
    df = df.rename(columns={"lat_lowestmode": "latitude", "lon_lowestmode": "longitude",
                            "l4_quality_flag": "gedi_quality", "agbd": "gedi_agb"})

    # --- GeoParquet output ---
    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326")
        gdf.to_parquet(out_path, index=False)
    except Exception as exc:
        logger.warning("GeoParquet failed (%s); writing plain Parquet.", exc)
        df.to_parquet(out_path, index=False)
    logger.info("GEDI samples saved -> %s (%d rows)", out_path, len(df))

    # --- per-dataset metadata ---
    meta = {
        "product": cfg["datasets"]["gedi"]["product"],
        "product_name": "GEDI Level 4A Aboveground Biomass Density (footprints)",
        "product_version": "2.1",
        "retrieval_date_utc": dt.datetime.utcnow().isoformat() + "Z",
        "study_region": {"name": cfg["study_region"]["name"], "geometry": region_desc},
        "date_range": {"start": start, "end": end},
        "raw_observations_in_region": counts["raw_in_region_time"],
        "retained_after_filtering": counts["retained"],
        "removed_by_filters": counts["removed_total"],
        "filtering_criteria": cfg["datasets"]["gedi"]["filters"],
        "removed_per_filter": counts.get("removed_per_filter", {}),
        "variables_retrieved": KEEP_COLUMNS,
        "output_file": str(out_path.relative_to(PROJECT_ROOT)),
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "gedi_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    write_manifest(meta_dir, "gedi", {
        "dataset": "gedi", "product": meta["product"], "version": "2.1",
        "study_region": cfg["study_region"]["name"],
        "date_range": {"start": start, "end": end},
        "source": "Google Earth Engine Data Catalog (LARSE/GEDI)",
        "filters": cfg["datasets"]["gedi"]["filters"],
        "files": [str(out_path.relative_to(PROJECT_ROOT))],
        "record_count": int(len(df)),
    })
    return out_path


if __name__ == "__main__":
    from common import load_config, setup_logging
    setup_logging()
    run(load_config(PROJECT_ROOT / "config/config.yaml"))
