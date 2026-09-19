"""Sentinel-1 SAR GRD retrieval module.

Dataset (verified against the Earth Engine Data Catalog):
    COPERNICUS/S1_GRD — C-band Ground Range Detected, 10 m.
    Earth Engine has already applied, via the Sentinel-1 Toolbox:
      orbit-file update, border/thermal-noise removal, radiometric calibration
      and terrain correction, and converted to backscatter in DECIBELS (dB).
    We therefore do not re-preprocess; we only filter and export.

Filters applied server-side: region, date, instrumentMode ('IW'),
transmitterReceiverPolarisation (e.g. [VV, VH]) and optionally
orbitProperties_pass ('ASCENDING'/'DESCENDING'). Each exported scene keeps its
metadata (orbit pass, relative orbit, mode, resolution) so incompatible
configurations are never silently mixed — they are recorded per file.
"""
from __future__ import annotations

import json
import logging
import datetime as dt
from pathlib import Path
from typing import Any

from common import (PipelineError, EmptyQueryError, date_range, dataset_enabled,
                    load_region_geometry, ee_geometry, write_manifest, PROJECT_ROOT)

logger = logging.getLogger("biomass_pipeline.sentinel1")

META_PROPS = ["instrumentMode", "orbitProperties_pass", "relativeOrbitNumber_start",
              "transmitterReceiverPolarisation", "resolution_meters",
              "system:time_start"]


def _collection(cfg: dict[str, Any], ee_region):
    import ee
    s1 = cfg["datasets"]["sentinel1"]
    start, end = date_range(cfg)

    col = ee.ImageCollection(s1["product"]) \
        .filterBounds(ee_region) \
        .filterDate(start, end) \
        .filter(ee.Filter.eq("instrumentMode", s1["instrument_mode"])) \
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VH"))
    if s1.get("orbit_pass"):
        col = col.filter(ee.Filter.eq("orbitProperties_pass", s1["orbit_pass"]))
    return col.select(list(s1["polarization"]))


def run(cfg: dict[str, Any]) -> Path | None:
    if not dataset_enabled(cfg, "sentinel1"):
        logger.info("Sentinel-1 disabled in config — skipping.")
        return None

    import ee
    import geemap
    from common import init_earth_engine
    init_earth_engine(cfg)

    region_geom, region_desc = load_region_geometry(cfg)
    ee_region = ee_geometry(region_geom)
    start, end = date_range(cfg)
    s1_cfg = cfg["datasets"]["sentinel1"]

    raw_root = PROJECT_ROOT / "data/raw/sentinel1"
    meta_dir = PROJECT_ROOT / "metadata"
    raw_root.mkdir(parents=True, exist_ok=True)

    col = _collection(cfg, ee_region)
    n = int(col.size().getInfo())
    if n == 0:
        raise EmptyQueryError(
            f"No Sentinel-1 IW VV+VH scenes for '{cfg['study_region']['name']}' "
            f"in {start}..{end}. GEDI/S-1 coverage gap — widen the date range.")
    logger.info("Sentinel-1: %d scenes found.", n)

    # Fetch per-scene metadata in ONE server call (no per-scene chatter).
    feats = col.toList(col.size())
    scene_infos: list[dict[str, Any]] = []
    lo = 0
    while lo < n:
        hi = min(lo + 500, n)
        for f in ee.FeatureCollection(feats.slice(lo, hi)).getInfo().get("features", []):
            props = f.get("properties", {})
            # getInfo() may return the bare index or the full asset path as id.
            raw_id = f.get("id", "") or props.get("system:index", "")
            props["system:index"] = raw_id.split("/")[-1] if "/" in raw_id else raw_id
            scene_infos.append(props)
        lo = hi

    # If GEDI samples exist, filter S1 scenes to only those within max_days_difference of a GEDI date
    gedi_path = PROJECT_ROOT / "data/raw/gedi/gedi_samples.parquet"
    if gedi_path.exists():
        import pandas as pd
        gedi_df = pd.read_parquet(gedi_path)
        if "acquisition_date" in gedi_df.columns:
            g_dates = [dt.datetime.strptime(str(d), "%Y-%m-%d").date() for d in gedi_df["acquisition_date"].dropna().unique()]
            max_days = int(cfg.get("matching", {}).get("max_days_difference", 30))
            filtered_scenes = []
            for props in scene_infos:
                ms = props.get("system:time_start")
                if ms:
                    d_date = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()
                    if any(abs((d_date - g).days) <= max_days for g in g_dates):
                        filtered_scenes.append(props)
            if filtered_scenes:
                logger.info("Filtered Sentinel-1 scenes from %d to %d based on GEDI dates.", len(scene_infos), len(filtered_scenes))
                scene_infos = filtered_scenes

    written_files: list[str] = []
    for props in scene_infos:
        ms = props.get("system:time_start")
        day = dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
        pass_dir = str(props.get("orbitProperties_pass", "UNKNOWN")).lower()
        idx = props.get("system:index", "unknown").replace("/", "_")
        out_dir = raw_root / day[:4] / day / pass_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tif = out_dir / f"{idx}.tif"
        if tif.exists():
            logger.info("  reuse %s", tif.name)
        else:
            img = ee.Image(f"{s1_cfg['product']}/{idx}").clip(ee_region)
            try:
                geemap.ee_export_image(
                    img, filename=str(tif), scale=float(s1_cfg["export_scale"]),
                    region=ee_region, file_per_band=False)
            except Exception as exc:
                raise PipelineError(f"Sentinel-1 export failed for {idx}: {exc}") from exc
            if not tif.exists() or tif.stat().st_size == 0:
                tif.unlink(missing_ok=True)
                raise PipelineError(f"Corrupted/empty export for scene {idx}.")
        written_files.append(str(tif.relative_to(PROJECT_ROOT)))
        logger.info("  wrote %s", tif.name)

    meta = {
        "product": s1_cfg["product"],
        "units": "dB (sigma-nought backscatter)",
        "preprocessing_by_earth_engine": [
            "apply orbit file", "remove border noise", "remove thermal noise",
            "radiometric calibration", "terrain correction (orthorectification)"],
        "retrieval_date_utc": dt.datetime.utcnow().isoformat() + "Z",
        "study_region": {"name": cfg["study_region"]["name"], "geometry": region_desc},
        "date_range": {"start": start, "end": end},
        "filters": {"instrument_mode": s1_cfg["instrument_mode"],
                    "polarization": s1_cfg["polarization"],
                    "orbit_pass": s1_cfg.get("orbit_pass")},
        "scene_count": n,
        "scenes": [{
            "system_index": p.get("system:index"),
            "acquisition_date": dt.datetime.utcfromtimestamp(
                p["system:time_start"] / 1000).strftime("%Y-%m-%d") if p.get("system:time_start") else None,
            "instrument_mode": p.get("instrumentMode"),
            "orbit_direction": p.get("orbitProperties_pass"),
            "relative_orbit": p.get("relativeOrbitNumber_start"),
            "polarizations": p.get("transmitterReceiverPolarisation"),
        } for p in scene_infos],
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sentinel1_metadata.json").write_text(json.dumps(meta, indent=2))

    write_manifest(meta_dir, "sentinel1", {
        "dataset": "sentinel1", "product": s1_cfg["product"], "version": "GRD dB",
        "study_region": cfg["study_region"]["name"],
        "date_range": {"start": start, "end": end},
        "source": "Google Earth Engine Data Catalog (COPERNICUS)",
        "filters": meta["filters"],
        "files": written_files,
        "record_count": n,
    })
    return raw_root


if __name__ == "__main__":
    from common import load_config, setup_logging
    setup_logging()
    run(load_config(PROJECT_ROOT / "config/config.yaml"))
