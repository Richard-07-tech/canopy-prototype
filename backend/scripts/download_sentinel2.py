"""Sentinel-2 surface-reflectance retrieval module.

Datasets (verified against the Earth Engine Data Catalog):
  * COPERNICUS/S2_SR_HARMONIZED           — harmonized L2A surface reflectance.
      Bands B2 (blue), B3 (green), B4 (red), B8 (NIR), B11/B12 (SWIR1/2).
      Integer reflectance with scale factor 10000.
      Granule-level cloud property: CLOUDY_PIXEL_PERCENTAGE.
  * GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED — per-pixel clear-sky QA.
      Band 'cs' in [0,1]; Google's reference implementation masks pixels with
      cs < 0.6 (cs handles BOTH clouds and cloud shadows).

De-duplication: several Sentinel-2 granules can share one calendar day in a
region (neighbouring MGRS tiles / swath overlap). We group images by
YYYY-MM-DD and export ONE cloud-masked mosaic per day, so downstream matching
never sees duplicate/overlapping exports for the same acquisition date.

Output layout:
  data/raw/sentinel2/YYYY/YYYY-MM-DD/B02.tif ... NDVI.tif
Each GeoTIFF carries acquisition metadata in its tags and the per-day record
is appended to metadata/sentinel2_metadata.json.
"""
from __future__ import annotations

import json
import logging
import datetime as dt
from pathlib import Path
from typing import Any

from common import (PipelineError, EmptyQueryError, date_range, dataset_enabled,
                    load_region_geometry, ee_geometry, write_manifest, PROJECT_ROOT)

logger = logging.getLogger("biomass_pipeline.sentinel2")

BAND_OUT_NAMES = {"B2": "B02", "B3": "B03", "B4": "B04",
                  "B8": "B08", "B11": "B11", "B12": "B12"}
SCALE_FACTOR = 10000.0  # verified on the catalog page


def _masked_collection(cfg: dict[str, Any], ee_region):
    """Server-side: region + date + granule cloud filter + pixel cloud mask."""
    import ee

    s2_cfg = cfg["datasets"]["sentinel2"]
    start, end = date_range(cfg)

    s2 = ee.ImageCollection(s2_cfg["product"]) \
        .filterBounds(ee_region) \
        .filterDate(start, end) \
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(s2_cfg["cloud_percentage"])))

    cs_plus = ee.ImageCollection(s2_cfg["cloud_product"]).filterBounds(ee_region)

    # Join Cloud Score+ to each S2 image by shared system:index.
    linked = ee.Join.saveFirst("cs").apply(
        primary=s2, secondary=cs_plus,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))

    thr = float(s2_cfg["cs_threshold"])
    bands = list(s2_cfg["bands"])

    def mask_and_scale(img):
        img = ee.Image(img)
        cs = ee.Image(img.get("cs")).select("cs")
        clear = cs.gte(thr)
        refl = img.select(bands).multiply(SCALE_FACTOR)  # -> reflectance in DN units
        refl = refl.updateMask(clear)
        ndvi = refl.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return refl.addBands(ndvi).copyProperties(img, [
            "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE",
            "SENSING_ORBIT_DIRECTION", "PROCESSING_BASELINE"])

    return ee.ImageCollection(linked).map(mask_and_scale)


def _daily_groups(cfg: dict[str, Any], col) -> list[tuple[str, str]]:
    """Return sorted list of (date_str, system:time_start) per calendar day."""
    import ee
    info = col.aggregate_array("system:time_start").getInfo() or []
    seen: dict[str, int] = {}
    for ms in sorted(info):
        day = dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
        seen.setdefault(day, ms)  # keep earliest granule time of the day
    return sorted(seen.items())


def export_day_to_drive(cfg: dict[str, Any], col, day: str, ee_region,
                        folder: str = "forest_agb_pipeline_exports") -> list[dict[str, Any]]:
    """Submit batch export tasks to Google Drive via ee.batch.Export.image.toDrive()."""
    import ee

    s2_cfg = cfg["datasets"]["sentinel2"]
    day_start = ee.Date(day)
    day_end = day_start.advance(1, "day")
    mosaic = col.filterDate(day_start, day_end).mosaic().clip(ee_region)
    region_name = cfg.get("study_region", {}).get("name", "region")

    submitted = []
    for band in list(s2_cfg["bands"]) + ["NDVI"]:
        out_name = BAND_OUT_NAMES.get(band, band)
        img = mosaic.select(band)
        desc = f"S2_{region_name}_{day}_{out_name}"
        filename = f"S2_{region_name}_{day}_{out_name}"
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder=folder,
            fileNamePrefix=filename,
            scale=float(s2_cfg["export_scale"]),
            region=ee_region,
            maxPixels=1e9
        )
        task.start()
        submitted.append({
            "task_id": task.id,
            "description": desc,
            "day": day,
            "band": out_name,
            "filename": f"{filename}.tif",
            "task_object": task
        })
        logger.info("Submitted EE Drive Export Task: %s (ID: %s)", desc, task.id)
    return submitted


def _export_day(cfg: dict[str, Any], col, day: str, day_ms: int,
                ee_region, out_dir: Path) -> list[str]:
    """Mosaic one day's granules (cloud-masked) and export one GeoTIFF per band."""
    import ee
    import geemap

    s2_cfg = cfg["datasets"]["sentinel2"]
    day_start = ee.Date(day)
    day_end = day_start.advance(1, "day")
    mosaic = col.filterDate(day_start, day_end).mosaic().clip(ee_region)

    written: list[str] = []
    for band in list(s2_cfg["bands"]) + ["NDVI"]:
        out_name = BAND_OUT_NAMES.get(band, band)
        tif = out_dir / f"{out_name}.tif"
        if tif.exists():
            logger.info("  reuse %s", tif.name)
            written.append(tif.name)
            continue
        img = mosaic.select(band)
        try:
            geemap.ee_export_image(
                img, filename=str(tif), scale=float(s2_cfg["export_scale"]),
                region=ee_region, file_per_band=False)
        except Exception as exc:
            raise PipelineError(f"Export failed for {day} band {band}: {exc}") from exc
        if not tif.exists() or tif.stat().st_size == 0:
            tif.unlink(missing_ok=True)
            raise PipelineError(f"Corrupted/empty export for {day} band {band}.")
        written.append(tif.name)
        logger.info("  wrote %s", tif.name)
    return written


def run(cfg: dict[str, Any]) -> Path | None:
    if not dataset_enabled(cfg, "sentinel2"):
        logger.info("Sentinel-2 disabled in config — skipping.")
        return None

    from common import init_earth_engine
    init_earth_engine(cfg)

    region_geom, region_desc = load_region_geometry(cfg)
    ee_region = ee_geometry(region_geom)
    start, end = date_range(cfg)

    raw_root = PROJECT_ROOT / "data/raw/sentinel2"
    meta_dir = PROJECT_ROOT / "metadata"
    raw_root.mkdir(parents=True, exist_ok=True)

    col = _masked_collection(cfg, ee_region)
    n = int(col.size().getInfo())
    if n == 0:
        raise EmptyQueryError(
            f"No Sentinel-2 granules pass the cloud filter "
            f"(<={cfg['datasets']['sentinel2']['cloud_percentage']}%) for "
            f"'{cfg['study_region']['name']}' in {start}..{end}. "
            "Cloud-heavy period — widen the date range or raise cloud_percentage.")
    logger.info("Sentinel-2: %d cloud-filtered granules found.", n)

    days = _daily_groups(cfg, col)
    logger.info("Sentinel-2: %d unique acquisition days after de-duplication.", len(days))

    # If GEDI samples exist, filter S2 days to only those within max_days_difference of a GEDI date
    gedi_path = PROJECT_ROOT / "data/raw/gedi/gedi_samples.parquet"
    if gedi_path.exists():
        import pandas as pd
        gedi_df = pd.read_parquet(gedi_path)
        if "acquisition_date" in gedi_df.columns:
            g_dates = [dt.datetime.strptime(str(d), "%Y-%m-%d").date() for d in gedi_df["acquisition_date"].dropna().unique()]
            max_days = int(cfg.get("matching", {}).get("max_days_difference", 30))
            filtered_days = []
            for d_str, ms in days:
                d_date = dt.datetime.strptime(d_str, "%Y-%m-%d").date()
                if any(abs((d_date - g).days) <= max_days for g in g_dates):
                    filtered_days.append((d_str, ms))
            if filtered_days:
                logger.info("Filtered Sentinel-2 days from %d to %d based on GEDI dates.", len(days), len(filtered_days))
                days = filtered_days

    per_day_meta: list[dict[str, Any]] = []
    for day, day_ms in days:
        year = day[:4]
        out_dir = raw_root / year / day
        out_dir.mkdir(parents=True, exist_ok=True)
        files = _export_day(cfg, col, day, day_ms, ee_region, out_dir)
        per_day_meta.append({
            "date": day,
            "acquisition_time_utc": dt.datetime.utcfromtimestamp(day_ms / 1000).isoformat() + "Z",
            "files": [str(out_dir.relative_to(PROJECT_ROOT) / f) for f in files],
        })

    meta = {
        "product": cfg["datasets"]["sentinel2"]["product"],
        "cloud_product": cfg["datasets"]["sentinel2"]["cloud_product"],
        "retrieval_date_utc": dt.datetime.utcnow().isoformat() + "Z",
        "study_region": {"name": cfg["study_region"]["name"], "geometry": region_desc},
        "date_range": {"start": start, "end": end},
        "granule_cloud_percentage_max": cfg["datasets"]["sentinel2"]["cloud_percentage"],
        "pixel_mask": {"band": "cs", "threshold": cfg["datasets"]["sentinel2"]["cs_threshold"],
                       "note": "cs >= threshold = clear; masks clouds AND cloud shadows"},
        "reflectance_scale_factor": SCALE_FACTOR,
        "granules_found": n,
        "unique_days": len(days),
        "dedup_rule": "one cloud-masked mosaic per YYYY-MM-DD",
        "days": per_day_meta,
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sentinel2_metadata.json").write_text(json.dumps(meta, indent=2))

    write_manifest(meta_dir, "sentinel2", {
        "dataset": "sentinel2", "product": meta["product"],
        "version": "harmonized L2A (BOA reflectance)",
        "study_region": cfg["study_region"]["name"],
        "date_range": {"start": start, "end": end},
        "source": "Google Earth Engine Data Catalog (COPERNICUS)",
        "filters": {"cloud_percentage": cfg["datasets"]["sentinel2"]["cloud_percentage"],
                    "cs_threshold": cfg["datasets"]["sentinel2"]["cs_threshold"]},
        "files": [f for d in per_day_meta for f in d["files"]],
        "record_count": len(days),
    })
    return raw_root


if __name__ == "__main__":
    from common import load_config, setup_logging
    setup_logging()
    run(load_config(PROJECT_ROOT / "config/config.yaml"))
