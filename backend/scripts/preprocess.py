"""Optional supporting datasets: biomass reference map, DEM, land cover.

  * Biomass reference : LARSE/GEDI/GEDI04_A_002_MONTHLY — monthly raster
    composites of GEDI L4A AGBD (band 'agbd'), used for secondary visual
    comparison against the footprint-level ground truth.
  * Elevation (DEM)   : NASA/NASADEM_HGT/001, band 'elevation', 30 m.
  * Land cover        : ESA/WorldCover/v200 (ESA WorldCover 2021), band 'Map',
    10 m, 11-class global land cover.

All are exported clipped to the study region only — no global downloads.
"""
from __future__ import annotations

import json
import logging
import datetime as dt
from pathlib import Path

from common import (PipelineError, date_range, dataset_enabled,
                    load_region_geometry, ee_geometry, write_manifest, PROJECT_ROOT)

logger = logging.getLogger("biomass_pipeline.preprocess")


def _export(img, path: Path, ee_region, scale: float) -> Path:
    import geemap
    if path.exists():
        logger.info("  reuse %s", path.name)
        return path
    try:
        geemap.ee_export_image(img, filename=str(path), scale=scale,
                               region=ee_region, file_per_band=False)
    except Exception as exc:
        raise PipelineError(f"Export failed for {path.name}: {exc}") from exc
    if not path.exists() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise PipelineError(f"Corrupted/empty export: {path.name}")
    logger.info("  wrote %s", path.name)
    return path


def run(cfg: dict) -> dict[str, str]:
    """Returns dict of produced file paths (may be empty if all disabled)."""
    import ee
    from common import init_earth_engine
    init_earth_engine(cfg)

    region_geom, region_desc = load_region_geometry(cfg)
    ee_region = ee_geometry(region_geom)
    start, end = date_range(cfg)
    produced: dict[str, str] = {}

    # --- biomass reference (GEDI L4A monthly raster) ---
    if dataset_enabled(cfg, "biomass_reference"):
        ref_cfg = cfg["datasets"]["biomass_reference"]
        out_dir = PROJECT_ROOT / "data/raw/biomass_reference"
        out_dir.mkdir(parents=True, exist_ok=True)
        col = ee.ImageCollection(ref_cfg["product"]) \
            .filterBounds(ee_region).filterDate(start, end).select("agbd")
        n = int(col.size().getInfo())
        if n == 0:
            logger.warning("Biomass reference: no monthly composites in window; "
                           "using median of ALL months overlapping the region.")
            col = ee.ImageCollection(ref_cfg["product"]).filterBounds(ee_region).select("agbd")
        ref_img = col.median().clip(ee_region)
        p = _export(ref_img, out_dir / "gedi_l4a_agbd_reference.tif", ee_region, 25.0)
        produced["biomass_reference"] = str(p.relative_to(PROJECT_ROOT))

    # --- auxiliary: DEM + land cover ---
    aux_cfg = cfg["datasets"].get("auxiliary", {})
    out_dir = PROJECT_ROOT / "data/raw/auxiliary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if aux_cfg.get("dem"):
        dem = ee.Image(aux_cfg["dem"]).select("elevation").clip(ee_region)
        p = _export(dem, out_dir / "nasadem_elevation.tif", ee_region, 30.0)
        produced["dem"] = str(p.relative_to(PROJECT_ROOT))

    if aux_cfg.get("landcover"):
        lc = ee.ImageCollection(aux_cfg["landcover"]).first().select("Map").clip(ee_region)
        p = _export(lc, out_dir / "esa_worldcover_2021.tif", ee_region, 10.0)
        produced["landcover"] = str(p.relative_to(PROJECT_ROOT))

    if produced:
        write_manifest(PROJECT_ROOT / "metadata", "auxiliary", {
            "dataset": "auxiliary", "product": "biomass_reference+DEM+landcover",
            "version": "see per-file products",
            "study_region": cfg["study_region"]["name"],
            "date_range": {"start": start, "end": end},
            "source": "Google Earth Engine Data Catalog",
            "filters": {},
            "files": list(produced.values()),
            "record_count": len(produced),
        })
    return produced


if __name__ == "__main__":
    from common import load_config, setup_logging
    setup_logging()
    run(load_config(PROJECT_ROOT / "config/config.yaml"))
