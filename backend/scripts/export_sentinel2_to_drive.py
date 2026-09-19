"""Standalone launcher for Sentinel-2 Earth Engine Export-to-Drive.

This script submits background export tasks via ee.batch.Export.image.toDrive()
to bypass the 50MB direct download limit of Earth Engine's getDownloadURL/getPixels endpoints.
"""
from __future__ import annotations

import json
import logging
import sys
import datetime as dt
from pathlib import Path

# Add scripts directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, init_earth_engine, load_region_geometry, ee_geometry, date_range, PROJECT_ROOT
from download_sentinel2 import _masked_collection, _daily_groups, export_day_to_drive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("s2_drive_export")


def main():
    cfg_path = PROJECT_ROOT / "config/config.yaml"
    cfg = load_config(cfg_path)
    init_earth_engine(cfg)

    region_geom, region_desc = load_region_geometry(cfg)
    ee_region = ee_geometry(region_geom)
    start, end = date_range(cfg)

    logger.info("Setting up Sentinel-2 collection for region %s, dates %s to %s...",
                cfg["study_region"]["name"], start, end)

    col = _masked_collection(cfg, ee_region)
    n = int(col.size().getInfo())
    logger.info("Found %d cloud-filtered Sentinel-2 granules.", n)

    days = _daily_groups(cfg, col)
    logger.info("De-duplicated to %d unique acquisition days.", len(days))

    # Match against GEDI footprint dates if available
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
                logger.info("Filtered Sentinel-2 days from %d to %d based on GEDI dates (%s).",
                            len(days), len(filtered_days), g_dates)
                days = filtered_days

    all_tasks = []
    folder = "forest_agb_pipeline_exports"
    logger.info("Submitting Export.image.toDrive tasks to Drive folder '%s'...", folder)

    for day, day_ms in days:
        tasks = export_day_to_drive(cfg, col, day, ee_region, folder=folder)
        all_tasks.extend(tasks)

    print("\n" + "=" * 60)
    print(f"SUBMITTED {len(all_tasks)} EARTH ENGINE BATCH EXPORT TASKS TO DRIVE")
    print("=" * 60)
    summary_data = []
    for t in all_tasks:
        print(f"Task Description : {t['description']}")
        print(f"Task ID          : {t['task_id']}")
        print(f"Target File      : {folder}/{t['filename']}")
        print("-" * 60)
        summary_data.append({
            "task_id": t["task_id"],
            "description": t["description"],
            "day": t["day"],
            "band": t["band"],
            "filename": t["filename"]
        })

    tasks_log = PROJECT_ROOT / "metadata/s2_drive_export_tasks.json"
    tasks_log.parent.mkdir(parents=True, exist_ok=True)
    tasks_log.write_text(json.dumps(summary_data, indent=2))
    logger.info("Saved task records to %s", tasks_log)


if __name__ == "__main__":
    main()
