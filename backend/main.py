#!/usr/bin/env python3
"""Biomass data-acquisition pipeline — command-line interface.

Usage:
    python main.py --config config/config.yaml                 # full pipeline
    python main.py --config config/config.yaml --dataset gedi  # one dataset
    python main.py --config config/config.yaml --dataset all
    python main.py --config config/config.yaml --validate      # checks only

Exit codes: 0 = success, 2 = validation/config error, 1 = runtime failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from common import (AuthError, ConfigError, EmptyQueryError, GeometryError,
                    PipelineError, dataset_enabled, date_range, load_config,
                    load_region_geometry, setup_logging, PROJECT_ROOT)

logger = logging.getLogger("biomass_pipeline.main")

DATASETS = ("gedi", "sentinel1", "sentinel2", "all")


# ------------------------------------------------------------------ validate
def validate(cfg: dict) -> bool:
    """Check config, credentials, region, dataset availability, directories."""
    import datetime as dt
    ok = True

    def check(label: str, good: bool, hint: str = ""):
        nonlocal ok
        print(f"  [{'OK' if good else 'FAIL'}] {label}" + (f" — {hint}" if hint and not good else ""))
        ok = ok and good

    print("Validation report")
    print("-----------------")

    # 1. configuration already parsed by load_config (would have raised)
    check("config.yaml parses and has required sections", True)

    # 2. study region
    try:
        geom, desc = load_region_geometry(cfg)
        check(f"study region loads ({desc}), area ok", geom.is_valid and not geom.is_empty)
    except GeometryError as exc:
        check("study region loads", False, str(exc))

    # 3. date range sanity + GEDI availability (GEDI L4A starts 2019-04-18)
    start = dt.datetime.strptime(date_range(cfg)[0], "%Y-%m-%d").date()
    check("date_range is valid", True)
    if cfg["datasets"]["gedi"]["enabled"] and start < dt.date(2019, 4, 18):
        check("GEDI L4A covers requested dates", False,
              "GEDI L4A v2.1 starts 2019-04-18; move date_range.start forward.")
    elif cfg["datasets"]["gedi"]["enabled"]:
        check("GEDI L4A covers requested dates", True)

    # 4. output directories writable
    try:
        for d in ("data/raw/gedi", "data/raw/sentinel1", "data/raw/sentinel2",
                  "data/raw/biomass_reference", "data/raw/auxiliary",
                  "data/processed/training", "metadata", "logs"):
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        probe = PROJECT_ROOT / "logs/.write_probe"
        probe.write_text("ok"); probe.unlink()
        check("output directories exist and are writable", True)
    except OSError as exc:
        check("output directories exist and are writable", False, str(exc))

    # 5. Earth Engine credentials + dataset availability (only if any EE dataset on)
    any_ee = any(dataset_enabled(cfg, k)
                 for k in ("gedi", "sentinel1", "sentinel2", "biomass_reference"))
    if any_ee:
        try:
            from common import init_earth_engine
            import ee
            init_earth_engine(cfg)
            check("Earth Engine authentication", True)
            try:
                geom, _ = load_region_geometry(cfg)
                from common import ee_geometry
                ee_region = ee_geometry(geom)
                start_s, end_s = date_range(cfg)
                from download_gedi import load_gedi_collection
                probes = {
                    "GEDI L4A": load_gedi_collection(cfg, ee_region).size(),
                    "Sentinel-2": ee.ImageCollection(cfg["datasets"]["sentinel2"]["product"])
                        .filterBounds(ee_region).filterDate(start_s, end_s).size(),
                    "Sentinel-1": ee.ImageCollection(cfg["datasets"]["sentinel1"]["product"])
                        .filterBounds(ee_region).filterDate(start_s, end_s).size(),
                }
                for name, cnt in probes.items():
                    n = int(cnt.getInfo())
                    check(f"{name} availability in region/time ({n} items)", n > 0,
                          "empty query — widen date range or check the region.")
            except Exception as exc:
                check("dataset availability probe", False, str(exc))
        except AuthError as exc:
            check("Earth Engine authentication", False, str(exc).splitlines()[0])
    else:
        print("  [SKIP] Earth Engine checks (all EE datasets disabled)")

    print("-----------------")
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED — fix the items above.")
    return ok


# ------------------------------------------------------------------ pipeline
def run_pipeline(cfg: dict, dataset: str) -> None:
    import importlib
    steps: list[tuple[str, str]] = []
    if dataset in ("gedi", "all") and dataset_enabled(cfg, "gedi"):
        steps.append(("GEDI L4A footprints", "download_gedi"))
    if dataset in ("sentinel2", "all") and dataset_enabled(cfg, "sentinel2"):
        steps.append(("Sentinel-2 SR", "download_sentinel2"))
    if dataset in ("sentinel1", "all") and dataset_enabled(cfg, "sentinel1"):
        steps.append(("Sentinel-1 GRD", "download_sentinel1"))
    if dataset == "all" and (dataset_enabled(cfg, "biomass_reference")):
        steps.append(("Auxiliary (ref biomass/DEM/landcover)", "preprocess"))

    for label, module_name in steps:
        logger.info("=== Stage: %s ===", label)
        importlib.import_module(module_name).run(cfg)

    if dataset == "all":
        logger.info("=== Stage: training dataset build (spatial+temporal matching) ===")
        importlib.import_module("build_training_dataset").run(cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Geospatial dataset retrieval agent "
                                             "for forest AGB estimation.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config/config.yaml"),
                    help="Path to config.yaml")
    ap.add_argument("--dataset", choices=DATASETS, default=None,
                    help="Run only one dataset stage (default: full pipeline)")
    ap.add_argument("--validate", action="store_true",
                    help="Run validation checks only (no downloads)")
    args = ap.parse_args()

    setup_logging()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    if args.validate:
        return 0 if validate(cfg) else 2

    try:
        run_pipeline(cfg, args.dataset or "all")
    except EmptyQueryError as exc:
        logger.error("Empty query: %s", exc)
        return 1
    except AuthError as exc:
        logger.error("Authentication error: %s", exc)
        return 2
    except PipelineError as exc:
        logger.error("Pipeline failure: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error (traceback above) — not swallowed.")
        return 1
    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
