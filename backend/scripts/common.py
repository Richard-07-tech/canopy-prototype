"""Shared utilities: config loading, geometry handling, Earth Engine init,
manifest writing, and error types. No dataset-specific logic lives here.

Security: credentials are NEVER read from config files. Earth Engine uses the
standard credential chain (env var EE_PROJECT + `earthengine authenticate`
token, or the GOOGLE_APPLICATION_CREDENTIALS service-account path). See .env.example.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("biomass_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- errors
class PipelineError(RuntimeError):
    """Base class for expected, user-facing pipeline failures."""


class ConfigError(PipelineError):
    pass


class GeometryError(PipelineError):
    pass


class AuthError(PipelineError):
    pass


class EmptyQueryError(PipelineError):
    """Raised when a dataset search returns zero items for region+time."""


# ---------------------------------------------------------------- logging
def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    log_dir = log_dir or PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_dir / f"pipeline_{stamp}.log"))
    except OSError as exc:  # e.g. read-only fs — log to console only
        print(f"[warn] cannot write log file ({exc}); logging to console only")
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------- config
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in ("project", "study_region", "date_range", "datasets", "matching", "output"):
        if key not in cfg:
            raise ConfigError(f"Missing required config section: '{key}'")
    start = _parse_date(cfg["date_range"]["start"])
    end = _parse_date(cfg["date_range"]["end"])
    if start > end:
        raise ConfigError(
            f"date_range.start ({start}) is after date_range.end ({end})"
        )
    sr = cfg["study_region"]
    if not sr.get("geometry_file") and not sr.get("bbox"):
        raise ConfigError(
            "study_region needs either 'geometry_file' or 'bbox'."
        )
    if sr.get("bbox"):
        bbox = sr["bbox"]
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ConfigError(f"Invalid bbox {bbox}: must be [xmin,ymin,xmax,ymax] "
                              "with xmin<xmax, ymin<ymax.")


def _parse_date(s: str) -> dt.date:
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ConfigError(f"Invalid date '{s}' (expected YYYY-MM-DD)") from exc


# ---------------------------------------------------------------- geometry
def load_region_geometry(cfg: dict[str, Any]):
    """Return (shapely_geometry, description). GeoJSON/Shapefile/GPKG or bbox."""
    sr = cfg["study_region"]
    if sr.get("geometry_file"):
        path = Path(sr["geometry_file"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise GeometryError(f"Geometry file not found: {path}")
        import geopandas as gpd
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise GeometryError(f"Geometry file {path} contains no features.")
        if gdf.crs is None:
            raise GeometryError(f"Geometry file {path} has no CRS defined.")
        gdf = gdf.to_crs("EPSG:4326")
        geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
        return geom, f"file:{path.name}"
    bbox = sr["bbox"]
    from shapely.geometry import box
    return box(*bbox), f"bbox:{bbox}"


def ee_geometry(shapely_geom):
    """Convert a shapely geometry to ee.Geometry."""
    import ee
    try:
        return ee.Geometry(shapely_geom.__geo_interface__)
    except Exception:
        import geemap
        if hasattr(geemap, "geojson_to_ee"):
            return geemap.geojson_to_ee(shapely_geom.__geo_interface__)
        raise


# ---------------------------------------------------------------- Earth Engine
def init_earth_engine(cfg: dict[str, Any]) -> None:
    """Initialize EE. Raises AuthError with a remediation message on failure."""
    try:
        import ee
    except ImportError as exc:
        raise AuthError("earthengine-api is not installed. `pip install earthengine-api geemap`") from exc

    project = (cfg.get("project", {}).get("ee_project") or "").strip() \
        or os.environ.get("EE_PROJECT", "").strip() or None
    try:
        ee.Initialize(project=project)
        logger.info("Earth Engine initialized (project=%s)", project or "<default>")
    except Exception as exc:  # ee raises several exception types
        raise AuthError(
            "Earth Engine initialization failed: %s\n"
            "Fix: run `earthengine authenticate`, then set ee_project in "
            "config.yaml or export EE_PROJECT=<your-cloud-project>."
            % exc
        ) from exc


# ---------------------------------------------------------------- manifest
def write_manifest(meta_dir: Path, name: str, record: dict[str, Any]) -> Path:
    """Append/merge an acquisition manifest entry into acquisition_manifest.json."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "acquisition_manifest.json"
    base: dict[str, Any] = {"manifest_created_utc": dt.datetime.utcnow().isoformat() + "Z",
                            "acquisitions": []}
    if manifest_path.exists():
        try:
            base = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupted manifest %s — starting a new one.", manifest_path)
    record = dict(record)
    record.setdefault("retrieval_timestamp", dt.datetime.utcnow().isoformat() + "Z")
    base["acquisitions"] = [a for a in base.get("acquisitions", [])
                            if a.get("dataset") != name]
    base["acquisitions"].append(record)
    manifest_path.write_text(json.dumps(base, indent=2, default=str))
    logger.info("Manifest updated: %s (%s)", manifest_path, name)
    return manifest_path


def dataset_enabled(cfg: dict[str, Any], key: str) -> bool:
    return bool(cfg.get("datasets", {}).get(key, {}).get("enabled", False))


def date_range(cfg: dict[str, Any]) -> tuple[str, str]:
    return str(cfg["date_range"]["start"]), str(cfg["date_range"]["end"])
