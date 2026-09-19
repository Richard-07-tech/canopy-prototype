"""FastAPI Prediction Service for Aboveground Biomass (AGB) Estimation.

Serves predictions from the precomputed v4 grid across the study region:
- GET /agb?lat={lat}&lon={lon}
- GET /agb/region?min_lat={min_lat}&min_lon={min_lon}&max_lat={max_lat}&max_lon={max_lon}
- GET /health
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger("biomass_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global state loaded once at startup
STATE: dict[str, Any] = {
    "grid_df": None,
    "tree": None,
    "coords": None,
    "model": None,
    "bounds": {
        "min_lat": 11.20,
        "max_lat": 11.90,
        "min_lon": 76.20,
        "max_lon": 76.90,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in metres between two points."""
    R = 6371000.0  # Earth mean radius in metres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load grid dataset, spatial tree, and model once at startup."""
    grid_path = PROJECT_ROOT / "data/processed/agb_grid.parquet"
    model_path = PROJECT_ROOT / "models/agb_model_final_v4.joblib"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Precomputed grid not found at {grid_path}. "
            "Please run 'python scripts/precompute_agb_grid.py' first."
        )

    logger.info("Loading precomputed AGB grid from %s...", grid_path)
    df = pd.read_parquet(grid_path)
    STATE["grid_df"] = df

    logger.info("Building cKDTree spatial index for %d grid cells...", len(df))
    # Note: KDTree uses (lat, lon) coordinates
    coords = df[["lat", "lon"]].to_numpy()
    STATE["coords"] = coords
    STATE["tree"] = cKDTree(coords)

    if model_path.exists():
        logger.info("Loading v4 model from %s...", model_path)
        STATE["model"] = joblib.load(model_path)
    else:
        logger.warning("Model file %s not found. Grid inference will still function.", model_path)

    forest_count = int(df["is_forest"].sum())
    logger.info(
        "Biomass API initialized successfully: %d total cells, %d forest cells (%.1f%%).",
        len(df),
        forest_count,
        100.0 * forest_count / len(df),
    )
    yield
    logger.info("Shutting down Biomass API.")


app = FastAPI(
    title="Biomass & Carbon Stock Prediction API",
    description="FastAPI service for high-resolution Aboveground Biomass (AGB) and carbon stock estimation using Sentinel-1/2, GEDI, NASADEM, and ESA WorldCover.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health and status check")
async def health_check():
    """Returns service health status, model metadata, and region bounds."""
    df = STATE.get("grid_df")
    if df is None:
        raise HTTPException(status_code=503, detail="Grid data not loaded.")

    forest_mask = df["is_forest"]
    return {
        "status": "healthy",
        "model_version": "v4_random_forest",
        "study_region": STATE["bounds"],
        "grid_resolution_m": 250,
        "total_cells": len(df),
        "forest_cells": int(forest_mask.sum()),
        "mean_forest_agb_mg_ha": round(float(df.loc[forest_mask, "agb_mg_ha"].mean()), 2),
        "mean_forest_canopy_density_pct": round(float(df.loc[forest_mask, "canopy_density_pct"].mean()), 1) if "canopy_density_pct" in df.columns else None,
        "mean_forest_ndvi_canopy_proxy": round(float(df.loc[forest_mask, "ndvi_canopy_proxy"].mean()), 4) if "ndvi_canopy_proxy" in df.columns else None,
    }


@app.get("/agb", summary="Predict AGB at a specific coordinate")
async def get_agb(
    lat: float = Query(..., description="Latitude (WGS84)", examples=[11.55]),
    lon: float = Query(..., description="Longitude (WGS84)", examples=[76.55]),
):
    """Look up nearest precomputed grid cell for a given latitude and longitude.

    Returns:
    - agb_mg_ha: Aboveground Biomass density (Mg/ha)
    - carbon_tons_per_ha: Carbon density (AGB * 0.47)
    - co2_equivalent_tons_per_ha: CO₂-equivalent (Carbon * 3.67)
    - canopy_density_pct: Estimated canopy density percentage (NDVI proxy scaled to 0-100%)
    - ndvi_canopy_proxy: Sentinel-2 NDVI canopy density proxy
    - grid_cell_lat, grid_cell_lon: Coordinates of the matched cell center
    - distance_to_cell_m: Distance in metres to the matched cell center
    """
    bounds = STATE["bounds"]
    if not (bounds["min_lat"] <= lat <= bounds["max_lat"] and bounds["min_lon"] <= lon <= bounds["max_lon"]):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested point ({lat}, {lon}) is outside the study region bounds: "
                f"latitude [{bounds['min_lat']}, {bounds['max_lat']}], "
                f"longitude [{bounds['min_lon']}, {bounds['max_lon']}]."
            ),
        )

    tree: cKDTree = STATE["tree"]
    df: pd.DataFrame = STATE["grid_df"]

    # Query nearest neighbour
    dist_deg, idx = tree.query([lat, lon], k=1)
    matched_row = df.iloc[idx]

    cell_lat = float(matched_row["lat"])
    cell_lon = float(matched_row["lon"])
    distance_m = haversine_m(lat, lon, cell_lat, cell_lon)

    # Check non-forest mask
    if not bool(matched_row["is_forest"]) or pd.isna(matched_row["agb_mg_ha"]):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Non-forest land cover",
                "message": (
                    f"Requested coordinate ({lat:.5f}, {lon:.5f}) falls on non-forest land cover "
                    "(e.g., cropland, built-up, water, or barren) according to ESA WorldCover. "
                    "Biomass is not estimated for non-forest cells."
                ),
                "grid_cell_lat": round(cell_lat, 6),
                "grid_cell_lon": round(cell_lon, 6),
                "distance_to_cell_m": round(distance_m, 2),
            },
        )

    agb = float(matched_row["agb_mg_ha"])
    carbon = float(matched_row["carbon_tons_per_ha"])
    co2 = float(matched_row["co2_equivalent_tons_per_ha"])
    canopy_pct = float(matched_row["canopy_density_pct"]) if "canopy_density_pct" in matched_row and pd.notna(matched_row["canopy_density_pct"]) else None
    ndvi_canopy = float(matched_row["ndvi_canopy_proxy"]) if "ndvi_canopy_proxy" in matched_row and pd.notna(matched_row["ndvi_canopy_proxy"]) else None

    return {
        "agb_mg_ha": round(agb, 2),
        "carbon_tons_per_ha": round(carbon, 2),
        "co2_equivalent_tons_per_ha": round(co2, 2),
        "canopy_density_pct": round(canopy_pct, 1) if canopy_pct is not None else None,
        "ndvi_canopy_proxy": round(ndvi_canopy, 4) if ndvi_canopy is not None else None,
        "grid_cell_lat": round(cell_lat, 6),
        "grid_cell_lon": round(cell_lon, 6),
        "distance_to_cell_m": round(distance_m, 2),
    }


@app.get("/agb/region", summary="Query AGB grid cells for a sub-region (heatmap/dashboard)")
async def get_agb_region(
    min_lat: float = Query(..., description="Minimum latitude", examples=[11.40]),
    min_lon: float = Query(..., description="Minimum longitude", examples=[76.40]),
    max_lat: float = Query(..., description="Maximum latitude", examples=[11.50]),
    max_lon: float = Query(..., description="Maximum longitude", examples=[76.50]),
    forest_only: bool = Query(True, description="Filter to forest cells only"),
):
    """Retrieve grid cells within a bounding box for heatmap rendering and spatial aggregation."""
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bounding box: min coordinates must be <= max coordinates.")

    df: pd.DataFrame = STATE["grid_df"]
    mask = (df["lat"] >= min_lat) & (df["lat"] <= max_lat) & (df["lon"] >= min_lon) & (df["lon"] <= max_lon)

    if forest_only:
        mask &= df["is_forest"]

    sub_df = df[mask]
    has_canopy = "canopy_density_pct" in sub_df.columns
    has_ndvi = "ndvi_canopy_proxy" in sub_df.columns

    cells = []
    for r in sub_df.itertuples(index=False):
        cells.append({
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
            "agb_mg_ha": round(float(r.agb_mg_ha), 2) if pd.notna(r.agb_mg_ha) else None,
            "carbon_tons_per_ha": round(float(r.carbon_tons_per_ha), 2) if pd.notna(r.carbon_tons_per_ha) else None,
            "co2_equivalent_tons_per_ha": round(float(r.co2_equivalent_tons_per_ha), 2) if pd.notna(r.co2_equivalent_tons_per_ha) else None,
            "canopy_density_pct": round(float(r.canopy_density_pct), 1) if has_canopy and pd.notna(r.canopy_density_pct) else None,
            "ndvi_canopy_proxy": round(float(r.ndvi_canopy_proxy), 4) if has_ndvi and pd.notna(r.ndvi_canopy_proxy) else None,
            "is_forest": bool(r.is_forest),
        })

    return {
        "count": len(cells),
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "cells": cells,
    }
