# Biomass Dashboard Frontend Handoff

This document is the frontend build contract for the working biomass backend.
The frontend should be a browser dashboard that reads JSON from the FastAPI
service. It should not read Parquet files, load the model, or call Google Earth
Engine directly.

## Backend URL

For local development, start the backend from the repository root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use this as the frontend API base URL:

```text
http://127.0.0.1:8000
```

The backend currently allows browser requests through CORS. The interactive
OpenAPI reference is available at `http://127.0.0.1:8000/docs`.

## Study region

The current backend region is the Nilgiri / Western Ghats bounding box:

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| Latitude | 11.20 | 11.90 |
| Longitude | 76.20 | 76.90 |

Use WGS84 latitude/longitude (`EPSG:4326`). A coordinate outside these bounds
returns HTTP `400` from the point endpoint.

## Endpoints

### `GET /health`

Use this on application startup to confirm that the backend and precomputed
grid are ready.

Example response:

```json
{
  "status": "healthy",
  "model_version": "v4_random_forest",
  "study_region": {
    "min_lat": 11.2,
    "max_lat": 11.9,
    "min_lon": 76.2,
    "max_lon": 76.9
  },
  "grid_resolution_m": 250,
  "total_cells": 97344,
  "forest_cells": 64753,
  "mean_forest_agb_mg_ha": 123.45,
  "mean_forest_canopy_density_pct": 62.1,
  "mean_forest_ndvi_canopy_proxy": 0.621
}
```

The numeric summary values are generated from the current grid and may change
when the backend dataset is regenerated.

### `GET /agb`

Returns the nearest precomputed forest grid cell for one map coordinate.

```text
GET /agb?lat=11.55&lon=76.55
```

Successful response:

```json
{
  "agb_mg_ha": 185.42,
  "carbon_tons_per_ha": 87.15,
  "co2_equivalent_tons_per_ha": 319.74,
  "canopy_density_pct": 74.3,
  "ndvi_canopy_proxy": 0.743,
  "grid_cell_lat": 11.551,
  "grid_cell_lon": 76.549,
  "distance_to_cell_m": 83.24
}
```

Use this endpoint for a clicked-point detail panel or coordinate search.

Units and meaning:

* `agb_mg_ha`: above-ground biomass density in Mg/ha.
* `carbon_tons_per_ha`: estimated carbon stock in metric tons/ha.
* `co2_equivalent_tons_per_ha`: estimated CO2 equivalent in metric tons/ha.
* `canopy_density_pct`: NDVI-derived canopy proxy from 0 to 100 percent.
* `ndvi_canopy_proxy`: underlying NDVI-derived value.
* `distance_to_cell_m`: distance from the requested point to the matched grid
  cell center, in meters.

### `GET /agb/region`

Returns grid cells inside a bounding box for a heatmap, map layer, or regional
summary.

```text
GET /agb/region?min_lat=11.40&min_lon=76.40&max_lat=11.50&max_lon=76.50&forest_only=true
```

Response shape:

```json
{
  "count": 2,
  "bounds": {
    "min_lat": 11.4,
    "max_lat": 11.5,
    "min_lon": 76.4,
    "max_lon": 76.5
  },
  "cells": [
    {
      "lat": 11.451,
      "lon": 76.449,
      "agb_mg_ha": 185.42,
      "carbon_tons_per_ha": 87.15,
      "co2_equivalent_tons_per_ha": 319.74,
      "canopy_density_pct": 74.3,
      "ndvi_canopy_proxy": 0.743,
      "is_forest": true
    }
  ]
}
```

`forest_only=true` is the default and is recommended for biomass heatmaps.
Set `forest_only=false` only when the UI needs to show non-forest land-cover
cells as a separate map state. Large bounding boxes can return many cells, so
request only the current map viewport.

## JavaScript example

Keep the API base URL in one environment variable in the frontend app. For a
Vite app, use `VITE_API_BASE_URL=http://127.0.0.1:8000`.

```js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export async function fetchAgbAtPoint(lat, lon) {
  const params = new URLSearchParams({ lat, lon });
  const response = await fetch(`${API_BASE_URL}/agb?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail?.message || error.detail || "AGB lookup failed");
  }

  return response.json();
}

export async function fetchAgbRegion(bounds) {
  const params = new URLSearchParams({
    min_lat: bounds.minLat,
    min_lon: bounds.minLon,
    max_lat: bounds.maxLat,
    max_lon: bounds.maxLon,
    forest_only: "true"
  });

  const response = await fetch(`${API_BASE_URL}/agb/region?${params}`);
  if (!response.ok) throw new Error("Regional biomass lookup failed");
  return response.json();
}
```

## Required UI states

The frontend should visibly handle these states:

* Loading: backend health check or map request is in progress.
* Healthy: display dashboard data and map layers.
* `400`: invalid coordinate or bounding box, or point outside the study area.
* `422`: selected point is non-forest and has no biomass estimate. Show the
  explanation from `detail.message` instead of treating it as a server crash.
* `503`: the backend grid is unavailable. Show a backend-not-ready message.
* Network failure: the API is stopped, unreachable, or the configured base URL
  is wrong.

## Suggested dashboard views

Build the first screen around the actual backend capabilities:

1. A map centered on the current study region with a biomass or canopy
   heatmap built from `/agb/region`.
2. A click interaction that calls `/agb` and opens a detail panel.
3. Summary cards populated from `/health` for total cells, forest cells, mean
   AGB, and mean canopy density.
4. A clear non-forest state for cells that return HTTP `422`.

Do not display a value as a precise field measurement: the API returns a
250-meter precomputed grid estimate and identifies the distance to the matched
cell center.

## Backend prerequisites

The API requires these repository files to exist:

* `data/processed/agb_grid.parquet`
* `models/agb_model_final_v4.joblib`

The frontend developer does not need to regenerate either file. If the backend
fails during startup because the grid is missing, the backend owner should run
the Earth Engine precomputation step described in the root `README.md`.
