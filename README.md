# 🌿 Canopy — Forest Biomass & Carbon Stock Monitoring System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900.svg?logo=leaflet&logoColor=white)](https://leafletjs.com/)

**Canopy** is an end-to-end geospatial monitoring platform for high-resolution **Aboveground Biomass (AGB)**, **Carbon Stock**, and **Canopy Density** estimation. Powered by multi-sensor satellite data and machine learning, Canopy provides interactive spatial exploration and point-level biomass predictions across the **Nilgiri Biosphere Reserve (Western Ghats)**.

---

## 📌 Features

- **Interactive Satellite Mapping**: High-performance Leaflet-based map with smooth layer toggling (Google Satellite imagery, biomass heatmap, and study area boundary).
- **Active Study Area Highlighting**: Visible emerald dashed boundary box marking active data coverage (`11.20°N–11.90°N, 76.20°E–76.90°E`) with corner coordinates, interactive tooltips, and a **Focus Area** shortcut.
- **Instant Point Estimation**: Click anywhere within the study region to retrieve localized aboveground biomass (Mg/ha), carbon stock (t/ha), CO₂ equivalent (t/ha), and canopy density percentage.
- **Biomass Heatmap Layer**: Dynamic canvas-based rendering querying viewport cells from the precomputed spatial grid.
- **Robust Error Discrimination**: Distinct, clear UI states for:
  - Outside Study Area (with quick "Focus Study Area" navigation)
  - Non-Forest Cells (with nearest forest distance in meters)
  - Backend Status (loading, healthy, unreachable, not-ready)
- **Precomputed Geospatial Grid**: Fast spatial indexing using `scipy.spatial.cKDTree` over 97,344 grid cells (64,753 active forest cells at 250m resolution).

---

## 🛰️ Data Sources & Methodology

Canopy integrates five authoritative Earth observation datasets processed through the **Google Earth Engine (GEE)** Python API:

| Dataset | Role | Source |
| :--- | :--- | :--- |
| **GEDI L4A** (`LARSE/GEDI/GEDI04_A_002`) | Ground-truth LiDAR Aboveground Biomass Density (AGBD) | NASA / GEE Catalog |
| **Sentinel-2 SR** (`COPERNICUS/S2_SR_HARMONIZED`) | Optical surface reflectance (B2, B3, B4, B8, B11, B12 + NDVI) | Copernicus / ESA |
| **Cloud Score+** (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) | QA cloud and cloud-shadow masking | Google Research |
| **Sentinel-1 GRD** (`COPERNICUS/S1_GRD`) | C-band SAR backscatter (VV, VH polarizations) | Copernicus / ESA |
| **NASADEM** (`NASA/NASADEM_HGT/001`) | High-resolution elevation and slope auxiliary features | NASA JPL |
| **ESA WorldCover v200** (`ESA/WorldCover/v200`) | 10m land cover classification & forest mask | ESA |

Predictions are generated using a validated Random Forest regressor (`v4_random_forest`) trained on coincident GEDI LiDAR footprints and multi-sensor features.

---

## 🏗️ Project Architecture

```
canopy-prototype/
├── frontend/                     # React + Vite dashboard
│   ├── index.html                # Entry HTML document
│   ├── App.tsx                   # Main dashboard UI, map layers & state
│   ├── clientApi.ts              # Typed API client with error handling
│   ├── index.css                 # Styling system, tokens & Leaflet fixes
│   ├── vite.config.ts            # Vite build configuration with API proxy
│   └── package.json              # Frontend dependencies
│
├── backend/                      # Python FastAPI service & ML pipeline
│   ├── api/
│   │   └── main.py               # FastAPI application, CORS & endpoints
│   ├── config/
│   │   └── config.yaml           # Study region bounds & pipeline config
│   ├── data/
│   │   └── processed/
│   │       └── agb_grid.parquet  # Precomputed 250m AGB grid dataset
│   ├── models/
│   │   └── agb_model_final_v4.joblib # Trained production Random Forest model
│   ├── scripts/                  # GEE acquisition, training & grid scripts
│   └── requirements.txt          # Python dependencies
│
├── .env.example                  # Root environment template
└── README.md                     # Documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Node.js**: v18+ and npm
- **Python**: v3.11 or v3.12

### 2. Backend Setup (FastAPI)

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend will initialize and load the precomputed grid and model:
- API URL: `http://127.0.0.1:8000`
- Swagger UI Documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 3. Frontend Setup (React + Vite)

In a new terminal window:

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)** in your browser.

---

## 📡 API Endpoints

### `GET /health`
Returns service status, model version, region bounds, and summary statistics.
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
  "mean_forest_agb_mg_ha": 59.07,
  "mean_forest_canopy_density_pct": 79.7,
  "mean_forest_ndvi_canopy_proxy": 0.7974
}
```

### `GET /agb?lat={lat}&lon={lon}`
Queries AGB and carbon stock for a specific coordinate inside the study region.
- **200 OK**: Forest cell with biomass estimates
- **400 Bad Request**: Point is outside the study area
- **422 Unprocessable Entity**: Point is inside study area but non-forest (returns nearest forest distance)

```json
{
  "agb_mg_ha": 57.07,
  "carbon_tons_per_ha": 26.82,
  "co2_equivalent_tons_per_ha": 98.44,
  "canopy_density_pct": 87.3,
  "ndvi_canopy_proxy": 0.8731,
  "grid_cell_lat": 11.499558,
  "grid_cell_lon": 76.499405,
  "distance_to_cell_m": 81.33
}
```

### `GET /agb/region?min_lat={}&min_lon={}&max_lat={}&max_lon={}&forest_only=true`
Fetches precomputed grid cells within a bounding box for heatmap rendering.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` or `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The frontend includes built-in proxying and fallback handling, ensuring seamless connectivity to the backend service.

---

## 📄 License

This project is licensed under the MIT License.
