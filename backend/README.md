# Forest Biomass (AGB) Data-Acquisition Pipeline

A reproducible, region-driven retrieval agent that builds a ready-for-ML
training dataset for **Above-Ground Biomass (AGB) estimation** from satellite
data. It never downloads global datasets — every stage applies the chain

```
Study Region → Dataset Search → Spatial Filtering → Temporal Filtering
→ Quality Filtering → Data Extraction → Local Storage → Metadata → ML Dataset
```

---

## 1. Project objective

Given a user-defined study region (GeoJSON/Shapefile **or** bbox) and a date
range, retrieve and organize:

| Dataset | Role | Official source used |
|---|---|---|
| GEDI L4A (`LARSE/GEDI/GEDI04_A_002`, v2.1) | Ground truth AGBD (footprints) | Google Earth Engine Data Catalog |
| Sentinel-2 SR (`COPERNICUS/S2_SR_HARMONIZED`) | Optical features (B2,B3,B4,B8,B11,B12 + NDVI) | Google Earth Engine Data Catalog |
| Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) | Pixel cloud & cloud-shadow masking (`cs` band) | Google Earth Engine Data Catalog |
| Sentinel-1 GRD (`COPERNICUS/S1_GRD`) | SAR features (VV, VH, dB) | Google Earth Engine Data Catalog |
| GEDI L4A monthly raster (`LARSE/GEDI/GEDI04_A_002_MONTHLY`) | Reference biomass map (visual validation) | Google Earth Engine Data Catalog |
| NASADEM (`NASA/NASADEM_HGT/001`) | Elevation auxiliary feature | Google Earth Engine Data Catalog |
| ESA WorldCover v200 (`ESA/WorldCover/v200`) | Forest/land-cover mask | Google Earth Engine Data Catalog |

All dataset IDs above were verified against the Earth Engine Data Catalog
(developers.google.com/earth-engine/datasets) at the time of writing.

## 2. Dataset sources & access

All retrieval goes through the **official Google Earth Engine Python API**.
No web scraping, no unofficial mirrors. Sentinel products originate from the
Copernicus programme; GEDI products from NASA LARSE/ORNL DAAC — both are
served natively inside the EE catalog, so a single authenticated client covers
everything. (If you prefer pulling raw GEDI HDF5 granules directly, use NASA
Earthdata login — see `.env.example` — but that path is not required here.)

## 3. Required accounts / authentication

1. A **Google Cloud project** with the **Earth Engine API enabled**
   (registered for noncommercial or commercial use).
2. Run once: `earthengine authenticate` (stores a token locally).
3. Put the project id in `config/config.yaml → project.ee_project` **or**
   `export EE_PROJECT=...`. For CI, set
   `GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json`.

No credentials are ever stored in this repo — see `.env.example`.

## 4. Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Configuration

Edit `config/config.yaml`. Key knobs:

* `study_region.geometry_file` (GeoJSON/SHP/GPKG) **or** `study_region.bbox`
  `[xmin, ymin, xmax, ymax]` in EPSG:4326. If a file is provided it wins;
  scripts never hard-code a region.
* `date_range.start/end` (YYYY-MM-DD).
* `datasets.*.enabled` and per-dataset filters (GEDI quality flags,
  Sentinel-2 cloud percentage & Cloud Score+ threshold, Sentinel-1 mode /
  polarization / orbit pass).
* `matching.max_days_difference` — temporal matching window (days).
* `matching.spatial.method` — `center_pixel` | `footprint_mean` |
  `footprint_median` (see §10).

## 6. Running the pipeline

```bash
python main.py --config config/config.yaml --validate   # checks only
python main.py --config config/config.yaml              # full pipeline
python main.py --config config/config.yaml --dataset gedi
python main.py --config config/config.yaml --dataset sentinel1
python main.py --config config/config.yaml --dataset sentinel2
python main.py --config config/config.yaml --dataset all
```

**First milestone (recommended):** keep the shipped small test region
(~25×28 km in Queensland, Australia) and a 3-month window. A successful first
run proves: region → GEDI retrieved → S-1 retrieved → S-2 retrieved → data
saved → metadata saved. Only then scale `study_region`/`date_range` up.

`--validate` checks: config, credentials, study region validity, dataset
availability probes (counts per dataset for your region+dates), and output
directory writability — without downloading anything.

## 7. Directory structure

```
biomass_project/
├── data/
│   ├── raw/           # as-retrieved rasters/parquet (idempotent cache)
│   │   ├── gedi/  sentinel1/  sentinel2/  biomass_reference/  auxiliary/
│   ├── processed/     # gedi/ sentinel1/ sentinel2/ training/
│   └── samples/
├── metadata/          # per-dataset metadata + acquisition_manifest.json
├── config/config.yaml
├── scripts/           # download_gedi / download_sentinel1 / _sentinel2 /
│                      # preprocess / build_training_dataset / common
├── logs/
├── main.py  requirements.txt  .env.example  README.md
```

## 8. Dataset descriptions

**GEDI L4A** — footprint-level AGBD (Mg/ha) with prediction SE, quality flags,
sensitivity, beam id and acquisition time. Only footprints passing documented
filters are kept: `l4_quality_flag==1`, `degrade_flag==0`,
`sensitivity>=0.9`, `solar_elevation<0`, `agbd<=600`. Raw vs. retained counts
are reported — nothing is silently dropped.

**Sentinel-2 SR** — harmonized BOA reflectance (scale 10000). Granules are
pre-filtered by `CLOUDY_PIXEL_PERCENTAGE`, pixels masked with Cloud Score+
`cs>=0.6` (clouds **and** shadows), and granules sharing a calendar day are
mosaicked into **one** export per day (no duplicate/overlapping downloads).
NDVI is added as a band. Layout: `raw/sentinel2/YYYY/YYYY-MM-DD/B02.tif…NDVI.tif`.

**Sentinel-1 GRD** — σ⁰ backscatter in **dB**; EE has already applied orbit
update, border/thermal-noise removal, radiometric calibration and terrain
correction. We filter by `instrumentMode=IW`, dual-pol VV+VH and optional
`orbitProperties_pass`; per-scene orbit direction, relative orbit and mode are
stored in `metadata/sentinel1_metadata.json`, so incompatible configurations
are recorded, never silently mixed.

## 9. Quality control

`build_training_dataset.py` writes `metadata/qc_report.json` with:
total/valid/removed GEDI counts, S-1/S-2 match counts, final sample count,
per-feature missing-value counts, AGB min/max/mean/median/std, and AGB values
outside configurable `[agb_min, agb_max]` **flagged (listed) but not deleted**.

## 10. Spatial & temporal matching

* **Temporal:** for each GEDI footprint we pick the S-2 day and S-1 scene
  with the smallest `|Δdays|` within `matching.max_days_difference`. Actual
  differences are stored as `sentinel2_day_difference` /
  `sentinel1_day_difference`. No match in-window ⇒ explicit `NaN`.
* **Spatial:** a GEDI footprint (~25 m diameter) spans several 10 m Sentinel
  pixels, so extraction is footprint-aware: `footprint_mean` (default) or
  `footprint_median` over a buffered window, or `center_pixel` if configured
  (`fallback_method` covers fully-masked windows). The method used is recorded
  in the QC report.

Output: `data/processed/training/biomass_training_dataset.parquet` with
`latitude, longitude, gedi_agb, gedi_quality, gedi_date, sentinel2_date,
sentinel2_day_difference, s2_B2…s2_B12, ndvi, sentinel1_date,
sentinel1_day_difference, sentinel1_orbit_direction, s1_VV, s1_VH,
elevation, landcover`.

## 11. Expected outputs & reproducibility

Every stage appends to `metadata/acquisition_manifest.json` (dataset, product,
version, region, date range, retrieval timestamp, source, filters, files,
record count). All downloads are **idempotent**: existing files are reused,
so re-running only fills gaps. Formats: GeoParquet/Parquet (tables), GeoTIFF
(rasters), GeoJSON (vectors), JSON (metadata), YAML (config). CSV only as an
optional human-readable export.

## 12. Known limitations

* GEDI coverage is limited to ±51.6° latitude and starts 2019-04-18; `--validate`
  warns if your dates precede that.
* EE table/image exports have per-request size caps; very large regions are
  batched (GEDI) but still take time — scale up gradually after the milestone.
* S-2 exports use one scale (10 m) for all bands; B11/B12 are natively 20 m
  and are resampled by EE at export (bilinear). Recorded in metadata.
* S-1 in dB is standard for biomass work, but mixing orbit directions changes
  incidence-angle geometry — set `datasets.sentinel1.orbit_pass` for a
  homogeneous stack if your model is sensitive to it.
* The footprint buffer is applied in each raster's projected CRS (metres for
  UTM exports); for geographic-CRS rasters the approximation is coarse at
  high latitudes — export region should be modest in size (recommended anyway).
* Earth Engine quotas/rate limits apply; the pipeline raises meaningful errors
  (auth, empty query, corrupted export) instead of retry-storming silently.

## 13. Prediction API & Grid Precomputation
The production inference service uses a precomputed grid approach for fast, reliable, sub-millisecond API responses without per-request Earth Engine latency.

### Step 1: Precompute the Biomass Grid
Extracts the 12 features (Sentinel-2, Sentinel-1, NASADEM elevation/slope, and ESA WorldCover forest mask) across a ~250m regular grid over the study region, runs inference with the locked-in v4 Random Forest model (`models/agb_model_final_v4.joblib`), and saves results to `data/processed/agb_grid.parquet`:

```bash
python scripts/precompute_agb_grid.py
```

### Step 2: Launch the FastAPI Service
Loads the precomputed grid and spatial index once at startup:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation (Swagger UI) is available at:
`http://127.0.0.1:8000/docs`

### API Endpoints
- **`GET /health`**: Health check and metadata about the study region, loaded grid, forest cell count, mean forest AGB, and mean canopy density metrics.
- **`GET /agb?lat={lat}&lon={lon}`**: Nearest-neighbor lookup returning:
  - `agb_mg_ha`: Aboveground Biomass density (Mg/ha)
  - `carbon_tons_per_ha`: Forest carbon stock (AGB × 0.47)
  - `co2_equivalent_tons_per_ha`: Carbon dioxide equivalent in gas-mass equivalent (Carbon × 3.67)
  - `canopy_density_pct`: Resolution-appropriate canopy density percentage (NDVI proxy scaled to 0–100%)
  - `ndvi_canopy_proxy`: Sentinel-2 NDVI canopy density proxy
  - `grid_cell_lat`, `grid_cell_lon`: Coordinates of the matched grid cell center
  - `distance_to_cell_m`: Great-circle distance in metres to the matched cell center (mean ~86m, max ~166m across the 250m grid)
  - Returns `400` if outside the study region bounds or `422` with a detailed error payload if classified as non-forest.
- **`GET /agb/region?min_lat={min_lat}&min_lon={min_lon}&max_lat={max_lat}&max_lon={max_lon}`**: Bounding-box spatial query returning all grid cells (with `agb_mg_ha`, `carbon_tons_per_ha`, `co2_equivalent_tons_per_ha`, `canopy_density_pct`, and `ndvi_canopy_proxy`) for dashboard heatmap and raster map rendering. Filterable via `forest_only=true` (default).

## 14. Frontend handoff

The frontend can treat the FastAPI service as a read-only data source. Start
the backend locally, then use `http://127.0.0.1:8000` as the API base URL:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The service enables CORS for browser-based dashboard development. The grid
is loaded once at startup, so `data/processed/agb_grid.parquet` must exist
before the API starts. A fresh setup can generate it with:

```bash
python scripts/precompute_agb_grid.py
```

This generation step requires authenticated Earth Engine access. Once the
grid has been generated, the API itself serves local files and does not need
to query Earth Engine for each map interaction.

### Frontend endpoints

| Endpoint | Use | Successful response |
|---|---|---|
| `GET /health` | Backend status and summary statistics | JSON object with `status`, `study_region`, `total_cells`, `forest_cells`, and mean metrics |
| `GET /agb?lat=11.55&lon=76.55` | Detail panel for one coordinate | JSON object with AGB, carbon, CO2, canopy, matched grid coordinate, and distance |
| `GET /agb/region?min_lat=11.4&min_lon=76.4&max_lat=11.5&max_lon=76.5&forest_only=true` | Heatmap or map layer | JSON object with `count`, `bounds`, and `cells[]` |

Important response fields include:

* `agb_mg_ha`: above-ground biomass density in megagrams per hectare.
* `carbon_tons_per_ha`: estimated carbon stock using an AGB x 0.47 factor.
* `co2_equivalent_tons_per_ha`: carbon dioxide equivalent using a 3.67 factor.
* `canopy_density_pct`: NDVI-derived canopy density proxy from 0 to 100.
* `lat`, `lon`, `grid_cell_lat`, `grid_cell_lon`: WGS84 coordinates.

HTTP error behavior:

* `400`: coordinate or bounding-box input is invalid, or a point is outside
  the configured study-region bounds.
* `422`: the nearest grid cell is classified as non-forest and has no biomass
  estimate. The response includes the matched cell coordinates.
* `503`: the grid is not loaded; check the backend startup logs.

OpenAPI documentation is available at `http://127.0.0.1:8000/docs` after the
service starts. The frontend should use the documented response fields rather
than depending on internal model or raster files.

### Fresh-clone checklist

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For pipeline downloads or grid regeneration, copy `.env.example` to `.env`,
set `EE_PROJECT`, and authenticate with `earthengine authenticate`. Never
commit `.env`, service-account JSON files, or downloaded raw satellite data.
