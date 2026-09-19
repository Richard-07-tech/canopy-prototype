/**
 * Client API for the FastAPI Biomass backend.
 *
 * All requests go through the single VITE_API_BASE_URL env var.
 * Error handling distinguishes four failure modes:
 *   - OutsideStudyAreaError  (HTTP 400)
 *   - NonForestError         (HTTP 422)
 *   - BackendNotReadyError   (HTTP 503)
 *   - NetworkError           (fetch itself failed)
 */

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL = (RAW_BASE_URL ? String(RAW_BASE_URL).trim().replace(/\/+$/, "") : "") || "http://127.0.0.1:8000";

// ─── Error Types ──────────────────────────────────────────────────────────────

export class NetworkError extends Error {
  constructor() {
    super(`Cannot reach backend at ${API_BASE_URL || "(no URL configured)"} — is the server running?`);
    this.name = "NetworkError";
  }
}

export class OutsideStudyAreaError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = "OutsideStudyAreaError";
  }
}

export class NonForestError extends Error {
  gridCellLat: number;
  gridCellLon: number;
  distanceToCellM: number;

  constructor(detail: { message: string; grid_cell_lat: number; grid_cell_lon: number; distance_to_cell_m: number }) {
    super(detail.message);
    this.name = "NonForestError";
    this.gridCellLat = detail.grid_cell_lat;
    this.gridCellLon = detail.grid_cell_lon;
    this.distanceToCellM = detail.distance_to_cell_m;
  }
}

export class BackendNotReadyError extends Error {
  constructor(detail: string) {
    super(detail || "Backend data not ready — the grid may still be loading.");
    this.name = "BackendNotReadyError";
  }
}

// ─── Response Types ───────────────────────────────────────────────────────────

export interface StudyRegion {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  study_region: StudyRegion;
  grid_resolution_m: number;
  total_cells: number;
  forest_cells: number;
  mean_forest_agb_mg_ha: number;
  mean_forest_canopy_density_pct: number | null;
  mean_forest_ndvi_canopy_proxy: number | null;
}

export interface AgbResponse {
  agb_mg_ha: number;
  carbon_tons_per_ha: number;
  co2_equivalent_tons_per_ha: number;
  canopy_density_pct: number | null;
  ndvi_canopy_proxy: number | null;
  grid_cell_lat: number;
  grid_cell_lon: number;
  distance_to_cell_m: number;
}

export interface RegionCell {
  lat: number;
  lon: number;
  agb_mg_ha: number | null;
  carbon_tons_per_ha: number | null;
  co2_equivalent_tons_per_ha: number | null;
  canopy_density_pct: number | null;
  ndvi_canopy_proxy: number | null;
  is_forest: boolean;
}

export interface RegionResponse {
  count: number;
  bounds: StudyRegion;
  cells: RegionCell[];
}

// ─── Fetch Helpers ────────────────────────────────────────────────────────────

async function handleResponse<T>(_url: string, res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>;

  // Try to extract JSON detail from error responses
  let body: any;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (res.status === 400) {
    const detail = typeof body?.detail === "string" ? body.detail : "Coordinate is outside the study region.";
    throw new OutsideStudyAreaError(detail);
  }

  if (res.status === 422) {
    const detail = body?.detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      throw new NonForestError(detail);
    }
    throw new OutsideStudyAreaError(typeof detail === "string" ? detail : "Unprocessable request.");
  }

  if (res.status === 503) {
    const detail = typeof body?.detail === "string" ? body.detail : "";
    throw new BackendNotReadyError(detail);
  }

  throw new Error(`API error ${res.status}: ${typeof body?.detail === "string" ? body.detail : res.statusText}`);
}

async function apiGet<T>(path: string): Promise<T> {
  const fullUrl = `${API_BASE_URL}${path}`;
  let res: Response;
  try {
    res = await fetch(fullUrl);
  } catch {
    throw new NetworkError();
  }
  return handleResponse<T>(fullUrl, res);
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function fetchHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health");
}

export function fetchAgb(lat: number, lon: number): Promise<AgbResponse> {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  return apiGet<AgbResponse>(`/agb?${params}`);
}

export interface ViewportBounds {
  minLat: number;
  minLon: number;
  maxLat: number;
  maxLon: number;
}

export function fetchRegionCells(bounds: ViewportBounds, forestOnly = true): Promise<RegionResponse> {
  const params = new URLSearchParams({
    min_lat: String(bounds.minLat),
    min_lon: String(bounds.minLon),
    max_lat: String(bounds.maxLat),
    max_lon: String(bounds.maxLon),
    forest_only: String(forestOnly),
  });
  return apiGet<RegionResponse>(`/agb/region?${params}`);
}
