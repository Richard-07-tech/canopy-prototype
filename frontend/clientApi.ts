export interface BiomassRecord {
  id: string;
  regionId: string;
  latitude: number;
  longitude: number;
  biomassMgHa: number;
  landCover?: string;
}

export interface PointEstimate {
  lat: number;
  lng: number;
  agb: number;
  carbonStock: number;
  co2e: number;
  landCover: string;
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`);
  if (!response.ok) throw new Error(`API request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export function fetchRegions<T>(): Promise<T[]> {
  return request<T[]>("/api/regions");
}

export function fetchBiomass(regionId: string): Promise<BiomassRecord[]> {
  return request<BiomassRecord[]>(`/api/regions/${encodeURIComponent(regionId)}/biomass`);
}

export function fetchPointEstimate(regionId: string, lat: number, lng: number): Promise<PointEstimate> {
  const params = new URLSearchParams({ regionId, lat: String(lat), lng: String(lng) });
  return request<PointEstimate>(`/api/point-estimates?${params.toString()}`);
}
