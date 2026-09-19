import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { MapContainer, TileLayer, useMapEvents, useMap } from "react-leaflet";
import L from "leaflet";
import {
  fetchHealth,
  fetchAgb,
  fetchRegionCells,
  NetworkError,
  OutsideStudyAreaError,
  NonForestError,
  BackendNotReadyError,
  type HealthResponse,
  type AgbResponse,
  type RegionCell,
  type StudyRegion,
} from "./clientApi";

// ─── Types ────────────────────────────────────────────────────────────────────

interface LayerState {
  satellite: boolean;
  biomass: boolean;
  studyBoundary: boolean;
}

type SelectionError =
  | { type: "outside"; message: string }
  | { type: "non-forest"; message: string; gridLat: number; gridLon: number; distanceM: number }
  | { type: "backend-not-ready"; message: string }
  | { type: "network"; message: string };

type BackendStatus = "loading" | "healthy" | "unreachable" | "not-ready";

// ─── Color Scale ──────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number) {
  return Math.round(a + (b - a) * t);
}

/**
 * Restrained environmental palette: slate blue → teal → amber → terracotta.
 */
function heatmapRGB(agb: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, (Number.isFinite(agb) ? agb : 0) / 150));
  const stops: [number, number, number][] = [[49, 91, 120], [77, 146, 161], [111, 159, 134], [196, 161, 91], [189, 103, 77]];
  const position = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(position));
  const mix = position - index;
  return [
    lerp(stops[index][0], stops[index + 1][0], mix),
    lerp(stops[index][1], stops[index + 1][1], mix),
    lerp(stops[index][2], stops[index + 1][2], mix),
  ];
}

// ─── Components ───────────────────────────────────────────────────────────────

function ColorScaleLegend() {
  return (
    <div className="biomass-legend">
      <p className="text-[10px] font-medium uppercase tracking-widest text-white/75 mb-2">
        Biomass density (Mg/ha)
      </p>
      <div
        className="h-3 rounded-full w-full mb-1.5"
        style={{
          background:
            "linear-gradient(to right, #315b78, #4d92a1, #6f9f86, #c4a15b, #bd674d)",
        }}
      />
      <div className="flex justify-between">
        <span className="text-[9px] text-white/60 font-mono">0</span>
        <span className="text-[9px] text-white/60 font-mono">75</span>
        <span className="text-[9px] text-white/60 font-mono">150+</span>
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  unit: string;
  sub?: string;
  highlight?: boolean;
}

function StatCard({ label, value, unit, sub, highlight }: StatCardProps) {
  return (
    <div
      className={`rounded-xl p-5 border transition-all ${
        highlight
          ? "bg-[var(--color-primary)] border-[var(--color-primary)] text-white"
          : "bg-white border-[var(--color-border)] text-[var(--color-foreground)]"
      }`}
    >
      <p
        className={`text-xs font-medium uppercase tracking-widest mb-3 ${
          highlight ? "text-white/70" : "text-[var(--color-muted-foreground)]"
        }`}
      >
        {label}
      </p>
      <div className="flex items-baseline gap-2">
        <span
          className="font-mono text-3xl font-medium leading-none"
          style={{ fontFamily: "var(--font-mono)" }}
        >
          {value}
        </span>
        <span
          className={`text-xs ${highlight ? "text-white/60" : "text-[var(--color-muted-foreground)]"}`}
        >
          {unit}
        </span>
      </div>
      {sub && (
        <p
          className={`text-xs mt-2 ${highlight ? "text-white/60" : "text-[var(--color-muted-foreground)]"}`}
        >
          {sub}
        </p>
      )}
    </div>
  );
}

interface LayerToggleProps {
  layers: LayerState;
  onChange: (key: keyof LayerState) => void;
}

function LayerToggleControl({ layers, onChange }: LayerToggleProps) {
  const items: { key: keyof LayerState; label: string; color: string }[] = [
    { key: "satellite", label: "Satellite", color: "#6B7280" },
    { key: "biomass", label: "Biomass Heatmap", color: "#6f9f86" },
    { key: "studyBoundary", label: "Study Area Zone", color: "#10B981" },
  ];
  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl p-3 shadow-lg border border-[var(--color-border)] w-48">
      <p className="text-[10px] font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-3">
        Layers
      </p>
      <div className="flex flex-col gap-2">
        {items.map(({ key, label, color }) => (
          <button
            key={key}
            onClick={() => onChange(key)}
            className="flex items-center justify-between group"
          >
            <div className="flex items-center gap-2">
              <div
                className="w-2.5 h-2.5 rounded-full transition-opacity"
                style={{
                  backgroundColor: color,
                  opacity: layers[key] ? 1 : 0.3,
                }}
              />
              <span
                className={`text-xs font-medium transition-colors ${
                  layers[key]
                    ? "text-[var(--color-foreground)]"
                    : "text-[var(--color-muted-foreground)]"
                }`}
              >
                {label}
              </span>
            </div>
            <div
              className={`w-8 h-4 rounded-full transition-colors relative ${
                layers[key] ? "bg-[var(--color-primary)]" : "bg-gray-200"
              }`}
            >
              <div
                className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow-sm transition-transform ${
                  layers[key] ? "translate-x-4" : "translate-x-0.5"
                }`}
              />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Biomass Heatmap Canvas Overlay ──────────────────────────────────────────

function BiomassHeatmapLayer({
  visible,
  cells,
}: {
  visible: boolean;
  cells: RegionCell[];
}) {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Precompute intensity from actual data range
  const renderData = useMemo(() => {
    if (!cells.length) return [];
    const values = cells
      .map((c) => c.agb_mg_ha)
      .filter((v): v is number => v !== null && Number.isFinite(v) && v >= 0);
    if (!values.length) return [];
    const sorted = [...values].sort((a, b) => a - b);
    const p02 = sorted[Math.floor(sorted.length * 0.02)] ?? 0;
    const p98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] ?? 0;
    const range = Math.max(p98 - p02, 1);

    return cells
      .filter((c) => c.agb_mg_ha !== null && Number.isFinite(c.agb_mg_ha))
      .map((c) => {
        const agb = Math.max(0, c.agb_mg_ha!);
        const intensity = Math.max(0, Math.min(1, (agb - p02) / range));
        return { lat: c.lat, lon: c.lon, agb, intensity };
      });
  }, [cells]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visible) return;
    const size = map.getSize();
    canvas.width = size.x;
    canvas.height = size.y;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, size.x, size.y);
    if (!renderData.length) return;

    // Grid resolution is ~250m → roughly 0.0022° at this latitude
    // Use a radius that creates slight overlap for visual continuity
    const cellRadius = 0.0015;

    // Pass 1: soft glow
    ctx.globalCompositeOperation = "lighter";
    for (const cell of renderData) {
      const center = map.latLngToContainerPoint([cell.lat, cell.lon]);
      const edge = map.latLngToContainerPoint([cell.lat, cell.lon + cellRadius * 2.5]);
      const rpx = Math.abs(edge.x - center.x);
      if (rpx < 1) continue;
      const [cr, cg, cb] = heatmapRGB(cell.agb);
      const glowAlpha = 0.08 + cell.intensity * 0.18;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, rpx);
      grad.addColorStop(0, `rgba(${cr},${cg},${cb},${glowAlpha.toFixed(3)})`);
      grad.addColorStop(0.5, `rgba(${cr},${cg},${cb},${(glowAlpha * 0.4).toFixed(3)})`);
      grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(center.x, center.y, rpx, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pass 2: sharp core cells, sorted low→high so hot cells render on top
    ctx.globalCompositeOperation = "source-over";
    const sorted = [...renderData].sort((a, b) => a.intensity - b.intensity);
    for (const cell of sorted) {
      const center = map.latLngToContainerPoint([cell.lat, cell.lon]);
      const edge = map.latLngToContainerPoint([cell.lat, cell.lon + cellRadius]);
      const rpx = Math.abs(edge.x - center.x);
      if (rpx < 1) continue;
      const [cr, cg, cb] = heatmapRGB(cell.agb);
      const coreAlpha = 0.3 + cell.intensity * 0.55;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, rpx);
      grad.addColorStop(0, `rgba(${cr},${cg},${cb},${coreAlpha.toFixed(3)})`);
      grad.addColorStop(0.5, `rgba(${cr},${cg},${cb},${(coreAlpha * 0.5).toFixed(3)})`);
      grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(center.x, center.y, rpx, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
  }, [map, visible, renderData]);

  useEffect(() => {
    if (!visible) {
      if (canvasRef.current) canvasRef.current.style.display = "none";
      return;
    }

    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.className = "biomass-canvas";
      canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:450;";
      const biomassPane = map.getPane("biomass-overlay") ?? map.createPane("biomass-overlay", map.getContainer());
      biomassPane.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:450;";
      biomassPane.appendChild(canvas);
      canvasRef.current = canvas;
    }
    canvas.style.display = "";

    draw();
    map.on("move zoom viewreset moveend zoomend resize", draw);
    return () => {
      map.off("move zoom viewreset moveend zoomend resize", draw);
    };
  }, [map, visible, draw]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      canvasRef.current?.remove();
      canvasRef.current = null;
    };
  }, []);

  return null;
}

// ─── Map Utilities ────────────────────────────────────────────────────────────

function MapSizeSync() {
  const map = useMap();

  useEffect(() => {
    const syncMapSize = () => map.invalidateSize({ pan: false, debounceMoveend: true });
    const observer = new ResizeObserver(syncMapSize);
    observer.observe(map.getContainer());
    syncMapSize();
    window.addEventListener("resize", syncMapSize);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", syncMapSize);
    };
  }, [map]);

  return null;
}

/** Fits the map to the study region bounds on mount. */
function FitStudyRegion({ bounds }: { bounds: StudyRegion }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(
      [
        [bounds.min_lat, bounds.min_lon],
        [bounds.max_lat, bounds.max_lon],
      ],
      { padding: [20, 20], maxZoom: 12 }
    );
  }, [map, bounds]);
  return null;
}

const DEFAULT_STUDY_REGION: StudyRegion = {
  min_lat: 11.20,
  max_lat: 11.90,
  min_lon: 76.20,
  max_lon: 76.90,
};

/**
 * Highlights the precomputed data coverage area with an emerald dashed boundary,
 * subtle illuminated tint, and corner boundary indicators.
 */
function StudyAreaHighlight({
  bounds,
  visible,
}: {
  bounds: StudyRegion;
  visible: boolean;
}) {
  const map = useMap();
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!visible) {
      if (layerGroupRef.current) {
        layerGroupRef.current.remove();
        layerGroupRef.current = null;
      }
      return;
    }

    if (layerGroupRef.current) {
      layerGroupRef.current.remove();
    }

    const group = L.layerGroup().addTo(map);
    layerGroupRef.current = group;

    const latLngBounds: L.LatLngBoundsExpression = [
      [bounds.min_lat, bounds.min_lon],
      [bounds.max_lat, bounds.max_lon],
    ];

    // Main highlight rectangle with glowing emerald dashed border and gentle tint
    const rect = L.rectangle(latLngBounds, {
      color: "#059669",
      weight: 2.5,
      dashArray: "10, 6",
      fillColor: "#10B981",
      fillOpacity: 0.08,
      interactive: true,
    });

    rect.bindTooltip(
      `<div style="font-family: var(--font-body, system-ui); padding: 4px 6px;">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
          <strong style="color: #34D399; font-size: 12px; letter-spacing: 0.5px;">ACTIVE STUDY REGION</strong>
        </div>
        <div style="font-size: 11px; color: #E2E8F0; font-weight: 500;">Nilgiri Biosphere Reserve</div>
        <div style="font-family: monospace; font-size: 10px; color: #94A3B8; margin-top: 2px;">
          ${bounds.min_lat.toFixed(2)}°–${bounds.max_lat.toFixed(2)}°N · ${bounds.min_lon.toFixed(2)}°–${bounds.max_lon.toFixed(2)}°E
        </div>
        <div style="font-size: 10px; color: #6EE7B7; margin-top: 4px;">
          ✓ Click inside for live AGB & carbon estimation
        </div>
      </div>`,
      { sticky: true, className: "study-area-tooltip", direction: "top" }
    );

    rect.addTo(group);

    // 4 Corner boundary pins for high-precision geospatial aesthetic
    const corners = [
      { lat: bounds.max_lat, lon: bounds.min_lon, label: `NW (${bounds.max_lat}°N, ${bounds.min_lon}°E)` },
      { lat: bounds.max_lat, lon: bounds.max_lon, label: `NE (${bounds.max_lat}°N, ${bounds.max_lon}°E)` },
      { lat: bounds.min_lat, lon: bounds.min_lon, label: `SW (${bounds.min_lat}°N, ${bounds.min_lon}°E)` },
      { lat: bounds.min_lat, lon: bounds.max_lon, label: `SE (${bounds.min_lat}°N, ${bounds.max_lon}°E)` },
    ];

    corners.forEach((c) => {
      const marker = L.circleMarker([c.lat, c.lon], {
        radius: 4,
        color: "#059669",
        weight: 2,
        fillColor: "#FFFFFF",
        fillOpacity: 1,
      });
      marker.bindTooltip(c.label, { permanent: false, direction: "auto", className: "corner-tooltip" });
      marker.addTo(group);
    });

    return () => {
      group.remove();
      layerGroupRef.current = null;
    };
  }, [map, bounds, visible]);

  return null;
}

/** Smoothly flies to the study area when triggered */
function MapFlyTo({
  trigger,
  bounds,
}: {
  trigger: number;
  bounds: StudyRegion | null;
}) {
  const map = useMap();
  useEffect(() => {
    if (trigger > 0 && bounds) {
      map.flyToBounds(
        [
          [bounds.min_lat, bounds.min_lon],
          [bounds.max_lat, bounds.max_lon],
        ],
        { padding: [40, 40], duration: 1.0, maxZoom: 11 }
      );
    }
  }, [map, trigger, bounds]);
  return null;
}

// ─── Map Click Handler ────────────────────────────────────────────────────────

function MapClickHandler({
  studyRegion,
  onResult,
  onError,
  onLoading,
}: {
  studyRegion: StudyRegion | null;
  onResult: (data: AgbResponse, clickLat: number, clickLon: number) => void;
  onError: (error: SelectionError) => void;
  onLoading: (loading: boolean) => void;
}) {
  useMapEvents({
    async click(e) {
      const { lat, lng: lon } = e.latlng;

      // Client-side bounds check
      if (studyRegion) {
        if (
          lat < studyRegion.min_lat ||
          lat > studyRegion.max_lat ||
          lon < studyRegion.min_lon ||
          lon > studyRegion.max_lon
        ) {
          onError({
            type: "outside",
            message: `Clicked point (${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E) is outside the study area (${studyRegion.min_lat}–${studyRegion.max_lat}°N, ${studyRegion.min_lon}–${studyRegion.max_lon}°E). No data available.`,
          });
          return;
        }
      }

      onLoading(true);
      try {
        const data = await fetchAgb(lat, lon);
        onResult(data, lat, lon);
      } catch (err) {
        if (err instanceof OutsideStudyAreaError) {
          onError({ type: "outside", message: err.message });
        } else if (err instanceof NonForestError) {
          onError({
            type: "non-forest",
            message: err.message,
            gridLat: err.gridCellLat,
            gridLon: err.gridCellLon,
            distanceM: err.distanceToCellM,
          });
        } else if (err instanceof BackendNotReadyError) {
          onError({ type: "backend-not-ready", message: err.message });
        } else if (err instanceof NetworkError) {
          onError({ type: "network", message: err.message });
        } else {
          onError({ type: "network", message: String(err) });
        }
      } finally {
        onLoading(false);
      }
    },
  });
  return null;
}

// ─── Selected Point Marker ────────────────────────────────────────────────────

function SelectedMarker({ lat, lon }: { lat: number | null; lon: number | null }) {
  const map = useMap();
  const markerRef = useRef<L.Marker | null>(null);

  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }
    if (lat === null || lon === null) return;

    const icon = L.divIcon({
      html: `<div style="
        width:16px;height:16px;border-radius:50%;
        background:#6f9f86;border:3px solid white;
        box-shadow:0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
      className: "",
    });

    markerRef.current = L.marker([lat, lon], { icon }).addTo(map);
    return () => {
      markerRef.current?.remove();
    };
  }, [map, lat, lon]);

  return null;
}

// ─── Region Data Loader (fetches /agb/region on viewport change) ──────────────

function RegionDataLoader({
  enabled,
  onData,
}: {
  enabled: boolean;
  onData: (cells: RegionCell[]) => void;
}) {
  const map = useMap();
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadViewport = useCallback(() => {
    if (!enabled) return;
    abortRef.current?.abort();
    const bounds = map.getBounds();
    const viewport = {
      minLat: bounds.getSouth(),
      minLon: bounds.getWest(),
      maxLat: bounds.getNorth(),
      maxLon: bounds.getEast(),
    };
    fetchRegionCells(viewport, true)
      .then((res) => onData(res.cells))
      .catch(() => { /* silently ignore — heatmap just won't update */ });
  }, [map, enabled, onData]);

  // Debounced load on viewport change
  const debouncedLoad = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(loadViewport, 400);
  }, [loadViewport]);

  useEffect(() => {
    if (!enabled) return;
    loadViewport(); // initial load
    map.on("moveend zoomend", debouncedLoad);
    return () => {
      map.off("moveend zoomend", debouncedLoad);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [map, enabled, loadViewport, debouncedLoad]);

  return null;
}

// ─── Top Bar ──────────────────────────────────────────────────────────────────

function TopBar({
  health,
  status,
  onFocusRegion,
}: {
  health: HealthResponse | null;
  status: BackendStatus;
  onFocusRegion?: () => void;
}) {
  return (
    <header className="h-14 bg-white border-b border-[var(--color-border)] flex items-center px-5 gap-4 shrink-0 z-20">
      {/* Logo */}
      <div className="flex items-center gap-2 mr-2">
        <div className="w-7 h-7 rounded-lg bg-[var(--color-primary)] flex items-center justify-center">
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2"
          >
            <path d="M17 8C8 10 5.9 16.17 3.82 22" />
            <path d="M9.09 9.91C7.17 12.58 5.56 14.16 2 16" />
            <path d="M12 2a4 4 0 0 0 4 4c0-2.21-1.79-4-4-4z" />
          </svg>
        </div>
        <span
          className="font-display text-lg font-semibold text-[var(--color-foreground)] tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Canopy
        </span>
      </div>

      <div className="w-px h-6 bg-[var(--color-border)]" />

      {/* Region label with Active Study Area badge */}
      <button
        onClick={onFocusRegion}
        className="flex items-center gap-2 px-2.5 py-1.5 -my-1 rounded-lg hover:bg-slate-100/80 transition-colors text-left cursor-pointer group"
        title="Click to center map on study area"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="text-emerald-600 group-hover:scale-110 transition-transform"
        >
          <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
          <circle cx="12" cy="10" r="3" />
        </svg>
        <span className="text-sm font-medium text-[var(--color-foreground)]">
          Nilgiri Biosphere Reserve
        </span>
        <span className="text-[10px] font-semibold px-2 py-0.5 bg-emerald-50 text-emerald-800 rounded-full border border-emerald-200 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          Active Study Area
        </span>
      </button>

      {health && (
        <div className="flex items-center gap-1.5 text-xs text-[var(--color-muted-foreground)] ml-1">
          <span>·</span>
          <span>{health.model_version}</span>
          <span>·</span>
          <span>{health.grid_resolution_m}m grid</span>
        </div>
      )}

      <div className="ml-auto flex items-center gap-2">
        {/* Backend status indicator */}
        <div
          className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full ${
            status === "healthy"
              ? "bg-[var(--color-status-bg)] text-[var(--color-status)]"
              : status === "loading"
              ? "bg-[var(--color-muted)] text-[var(--color-muted-foreground)]"
              : "bg-red-50 text-red-600"
          }`}
        >
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              status === "healthy"
                ? "bg-[var(--color-accent)] animate-pulse"
                : status === "loading"
                ? "bg-[var(--color-muted-foreground)] animate-pulse"
                : "bg-red-400"
            }`}
          />
          {status === "healthy"
            ? "Backend Connected"
            : status === "loading"
            ? "Connecting…"
            : status === "unreachable"
            ? "Backend Unreachable"
            : "Backend Not Ready"}
        </div>
      </div>
    </header>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar({
  health,
  selected,
  clickLat,
  clickLon,
  selectionError,
  onClearSelection,
  pointLoading,
  backendStatus,
  onFocusRegion,
}: {
  health: HealthResponse | null;
  selected: AgbResponse | null;
  clickLat: number | null;
  clickLon: number | null;
  selectionError: SelectionError | null;
  onClearSelection: () => void;
  pointLoading: boolean;
  backendStatus: BackendStatus;
  onFocusRegion?: () => void;
}) {
  const hasSelection = selected !== null || selectionError !== null;

  return (
    <aside className="w-[360px] min-w-[320px] bg-[var(--color-background)] border-l border-[var(--color-border)] flex flex-col overflow-hidden shrink-0">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-[var(--color-border)]">
        {hasSelection ? (
          <div>
            <button
              onClick={onClearSelection}
              className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)] hover:text-[var(--color-primary)] transition-colors mb-2"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to region overview
            </button>
            <h2
              className="text-base font-semibold"
              style={{ fontFamily: "var(--font-display)" }}
            >
              {selected ? "Point Detail" : "No Data"}
            </h2>
            {clickLat !== null && clickLon !== null && (
              <p className="text-xs text-[var(--color-muted-foreground)] mt-0.5 font-mono">
                Clicked: {clickLat.toFixed(5)}°N, {clickLon.toFixed(5)}°E
              </p>
            )}
          </div>
        ) : (
          <div>
            <h2
              className="text-base font-semibold"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Region Overview
            </h2>
            <p className="text-xs text-[var(--color-muted-foreground)] mt-0.5">
              Nilgiri Biosphere Reserve · Western Ghats
            </p>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-3">
        {/* Global error states */}
        {backendStatus === "unreachable" && !hasSelection && (
          <div className="rounded-xl p-5 border border-red-200 bg-red-50">
            <div className="flex gap-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#DC2626" strokeWidth="2" className="shrink-0 mt-0.5">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <div>
                <p className="text-sm font-medium text-red-800">Backend Unreachable</p>
                <p className="text-xs text-red-600 mt-1">
                  Cannot reach the API server. Please ensure the FastAPI backend is running.
                </p>
              </div>
            </div>
          </div>
        )}

        {backendStatus === "not-ready" && !hasSelection && (
          <div className="rounded-xl p-5 border border-amber-200 bg-amber-50">
            <div className="flex gap-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2" className="shrink-0 mt-0.5">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <div>
                <p className="text-sm font-medium text-amber-800">Backend Data Not Ready</p>
                <p className="text-xs text-amber-600 mt-1">
                  The precomputed grid is still loading. Please wait and try again.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Point loading skeleton */}
        {pointLoading && (
          <div className="flex flex-col gap-3">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="rounded-xl p-5 border border-[var(--color-border)] bg-white animate-pulse"
              >
                <div className="h-2.5 bg-[var(--color-muted)] rounded w-24 mb-4" />
                <div className="h-8 bg-[var(--color-muted)] rounded w-32" />
              </div>
            ))}
          </div>
        )}

        {/* Selection error states */}
        {!pointLoading && selectionError?.type === "outside" && (
          <div className="rounded-xl p-5 border border-amber-200 bg-amber-50/90 shadow-2xs">
            <div className="flex gap-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2" className="shrink-0 mt-0.5">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                <line x1="9" y1="7" x2="15" y2="13" />
                <line x1="15" y1="7" x2="9" y2="13" />
              </svg>
              <div className="flex-1">
                <p className="text-sm font-semibold text-amber-900">Outside Study Area</p>
                <p className="text-xs text-amber-800 mt-1.5 leading-relaxed">
                  {selectionError.message}
                </p>
                <div className="mt-3 p-2.5 rounded-lg bg-white/80 border border-amber-200/80 text-xs text-amber-900 leading-relaxed">
                  💡 Model predictions are only available within the <strong>green dashed Nilgiri study region</strong>.
                </div>
                {onFocusRegion && (
                  <button
                    onClick={onFocusRegion}
                    className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-800 rounded-lg shadow-sm transition-colors cursor-pointer"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
                    </svg>
                    Focus Study Area
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {!pointLoading && selectionError?.type === "non-forest" && (
          <div className="rounded-xl p-5 border border-amber-200 bg-amber-50/80">
            <div className="flex gap-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#92704C" strokeWidth="2" className="shrink-0 mt-0.5">
                <path d="M17 8C8 10 5.9 16.17 3.82 22" />
                <path d="M9.09 9.91C7.17 12.58 5.56 14.16 2 16" />
                <line x1="4" y1="4" x2="20" y2="20" />
              </svg>
              <div>
                <p className="text-sm font-semibold text-amber-900">Non-Forest Cell</p>
                <p className="text-xs text-amber-700 mt-1.5 leading-relaxed">
                  {selectionError.message}
                </p>
                <div className="mt-3 flex gap-4">
                  <div>
                    <p className="text-[10px] uppercase tracking-widest text-amber-600">Grid Cell</p>
                    <p className="text-xs font-mono text-amber-800 mt-0.5">
                      {selectionError.gridLat.toFixed(5)}°N, {selectionError.gridLon.toFixed(5)}°E
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-widest text-amber-600">Distance</p>
                    <p className="text-xs font-mono text-amber-800 mt-0.5">
                      {selectionError.distanceM.toFixed(0)} m
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!pointLoading && selectionError?.type === "backend-not-ready" && (
          <div className="rounded-xl p-5 border border-amber-200 bg-amber-50">
            <p className="text-sm font-medium text-amber-800">Backend Data Not Ready</p>
            <p className="text-xs text-amber-600 mt-1">{selectionError.message}</p>
          </div>
        )}

        {!pointLoading && selectionError?.type === "network" && (
          <div className="rounded-xl p-5 border border-red-200 bg-red-50">
            <p className="text-sm font-medium text-red-800">Connection Error</p>
            <p className="text-xs text-red-600 mt-1">{selectionError.message}</p>
          </div>
        )}

        {/* Successful point selection */}
        {!pointLoading && selected && (
          <>
            <StatCard
              label="AGB"
              value={selected.agb_mg_ha.toFixed(2)}
              unit="Mg/ha"
              sub="Above-Ground Biomass"
              highlight={false}
            />
            <StatCard
              label="Carbon Stock"
              value={selected.carbon_tons_per_ha.toFixed(2)}
              unit="tons C/ha"
              sub="AGB × 0.47 (server-computed)"
            />
            <StatCard
              label="CO₂ Equivalent"
              value={selected.co2_equivalent_tons_per_ha.toFixed(2)}
              unit="tons CO₂e/ha"
              sub="Carbon × 3.67 (server-computed)"
              highlight={true}
            />

            {/* Canopy & NDVI */}
            <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-2">
                Canopy Density
              </p>
              {selected.canopy_density_pct !== null ? (
                <>
                  <div className="flex items-baseline gap-2 mb-3">
                    <span className="font-mono text-2xl font-medium" style={{ fontFamily: "var(--font-mono)" }}>
                      {selected.canopy_density_pct.toFixed(1)}
                    </span>
                    <span className="text-xs text-[var(--color-muted-foreground)]">%</span>
                  </div>
                  <div className="h-2 rounded-full bg-[var(--color-muted)] overflow-hidden mb-1.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-[var(--color-chart-low)] via-[var(--color-chart-mid)] to-[var(--color-primary)]"
                      style={{ width: `${Math.min(selected.canopy_density_pct, 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-[var(--color-muted-foreground)]">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </>
              ) : (
                <span className="text-sm text-[var(--color-muted-foreground)]">—</span>
              )}
            </div>

            {selected.ndvi_canopy_proxy !== null && (
              <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
                <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-2">
                  NDVI Canopy Proxy
                </p>
                <span className="font-mono text-2xl font-medium" style={{ fontFamily: "var(--font-mono)" }}>
                  {selected.ndvi_canopy_proxy.toFixed(4)}
                </span>
              </div>
            )}

            {/* Grid cell metadata */}
            <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-3">
                Matched Grid Cell
              </p>
              <div className="flex gap-4">
                <div>
                  <p className="text-[10px] uppercase tracking-widest text-[var(--color-muted-foreground)]">Cell Center</p>
                  <p className="text-xs font-mono mt-0.5">
                    {selected.grid_cell_lat.toFixed(5)}°N, {selected.grid_cell_lon.toFixed(5)}°E
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-widest text-[var(--color-muted-foreground)]">Distance</p>
                  <p className="text-xs font-mono mt-0.5">{selected.distance_to_cell_m.toFixed(0)} m</p>
                </div>
              </div>
            </div>

            {/* AGB percentile gauge */}
            <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-3">
                AGB Relative Position
              </p>
              <div className="h-2 rounded-full bg-[var(--color-muted)] overflow-hidden mb-1.5">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[var(--color-chart-low)] via-[var(--color-chart-mid)] to-[var(--color-primary)]"
                  style={{ width: `${Math.min((selected.agb_mg_ha / 250) * 100, 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-[var(--color-muted-foreground)]">
                <span>0 Mg/ha</span>
                <span className="text-[var(--color-primary)] font-medium">
                  {selected.agb_mg_ha.toFixed(1)} Mg/ha
                </span>
                <span>250+</span>
              </div>
            </div>
          </>
        )}

        {/* Region overview (no selection) */}
        {!pointLoading && !hasSelection && health && (
          <>
            {/* Active Study Area Coverage Card */}
            <div className="rounded-xl p-4 border border-emerald-200 bg-emerald-50/70">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
                  <span className="text-[11px] font-bold text-emerald-950 uppercase tracking-wider">
                    Active Study Area
                  </span>
                </div>
                {onFocusRegion && (
                  <button
                    onClick={onFocusRegion}
                    className="text-[11px] font-semibold text-emerald-700 hover:text-emerald-900 flex items-center gap-1 cursor-pointer transition-colors"
                  >
                    Center Map
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                  </button>
                )}
              </div>
              <p className="text-xs text-emerald-900/80 leading-relaxed mb-3">
                Predictions are trained on high-resolution Sentinel, GEDI, and NASADEM data covering the <strong>Nilgiri Biosphere Reserve</strong>.
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-white/90 rounded-lg p-2.5 border border-emerald-100 shadow-2xs">
                  <span className="text-[10px] uppercase tracking-wider text-emerald-700 block font-semibold">Latitude</span>
                  <span className="font-mono font-medium text-emerald-950 text-xs">11.20°N – 11.90°N</span>
                </div>
                <div className="bg-white/90 rounded-lg p-2.5 border border-emerald-100 shadow-2xs">
                  <span className="text-[10px] uppercase tracking-wider text-emerald-700 block font-semibold">Longitude</span>
                  <span className="font-mono font-medium text-emerald-950 text-xs">76.20°E – 76.90°E</span>
                </div>
              </div>
            </div>

            <StatCard
              label="Mean Forest AGB"
              value={health.mean_forest_agb_mg_ha.toFixed(1)}
              unit="Mg/ha"
              sub="Average across all forest cells"
              highlight={true}
            />
            {health.mean_forest_canopy_density_pct !== null && (
              <StatCard
                label="Mean Canopy Density"
                value={health.mean_forest_canopy_density_pct.toFixed(1)}
                unit="%"
                sub="NDVI-derived canopy proxy"
              />
            )}
            <StatCard
              label="Forest Cells"
              value={health.forest_cells.toLocaleString()}
              unit="cells"
              sub={`of ${health.total_cells.toLocaleString()} total grid cells`}
            />
            <StatCard
              label="Grid Resolution"
              value={String(health.grid_resolution_m)}
              unit="meters"
              sub="Precomputed cell spacing"
            />

            {/* Tip */}
            <div className="rounded-xl p-4 border border-dashed border-[var(--color-border)] bg-[var(--color-muted)]/50">
              <div className="flex gap-2.5">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-[var(--color-muted-foreground)] mt-0.5 shrink-0"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <p className="text-xs text-[var(--color-muted-foreground)] leading-relaxed">
                  Click any point on the map to view local biomass, carbon stock,
                  and canopy density from the precomputed grid.
                </p>
              </div>
            </div>
          </>
        )}

        {/* No health data and no error — still loading */}
        {!pointLoading && !hasSelection && !health && backendStatus === "loading" && (
          <div className="flex flex-col gap-3">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="rounded-xl p-5 border border-[var(--color-border)] bg-white animate-pulse"
              >
                <div className="h-2.5 bg-[var(--color-muted)] rounded w-24 mb-4" />
                <div className="h-8 bg-[var(--color-muted)] rounded w-32" />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-[var(--color-border)] flex items-center justify-between">
        <span className="text-[10px] text-[var(--color-muted-foreground)]">
          Data: Sentinel-2 · Sentinel-1 · GEDI L4A
        </span>
        <span className="text-[10px] text-[var(--color-muted-foreground)]">
          {health ? `Model: ${health.model_version}` : ""}
        </span>
      </div>
    </aside>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

// Default center for the study region (used before health data loads)
const DEFAULT_CENTER: [number, number] = [11.55, 76.55];
const DEFAULT_ZOOM = 10;

export default function App() {
  // Backend state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("loading");

  // Map & selection state
  const [selected, setSelected] = useState<AgbResponse | null>(null);
  const [clickLat, setClickLat] = useState<number | null>(null);
  const [clickLon, setClickLon] = useState<number | null>(null);
  const [selectionError, setSelectionError] = useState<SelectionError | null>(null);
  const [pointLoading, setPointLoading] = useState(false);
  const [regionCells, setRegionCells] = useState<RegionCell[]>([]);

  const [layers, setLayers] = useState<LayerState>({
    satellite: true,
    biomass: true,
    studyBoundary: true,
  });

  const [focusTrigger, setFocusTrigger] = useState(0);

  const handleFocusRegion = useCallback(() => {
    setFocusTrigger((c) => c + 1);
  }, []);

  // Fetch health on mount
  useEffect(() => {
    let active = true;
    setBackendStatus("loading");
    fetchHealth()
      .then((data) => {
        if (!active) return;
        setHealth(data);
        setBackendStatus("healthy");
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof NetworkError) {
          setBackendStatus("unreachable");
        } else if (err instanceof BackendNotReadyError) {
          setBackendStatus("not-ready");
        } else {
          setBackendStatus("unreachable");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const handleResult = useCallback((data: AgbResponse, lat: number, lon: number) => {
    setSelected(data);
    setClickLat(lat);
    setClickLon(lon);
    setSelectionError(null);
  }, []);

  const handleError = useCallback((error: SelectionError) => {
    setSelected(null);
    setSelectionError(error);
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelected(null);
    setClickLat(null);
    setClickLon(null);
    setSelectionError(null);
  }, []);

  const handleLayerToggle = useCallback((key: keyof LayerState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleRegionData = useCallback((cells: RegionCell[]) => {
    setRegionCells(cells);
  }, []);

  const studyRegion = health?.study_region ?? null;
  const activeRegionBounds = studyRegion ?? DEFAULT_STUDY_REGION;

  return (
    <div
      className="h-screen flex flex-col overflow-hidden"
      style={{ background: "var(--color-background)" }}
    >
      <TopBar health={health} status={backendStatus} onFocusRegion={handleFocusRegion} />

      <div className="dashboard-content flex flex-1 overflow-hidden">
        {/* Map panel */}
        <div className="map-panel flex-1 relative">
          <MapContainer
            center={DEFAULT_CENTER}
            zoom={DEFAULT_ZOOM}
            className="w-full h-full"
            zoomControl={true}
          >
            <MapSizeSync />
            {studyRegion && <FitStudyRegion bounds={studyRegion} />}
            <StudyAreaHighlight
              bounds={activeRegionBounds}
              visible={layers.studyBoundary}
            />
            <MapFlyTo
              trigger={focusTrigger}
              bounds={activeRegionBounds}
            />

            {/* Base tile */}
            {layers.satellite ? (
              <TileLayer
                url="https://{s}.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}"
                maxZoom={20}
                subdomains={["mt0", "mt1", "mt2", "mt3"]}
                attribution='&copy; Google'
              />
            ) : (
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; OpenStreetMap contributors'
              />
            )}

            <BiomassHeatmapLayer
              visible={layers.biomass}
              cells={regionCells}
            />
            <RegionDataLoader
              enabled={layers.biomass && backendStatus === "healthy"}
              onData={handleRegionData}
            />
            <MapClickHandler
              studyRegion={studyRegion}
              onResult={handleResult}
              onError={handleError}
              onLoading={setPointLoading}
            />
            <SelectedMarker
              lat={selected ? selected.grid_cell_lat : null}
              lon={selected ? selected.grid_cell_lon : null}
            />
          </MapContainer>

          {/* Floating Study Area Banner — top left (beside Leaflet zoom controls) */}
          <div className="absolute top-3 left-14 z-[10] flex items-center gap-2.5 bg-white/95 backdrop-blur-md px-3.5 py-2 rounded-xl shadow-lg border border-emerald-200/90 pointer-events-auto">
            <div className="relative flex items-center justify-center">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="absolute w-4 h-4 rounded-full bg-emerald-400/40 animate-ping" />
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-semibold text-slate-800">
                  Nilgiri Study Area
                </span>
                <span className="text-[9px] font-bold px-1.5 py-0.2 bg-emerald-100 text-emerald-800 rounded">
                  DATA READY
                </span>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">
                11.20°–11.90°N · 76.20°–76.90°E
              </span>
            </div>
            <button
              onClick={handleFocusRegion}
              className="ml-1 px-2.5 py-1 text-xs font-medium text-emerald-800 bg-emerald-50 hover:bg-emerald-100 rounded-lg border border-emerald-200 transition-colors flex items-center gap-1.5 cursor-pointer shadow-2xs active:scale-95"
              title="Center and fit map to active study area"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
              Focus
            </button>
          </div>

          {/* Layer toggle — top right, above all Leaflet panes */}
          <div className="map-controls-overlay">
            <LayerToggleControl layers={layers} onChange={handleLayerToggle} />
          </div>

          {/* Legend — bottom left */}
          {layers.biomass && (
            <div className="absolute bottom-6 left-3 z-[1000]">
              <ColorScaleLegend />
            </div>
          )}

          {/* Attribution */}
          <div className="absolute bottom-1 right-24 z-[1000]">
            <span className="text-[10px] text-white/70 bg-black/20 px-1.5 py-0.5 rounded backdrop-blur-sm">
              ESA Sentinel-2 · GEDI L4A · NASADEM
            </span>
          </div>
        </div>

        {/* Sidebar */}
        <Sidebar
          health={health}
          selected={selected}
          clickLat={clickLat}
          clickLon={clickLon}
          selectionError={selectionError}
          onClearSelection={handleClearSelection}
          pointLoading={pointLoading}
          backendStatus={backendStatus}
          onFocusRegion={handleFocusRegion}
        />
      </div>
    </div>
  );
}
