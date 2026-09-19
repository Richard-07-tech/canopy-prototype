import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { MapContainer, TileLayer, useMapEvents, useMap } from "react-leaflet";
import L from "leaflet";
import { fetchBiomass, fetchPointEstimate, fetchRegions, type BiomassRecord } from "./clientApi";
// L is used for Marker/DivIcon in SelectedMarker

// ─── Types ────────────────────────────────────────────────────────────────────

interface Region {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
  area: number;
  totalCO2e: number;
  meanAGB: number;
  samplePoints: number;
  dateRange: string;
}

interface SelectedPoint {
  lat: number;
  lng: number;
  agb: number;
  carbonStock: number;
  co2e: number;
  landCover: string;
}

interface LayerState {
  satellite: boolean;
  biomass: boolean;
  landCover: boolean;
}

// ─── Mock Data ────────────────────────────────────────────────────────────────

const REGIONS: Region[] = [
  {
    id: "nilgiri",
    name: "Nilgiri Biosphere Reserve",
    center: [11.4, 76.6],
    zoom: 9,
    area: 12400,
    totalCO2e: 2310000,
    meanAGB: 93.4,
    samplePoints: 1326,
    dateRange: "Jan 2022 – Dec 2023",
  },
  {
    id: "western-ghats-north",
    name: "Western Ghats North",
    center: [15.1, 74.4],
    zoom: 8,
    area: 18750,
    totalCO2e: 3840000,
    meanAGB: 88.2,
    samplePoints: 2104,
    dateRange: "Jan 2022 – Dec 2023",
  },
  {
    id: "agasthyamalai",
    name: "Agasthyamalai Hills",
    center: [8.6, 77.2],
    zoom: 9,
    area: 7200,
    totalCO2e: 1190000,
    meanAGB: 79.6,
    samplePoints: 894,
    dateRange: "Mar 2022 – Dec 2023",
  },
];

const LAND_COVER_TYPES = [
  "Tropical Evergreen",
  "Semi-Evergreen",
  "Mixed Deciduous",
  "Shola Forest",
  "Grassland-Forest Mosaic",
];

// ─── Color Scale ──────────────────────────────────────────────────────────────


function lerp(a: number, b: number, t: number) {
  return Math.round(a + (b - a) * t);
}

// ─── Components ───────────────────────────────────────────────────────────────

function ColorScaleLegend() {
  return (
    <div className="biomass-legend">
      <p className="text-[10px] font-medium uppercase tracking-widest text-white/75 mb-2">
        Biomass availability
      </p>
      <div
        className="h-3 rounded-full w-full mb-1.5"
        style={{
          background:
            "linear-gradient(to right, #315b78, #4d92a1, #6f9f86, #c4a15b, #bd674d)",
        }}
      />
      <div className="flex justify-between">
        <span className="text-[9px] text-white/60 font-mono">Low</span>
        <span className="text-[9px] text-white/60 font-mono">High</span>
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
    { key: "biomass", label: "Biomass", color: "#6f9f86" },
    { key: "landCover", label: "Land Cover", color: "#92704C" },
  ];
  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl p-3 shadow-lg border border-[var(--color-border)] w-44">
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

interface RegionDropdownProps {
  regions: Region[];
  selected: Region;
  onChange: (r: Region) => void;
}

function RegionDropdown({ regions, selected, onChange }: RegionDropdownProps) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 bg-white border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm font-medium text-[var(--color-foreground)] hover:bg-[var(--color-muted)] transition-colors min-w-[220px] justify-between"
      >
        <div className="flex items-center gap-2">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
            <circle cx="12" cy="10" r="3" />
          </svg>
          <span>{selected.name}</span>
        </div>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && (
        <div className="absolute top-full mt-1 left-0 bg-white border border-[var(--color-border)] rounded-lg shadow-xl z-50 py-1 min-w-full">
          {regions.map((r) => (
            <button
              key={r.id}
              onClick={() => {
                onChange(r);
                setOpen(false);
              }}
              className={`w-full text-left px-4 py-2.5 text-sm transition-colors ${
                r.id === selected.id
                  ? "bg-[var(--color-muted)] text-[var(--color-primary)] font-medium"
                  : "text-[var(--color-foreground)] hover:bg-[var(--color-muted)]"
              }`}
            >
              <div className="font-medium">{r.name}</div>
              <div className="text-xs text-[var(--color-muted-foreground)]">
                {r.area.toLocaleString()} ha · {r.samplePoints.toLocaleString()} points
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Biomass blob anchors (lat/lng) seeded per region ────────────────────────

type BlobEntry = {
  lat: number; lng: number; agb: number;
  rx: number; ry: number; angle: number;
  intensity: number;
};

function isValidCoordinate(lat: number, lng: number) {
  return Number.isFinite(lat) && Number.isFinite(lng) &&
    lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
}

function normalizeBiomass(value: unknown, min: number, max: number) {
  const biomass = typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 0;
  if (max <= min) return biomass > 0 ? 0.65 : 0;
  return Math.max(0, Math.min(1, (biomass - min) / (max - min)));
}

function generateBlobs(region: Region, records: BiomassRecord[] = []): BlobEntry[] {
  const validRecords = records.filter((record) =>
    isValidCoordinate(record.latitude, record.longitude) &&
    Number.isFinite(record.biomassMgHa),
  );
  if (validRecords.length) {
    const values = validRecords.map((record) => Math.max(0, record.biomassMgHa));
    const sortedValues = [...values].sort((a, b) => a - b);
    const percentile = (position: number) =>
      sortedValues[Math.min(sortedValues.length - 1, Math.floor(sortedValues.length * position))] ?? 0;
    const lowerBound = percentile(0.02);
    const upperBound = percentile(0.98);

    return validRecords.map((record, index) => {
      const agb = Math.max(0, record.biomassMgHa);
      const intensity = normalizeBiomass(agb, lowerBound, Math.max(lowerBound, upperBound));
      const radius = 0.06 + intensity * 0.14;
      return {
        lat: record.latitude,
        lng: record.longitude,
        agb,
        rx: radius,
        ry: radius * 0.78,
        angle: (index % 7) * Math.PI / 7,
        intensity,
      };
    });
  }

  const seed = region.id.charCodeAt(0) * 13 + region.id.charCodeAt(region.id.length - 1) * 7;
  const [clat, clng] = region.center;
  const spread = 0.65;
  const blobs: BlobEntry[] = [];
  // Dense background blobs
  for (let i = 0; i < 90; i++) {
    const agb = 20 + ((Math.sin(seed + i * 7.1) + 1) / 2) * 130;
    blobs.push({
      lat:  clat + Math.sin(seed + i * 17.3) * spread,
      lng:  clng + Math.cos(seed + i * 13.7) * spread,
      agb,
      rx:   0.14 + ((Math.cos(seed + i * 5.3) + 1) / 2) * 0.36,
      ry:   0.10 + ((Math.sin(seed + i * 4.1) + 1) / 2) * 0.30,
      angle:(Math.sin(seed + i * 9) * Math.PI) / 3,
      intensity: Math.min(agb / 150, 1),
    });
  }
  // A few extreme hotspots
  for (let i = 0; i < 12; i++) {
    const agb = 110 + ((Math.cos(seed + i * 3.7) + 1) / 2) * 40;
    blobs.push({
      lat:  clat + Math.sin(seed + i * 23.1) * spread * 0.6,
      lng:  clng + Math.cos(seed + i * 19.9) * spread * 0.6,
      agb,
      rx:   0.08 + ((Math.sin(seed + i * 2.9) + 1) / 2) * 0.14,
      ry:   0.06 + ((Math.cos(seed + i * 3.3) + 1) / 2) * 0.12,
      angle:(Math.cos(seed + i * 6) * Math.PI) / 4,
      intensity: Math.min(agb / 150, 1),
    });
  }
  const biomassValues = blobs.map(({ agb }) => agb).filter((value) => Number.isFinite(value) && value >= 0);
  const sortedBiomass = [...biomassValues].sort((a, b) => a - b);
  const percentile = (position: number) => sortedBiomass[Math.min(sortedBiomass.length - 1, Math.floor(sortedBiomass.length * position))] ?? 0;
  const lowerBound = percentile(0.02);
  const upperBound = percentile(0.98);

  return blobs
    .filter((blob) => isValidCoordinate(blob.lat, blob.lng))
    .map((blob) => ({
      ...blob,
      intensity: normalizeBiomass(blob.agb, lowerBound, Math.max(lowerBound, upperBound)),
    }));
}

// ─── Biomass Heatmap Canvas Overlay ──────────────────────────────────────────

function BiomassHeatmapLayer({
  visible,
  region,
  records,
}: {
  visible: boolean;
  region: Region;
  records: BiomassRecord[];
}) {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const blobs = useMemo(() => generateBlobs(region, records), [region, records]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visible) return;
    const size = map.getSize();
    canvas.width = size.x;
    canvas.height = size.y;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, size.x, size.y);

    // ── Pass 1: wide soft glow for high-intensity blobs ──────────────────────
    ctx.globalCompositeOperation = "lighter";
    for (const blob of blobs) {
      const center = map.latLngToContainerPoint([blob.lat, blob.lng]);
      const edgeX  = map.latLngToContainerPoint([blob.lat, blob.lng + blob.rx]);
      const edgeY  = map.latLngToContainerPoint([blob.lat + blob.ry, blob.lng]);
      const rpx = Math.abs(edgeX.x - center.x);
      const rpy = Math.abs(edgeY.y - center.y);
      if (rpx < 2) continue;
      const maxR = Math.max(rpx, rpy) * (1.25 + blob.intensity * 1.35);
      const [cr, cg, cb] = heatmapRGB(blob.agb);
      const glowAlpha = 0.1 + blob.intensity * 0.2;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, maxR);
      grad.addColorStop(0,   `rgba(${cr},${cg},${cb},${glowAlpha.toFixed(3)})`);
      grad.addColorStop(0.5, `rgba(${cr},${cg},${cb},${(glowAlpha * 0.4).toFixed(3)})`);
      grad.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(blob.angle);
      ctx.scale(rpx / Math.max(rpx, rpy), rpy / Math.max(rpx, rpy));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(0, 0, maxR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // ── Pass 2: sharp core blobs sorted low→high so hot blobs are on top ────
    ctx.globalCompositeOperation = "source-over";
    const sorted = [...blobs].sort((a, b) => a.intensity - b.intensity);
    for (const blob of sorted) {
      const center = map.latLngToContainerPoint([blob.lat, blob.lng]);
      const edgeX  = map.latLngToContainerPoint([blob.lat, blob.lng + blob.rx]);
      const edgeY  = map.latLngToContainerPoint([blob.lat + blob.ry, blob.lng]);
      const rpx = Math.abs(edgeX.x - center.x);
      const rpy = Math.abs(edgeY.y - center.y);
      if (rpx < 2 || rpy < 2) continue;
      const maxR = Math.max(rpx, rpy);
      const [cr, cg, cb] = heatmapRGB(blob.agb);
      const coreAlpha = 0.3 + blob.intensity * 0.62;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, maxR);
      grad.addColorStop(0,    `rgba(${cr},${cg},${cb},${coreAlpha.toFixed(3)})`);
      grad.addColorStop(0.42, `rgba(${cr},${cg},${cb},${(coreAlpha * 0.62).toFixed(3)})`);
      grad.addColorStop(0.78, `rgba(${cr},${cg},${cb},${(coreAlpha * 0.15).toFixed(3)})`);
      grad.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`);
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(blob.angle);
      ctx.scale(rpx / maxR, rpy / maxR);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(0, 0, maxR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.globalCompositeOperation = "source-over";
  }, [map, visible, blobs]);

  useEffect(() => {
    if (!visible) {
      if (canvasRef.current) canvasRef.current.style.display = "none";
      return;
    }

    // Create canvas attached to map container so it isn't affected by pane CSS transforms
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

// ─── Land Cover Overlay ───────────────────────────────────────────────────────

// Land-cover anchor points (geo-fixed, like biomass)
const LAND_COVER_ANCHORS = [
  { lat: 11.6, lng: 76.2, color: "#4A7C59", r: 0.5 },
  { lat: 11.1, lng: 76.9, color: "#92704C", r: 0.4 },
  { lat: 11.8, lng: 77.1, color: "#C5A96B", r: 0.35 },
  { lat: 10.9, lng: 76.5, color: "#7B9E87", r: 0.45 },
  { lat: 11.4, lng: 76.4, color: "#5B7FA6", r: 0.38 },
  { lat: 11.7, lng: 76.7, color: "#4A7C59", r: 0.3 },
  { lat: 10.8, lng: 77.0, color: "#92704C", r: 0.42 },
];

function LandCoverLayer({ visible }: { visible: boolean }) {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visible) return;
    const size = map.getSize();
    canvas.width = size.x;
    canvas.height = size.y;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, size.x, size.y);

    for (const anchor of LAND_COVER_ANCHORS) {
      const center = map.latLngToContainerPoint([anchor.lat, anchor.lng]);
      const edge = map.latLngToContainerPoint([anchor.lat, anchor.lng + anchor.r]);
      const rpx = Math.abs(edge.x - center.x);
      if (rpx < 3) continue;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, rpx);
      grad.addColorStop(0, anchor.color + "55");
      grad.addColorStop(0.6, anchor.color + "33");
      grad.addColorStop(1, anchor.color + "00");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(center.x, center.y, rpx, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [map, visible]);

  useEffect(() => {
    if (!visible) {
      if (canvasRef.current) canvasRef.current.style.display = "none";
      return;
    }
    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:451;";
      const landCoverPane = map.getPane("land-cover-overlay") ?? map.createPane("land-cover-overlay", map.getContainer());
      landCoverPane.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:451;";
      landCoverPane.appendChild(canvas);
      canvasRef.current = canvas;
    }
    canvas.style.display = "";
    draw();
    map.on("move zoom viewreset moveend zoomend", draw);
    return () => {
      map.off("move zoom viewreset moveend zoomend", draw);
    };
  }, [map, visible, draw]);

  useEffect(() => () => { canvasRef.current?.remove(); canvasRef.current = null; }, []);

  return null;
}

// ─── Map Click Handler ────────────────────────────────────────────────────────

function MapClickHandler({
  regionId,
  onSelect,
}: {
  regionId: string;
  onSelect: (p: SelectedPoint) => void;
}) {
  useMapEvents({
    async click(e) {
      const { lat, lng } = e.latlng;
      const agb =
        40 +
        Math.abs(Math.sin(lat * 100 + lng * 73)) * 100 +
        Math.abs(Math.cos(lat * 53 + lng * 97)) * 20;
      const carbonStock = agb * 0.47;
      const co2e = carbonStock * 3.67;
      const landCover =
        LAND_COVER_TYPES[
          Math.floor(Math.abs(Math.sin(lat + lng) * LAND_COVER_TYPES.length))
        ];
      const fallbackPoint = {
        lat: +lat.toFixed(5),
        lng: +lng.toFixed(5),
        agb: +agb.toFixed(1),
        carbonStock: +carbonStock.toFixed(1),
        co2e: +co2e.toFixed(1),
        landCover,
      };
      try {
        onSelect(await fetchPointEstimate(regionId, lat, lng));
      } catch {
        onSelect(fallbackPoint);
      }
    },
  });
  return null;
}

// ─── Selected Point Marker ────────────────────────────────────────────────────

function SelectedMarker({ point }: { point: SelectedPoint | null }) {
  const map = useMap();
  const markerRef = useRef<L.Marker | null>(null);

  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }
    if (!point) return;

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

    markerRef.current = L.marker([point.lat, point.lng], { icon }).addTo(map);
    return () => {
      markerRef.current?.remove();
    };
  }, [map, point]);

  return null;
}

// ─── Time Comparison ──────────────────────────────────────────────────────────

const TIME_YEARS = [2019, 2020, 2021, 2022, 2023];
const YEAR_CO2E: Record<number, number> = {
  2019: 1980000,
  2020: 2050000,
  2021: 2140000,
  2022: 2220000,
  2023: 2310000,
};

function TimeComparison() {
  const [expanded, setExpanded] = useState(false);
  const [fromYear, setFromYear] = useState(2021);
  const [toYear, setToYear] = useState(2023);

  const delta = YEAR_CO2E[toYear] - YEAR_CO2E[fromYear];
  const pct = ((delta / YEAR_CO2E[fromYear]) * 100).toFixed(1);
  const positive = delta >= 0;

  return (
    <div className="border-b border-[var(--color-border)]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-3 text-sm font-medium text-[var(--color-foreground)] hover:bg-[var(--color-muted)] transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span>Time Comparison</span>
        </div>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`transition-transform ${expanded ? "rotate-180" : ""}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="px-5 pb-4 bg-[var(--color-muted)]/40">
          <div className="flex items-center gap-3 mb-4 mt-2">
            <div className="flex-1">
              <label className="text-[10px] uppercase tracking-widest text-[var(--color-muted-foreground)] block mb-1">
                From
              </label>
              <select
                value={fromYear}
                onChange={(e) => setFromYear(+e.target.value)}
                className="w-full text-sm bg-white border border-[var(--color-border)] rounded-lg px-2 py-1.5 text-[var(--color-foreground)]"
              >
                {TIME_YEARS.filter((y) => y < toYear).map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </div>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="mt-5 text-[var(--color-muted-foreground)]"
            >
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
            <div className="flex-1">
              <label className="text-[10px] uppercase tracking-widest text-[var(--color-muted-foreground)] block mb-1">
                To
              </label>
              <select
                value={toYear}
                onChange={(e) => setToYear(+e.target.value)}
                className="w-full text-sm bg-white border border-[var(--color-border)] rounded-lg px-2 py-1.5 text-[var(--color-foreground)]"
              >
                {TIME_YEARS.filter((y) => y > fromYear).map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Delta bar */}
          <div className="bg-white rounded-xl p-4 border border-[var(--color-border)]">
            <p className="text-[10px] uppercase tracking-widest text-[var(--color-muted-foreground)] mb-2">
              CO₂e Change ({fromYear} → {toYear})
            </p>
            <div className="flex items-baseline gap-2 mb-3">
              <span
                className={`text-2xl font-mono font-medium ${positive ? "text-[var(--color-primary)]" : "text-red-500"}`}
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {positive ? "+" : ""}
                {(delta / 1000).toFixed(0)}k
              </span>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                tons CO₂e
              </span>
              <span
                className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full ${
                  positive
                    ? "bg-[var(--color-status-bg)] text-[var(--color-status)]"
                    : "bg-red-50 text-red-600"
                }`}
              >
                {positive ? "+" : ""}
                {pct}%
              </span>
            </div>
            {/* Progress bar */}
            <div className="h-1.5 bg-[var(--color-muted)] rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${positive ? "bg-[var(--color-primary)]" : "bg-red-400"}`}
                style={{ width: `${Math.min(Math.abs(+pct) * 3, 100)}%` }}
              />
            </div>
          </div>

          {/* Year timeline dots */}
          <div className="mt-3 relative">
            <div className="absolute top-2 left-0 right-0 h-px bg-[var(--color-border)]" />
            <div className="flex justify-between relative">
              {TIME_YEARS.map((y) => {
                const isFrom = y === fromYear;
                const isTo = y === toYear;
                const inRange = y >= fromYear && y <= toYear;
                return (
                  <div key={y} className="flex flex-col items-center">
                    <div
                      className={`w-3 h-3 rounded-full border-2 z-10 transition-all ${
                        isFrom || isTo
                          ? "bg-[var(--color-primary)] border-[var(--color-primary)] scale-110"
                          : inRange
                          ? "bg-white border-[var(--color-primary)]"
                          : "bg-white border-[var(--color-border)]"
                      }`}
                    />
                    <span className="text-[9px] text-[var(--color-muted-foreground)] mt-1">
                      {y}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Top Bar ──────────────────────────────────────────────────────────────────

function TopBar({
  region,
  regions,
  onRegionChange,
}: {
  region: Region;
  regions: Region[];
  onRegionChange: (r: Region) => void;
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

      <RegionDropdown
        regions={regions}
        selected={region}
        onChange={onRegionChange}
      />

      <div className="flex items-center gap-1.5 text-xs text-[var(--color-muted-foreground)] ml-1">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
        </svg>
        <span>{region.dateRange}</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 bg-[var(--color-muted)] rounded-full text-[var(--color-primary)]">
          <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" />
          Live Data
        </div>
        <button className="w-8 h-8 rounded-full bg-[var(--color-muted)] flex items-center justify-center text-[var(--color-muted-foreground)] hover:bg-[var(--color-border)] transition-colors">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="8" r="4" />
            <path d="M20 21a8 8 0 1 0-16 0" />
          </svg>
        </button>
      </div>
    </header>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar({
  region,
  selected,
  onClearSelection,
  loading,
}: {
  region: Region;
  selected: SelectedPoint | null;
  onClearSelection: () => void;
  loading: boolean;
}) {
  return (
    <aside className="w-[360px] min-w-[320px] bg-[var(--color-background)] border-l border-[var(--color-border)] flex flex-col overflow-hidden shrink-0">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-[var(--color-border)]">
        {selected ? (
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
              Selected Point
            </h2>
            <p className="text-xs text-[var(--color-muted-foreground)] mt-0.5 font-mono">
              {selected.lat}°N, {selected.lng}°E
            </p>
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
              {region.name}
            </p>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-3">
        {loading ? (
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
        ) : selected ? (
          <>
            <StatCard
              label="AGB"
              value={selected.agb.toFixed(1)}
              unit="Mg/ha"
              sub="Above-Ground Biomass"
              highlight={false}
            />
            <StatCard
              label="Carbon Stock"
              value={selected.carbonStock.toFixed(1)}
              unit="tons C/ha"
              sub="Estimated via AGB × 0.47"
            />
            <StatCard
              label="CO₂ Equivalent"
              value={selected.co2e.toFixed(1)}
              unit="tons CO₂e/ha"
              sub="Carbon stock × 3.67"
              highlight={true}
            />

            {/* Land cover chip */}
            <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-2">
                Land Cover Class
              </p>
              <span className="inline-flex items-center gap-2 text-sm font-medium text-[var(--color-foreground)]">
                <span className="w-2.5 h-2.5 rounded-full bg-[var(--color-secondary)]" />
                {selected.landCover}
              </span>
            </div>

            {/* Biomass gauge */}
            <div className="rounded-xl p-4 border border-[var(--color-border)] bg-white">
              <p className="text-xs font-medium uppercase tracking-widest text-[var(--color-muted-foreground)] mb-3">
                AGB Percentile
              </p>
              <div className="h-2 rounded-full bg-[var(--color-muted)] overflow-hidden mb-1.5">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[var(--color-chart-low)] via-[var(--color-chart-mid)] to-[var(--color-primary)]"
                  style={{ width: `${Math.min((selected.agb / 150) * 100, 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-[var(--color-muted-foreground)]">
                <span>Low</span>
                <span className="text-[var(--color-primary)] font-medium">
                  {Math.round((selected.agb / 150) * 100)}th pct
                </span>
                <span>High</span>
              </div>
            </div>
          </>
        ) : (
          <>
            <StatCard
              label="Total Carbon Stored"
              value={(region.totalCO2e / 1000000).toFixed(2) + "M"}
              unit="tons CO₂e"
              sub={`Across ${region.area.toLocaleString()} ha`}
              highlight={true}
            />
            <StatCard
              label="Mean AGB"
              value={region.meanAGB.toString()}
              unit="Mg/ha"
              sub="Above-Ground Biomass"
            />
            <StatCard
              label="Area Covered"
              value={region.area.toLocaleString()}
              unit="hectares"
              sub="Mapped forest boundary"
            />
            <StatCard
              label="Sample Points"
              value={region.samplePoints.toLocaleString()}
              unit="points"
              sub="Ground-truth validation"
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
                  Click any point on the map to view local biomass and carbon
                  estimates.
                </p>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Time comparison */}
      <div className="border-t border-[var(--color-border)]">
        <TimeComparison />
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-[var(--color-border)] flex items-center justify-between">
        <span className="text-[10px] text-[var(--color-muted-foreground)]">
          Satellite data: ESA Sentinel-2
        </span>
        <span className="text-[10px] text-[var(--color-muted-foreground)]">
          Model: Random Forest v2.3
        </span>
      </div>
    </aside>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [regions, setRegions] = useState(REGIONS);
  const [region, setRegion] = useState(REGIONS[0]);
  const [biomassByRegion, setBiomassByRegion] = useState<Record<string, BiomassRecord[]>>({});
  const [selected, setSelected] = useState<SelectedPoint | null>(null);
  const [loading, setLoading] = useState(false);
  const [layers, setLayers] = useState<LayerState>({
    satellite: true,
    biomass: true,
    landCover: false,
  });
  const [drawMode, setDrawMode] = useState(false);

  useEffect(() => {
    let active = true;
    fetchRegions<Region>()
      .then((apiRegions) => {
        if (!active || !apiRegions.length) return;
        setRegions(apiRegions);
        setRegion((current) => apiRegions.find((item) => item.id === current.id) ?? apiRegions[0]);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchBiomass(region.id)
      .then((records) => {
        if (!active) return;
        setBiomassByRegion((current) => ({ ...current, [region.id]: records }));
      })
      .catch(() => undefined)
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [region.id]);

  const handleRegionChange = useCallback((r: Region) => {
    setSelected(null);
    setRegion(r);
  }, []);

  const handleLayerToggle = useCallback((key: keyof LayerState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  return (
    <div
      className="h-screen flex flex-col overflow-hidden"
      style={{ background: "var(--color-background)" }}
    >
      <TopBar
        region={region}
        regions={regions}
        onRegionChange={handleRegionChange}
      />

      <div className="dashboard-content flex flex-1 overflow-hidden">
        {/* Map panel */}
        <div className="map-panel flex-1 relative">
          <MapContainer
            key={region.id}
            center={region.center}
            zoom={region.zoom}
            className="w-full h-full"
            zoomControl={true}
          >
            <MapSizeSync />
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
              region={region}
              records={biomassByRegion[region.id] ?? []}
            />
            <LandCoverLayer visible={layers.landCover} />
            <MapClickHandler regionId={region.id} onSelect={setSelected} />
            <SelectedMarker point={selected} />
          </MapContainer>

          {/* Loading overlay */}
          {loading && (
            <div className="absolute inset-0 bg-white/60 backdrop-blur-sm flex items-center justify-center z-[1000]">
              <div className="bg-white rounded-2xl px-8 py-6 shadow-2xl border border-[var(--color-border)] flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                <p className="text-sm font-medium text-[var(--color-foreground)]">
                  Loading biomass data…
                </p>
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  Fetching satellite tiles
                </p>
              </div>
            </div>
          )}

          {/* Layer toggle — top right */}
          <div className="absolute top-3 right-3 z-[1000] flex flex-col gap-2">
            <LayerToggleControl layers={layers} onChange={handleLayerToggle} />

            {/* Draw tool */}
            <button
              onClick={() => setDrawMode(!drawMode)}
              className={`bg-white/95 backdrop-blur-sm rounded-lg p-2.5 shadow-lg border transition-colors ${
                drawMode
                  ? "border-[var(--color-primary)] bg-[var(--color-muted)]"
                  : "border-[var(--color-border)]"
              }`}
              title="Draw area"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke={drawMode ? "var(--color-primary)" : "currentColor"}
                strokeWidth="2"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </button>
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
              ESA Sentinel-2 · ISRO Bhuvan
            </span>
          </div>

          {/* Draw mode hint */}
          {drawMode && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000]">
              <div className="bg-[var(--color-primary)] text-white text-xs font-medium px-4 py-2 rounded-full shadow-lg flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                Click on the map to select a point
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <Sidebar
          region={region}
          selected={selected}
          onClearSelection={() => setSelected(null)}
          loading={loading}
        />
      </div>
    </div>
  );
}
