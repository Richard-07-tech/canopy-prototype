import "dotenv/config";
import cors from "cors";
import express from "express";
import { MongoClient } from "mongodb";

const app = express();
const port = Number(process.env.PORT ?? 8787);
const mongoUri = process.env.MONGODB_URI;
const databaseName = process.env.MONGODB_DB ?? "forest_biomass";
const mongoClient = mongoUri ? new MongoClient(mongoUri) : null;
let database;

const regions = [
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

const landCoverTypes = [
  "Tropical Evergreen",
  "Semi-Evergreen",
  "Mixed Deciduous",
  "Shola Forest",
  "Grassland-Forest Mosaic",
];

function generatedBiomass(region) {
  const seed = region.id.charCodeAt(0) * 13 + region.id.charCodeAt(region.id.length - 1) * 7;
  const [centerLat, centerLng] = region.center;
  const spread = 0.65;
  const points = [];

  for (let index = 0; index < 90; index += 1) {
    points.push({
      id: `${region.id}-background-${index}`,
      regionId: region.id,
      latitude: centerLat + Math.sin(seed + index * 17.3) * spread,
      longitude: centerLng + Math.cos(seed + index * 13.7) * spread,
      biomassMgHa: 20 + ((Math.sin(seed + index * 7.1) + 1) / 2) * 130,
      landCover: landCoverTypes[index % landCoverTypes.length],
    });
  }

  for (let index = 0; index < 12; index += 1) {
    points.push({
      id: `${region.id}-hotspot-${index}`,
      regionId: region.id,
      latitude: centerLat + Math.sin(seed + index * 23.1) * spread * 0.6,
      longitude: centerLng + Math.cos(seed + index * 19.9) * spread * 0.6,
      biomassMgHa: 110 + ((Math.cos(seed + index * 3.7) + 1) / 2) * 40,
      landCover: landCoverTypes[(index + 2) % landCoverTypes.length],
    });
  }

  return points;
}

function getRegion(id) {
  return regions.find((region) => region.id === id);
}

function normalizePoint(document) {
  return {
    id: String(document.id ?? `${document.regionId}-${document.latitude}-${document.longitude}`),
    regionId: String(document.regionId),
    latitude: Number(document.latitude),
    longitude: Number(document.longitude),
    biomassMgHa: Number(document.biomassMgHa ?? document.agb ?? document.biomass ?? 0),
    landCover: document.landCover ?? "Unknown",
  };
}

async function connectMongo() {
  if (!mongoClient) return;
  await mongoClient.connect();
  database = mongoClient.db(databaseName);
  console.log(`MongoDB connected: ${databaseName}`);
}

async function getRegionDocuments() {
  if (!database) return regions;
  const documents = await database.collection("regions").find({}).toArray();
  return documents.length ? documents.map((document) => ({ ...document, _id: undefined })) : regions;
}

async function getBiomassDocuments(region) {
  if (!database) return generatedBiomass(region);
  const documents = await database.collection("biomass_observations").find({ regionId: region.id }).toArray();
  return documents.length ? documents.map(normalizePoint) : generatedBiomass(region);
}

app.use(cors());
app.use(express.json());

app.get("/api/health", (_request, response) => {
  response.json({ status: "ok", database: database ? "mongodb" : "memory" });
});

app.get("/api/regions", async (_request, response, next) => {
  try {
    response.json(await getRegionDocuments());
  } catch (error) {
    next(error);
  }
});

app.get("/api/regions/:regionId/biomass", async (request, response, next) => {
  try {
    const region = getRegion(request.params.regionId);
    if (!region) return response.status(404).json({ error: "Region not found" });
    response.json(await getBiomassDocuments(region));
  } catch (error) {
    next(error);
  }
});

app.get("/api/point-estimates", async (request, response, next) => {
  try {
    const region = getRegion(String(request.query.regionId ?? ""));
    const latitude = Number(request.query.lat);
    const longitude = Number(request.query.lng);
    if (!region || !Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      return response.status(400).json({ error: "regionId, lat, and lng are required" });
    }

    const points = await getBiomassDocuments(region);
    const nearest = points.reduce((current, point) => {
      const distance = Math.hypot(point.latitude - latitude, point.longitude - longitude);
      return !current || distance < current.distance ? { point, distance } : current;
    }, null)?.point;
    const agb = Number(nearest?.biomassMgHa ?? 0);
    const carbonStock = agb * 0.47;
    response.json({
      lat: Number(latitude.toFixed(5)),
      lng: Number(longitude.toFixed(5)),
      agb: Number(agb.toFixed(1)),
      carbonStock: Number(carbonStock.toFixed(1)),
      co2e: Number((carbonStock * 3.67).toFixed(1)),
      landCover: nearest?.landCover ?? "Unknown",
    });
  } catch (error) {
    next(error);
  }
});

app.use((error, _request, response, _next) => {
  console.error(error);
  response.status(500).json({ error: "Internal server error" });
});

connectMongo()
  .then(() => app.listen(port, () => console.log(`Forest Biomass API listening on http://localhost:${port}`)))
  .catch((error) => {
    console.error("MongoDB connection failed", error);
    process.exitCode = 1;
  });
