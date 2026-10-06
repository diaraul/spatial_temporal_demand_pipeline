"""FastAPI service exposing the demand forecasting model."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.model import load_model, predict_demand
from src.preprocessing import build_inference_features, zone_centers

config.configure_logging()
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Validated payload for a single demand prediction."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(..., description="Target hour, e.g. 2024-02-01T08:00:00.")
    lat: float = Field(
        ..., ge=config.LAT_MIN, lt=config.LAT_MAX, description="Latitude inside the service area."
    )
    lon: float = Field(
        ..., ge=config.LON_MIN, lt=config.LON_MAX, description="Longitude inside the service area."
    )
    temperature_c: float = Field(..., ge=-50.0, le=60.0, description="Air temperature in Celsius.")
    precipitation_mm: float = Field(
        0.0, ge=0.0, le=500.0, description="Hourly precipitation in millimetres."
    )


class PredictionResponse(BaseModel):
    """Predicted demand for the grid zone containing the requested location."""

    zone_x: int
    zone_y: int
    zone_center_lat: float
    zone_center_lon: float
    predicted_trips_per_hour: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model artifact once at application startup."""
    model_path = Path(os.environ.get("MODEL_PATH", str(config.DEFAULT_MODEL_PATH)))
    try:
        app.state.artifact = load_model(model_path)
        logger.info("Model loaded from %s.", model_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Model unavailable (%s). /predict will return 503.", exc)
        app.state.artifact = None
    yield


app = FastAPI(
    title="Spatial-Temporal Demand Forecasting API",
    version="1.0.0",
    description="Predicts hourly trip demand for a city grid zone.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    """Report service liveness and whether a model is loaded."""
    loaded = getattr(request.app.state, "artifact", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict expected trips for the zone containing (lat, lon) at the given hour."""
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    record = pd.DataFrame(
        [
            {
                "timestamp": payload.timestamp.replace(tzinfo=None),
                "lat": payload.lat,
                "lon": payload.lon,
                "temperature_c": payload.temperature_c,
                "precipitation_mm": payload.precipitation_mm,
            }
        ]
    )
    try:
        features = build_inference_features(record)
        prediction = predict_demand(artifact["pipeline"], features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction failure.")
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc

    zone_x = int(features["zone_x"].iloc[0])
    zone_y = int(features["zone_y"].iloc[0])
    center_lat, center_lon = zone_centers(zone_x, zone_y)
    return PredictionResponse(
        zone_x=zone_x,
        zone_y=zone_y,
        zone_center_lat=float(center_lat),
        zone_center_lon=float(center_lon),
        predicted_trips_per_hour=float(prediction[0]),
    )