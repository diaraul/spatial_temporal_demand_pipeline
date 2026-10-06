# 🚖 Spatial-Temporal Demand & Predictive Optimization Pipeline

Divides a city into a geospatial grid and forecasts hourly trip demand per zone, so operations teams can pre-position drivers or inventory and reduce idle time and missed requests. Built with Python, GeoPandas/Shapely, Scikit-Learn (Poisson gradient boosting with time-series cross-validation), FastAPI/Pydantic, and Pytest.

## Project Overview

**Business problem:** Ride-hailing, delivery, and micromobility operators lose revenue when supply is in the wrong place at the wrong time. This project forecasts *where* and *when* demand will occur.

**What it does**
- Ingests and validates raw trip pickups (CSV or built-in synthetic generator).
- Maps every pickup to a grid zone and builds a complete zone × hour demand panel, including zero-demand cells.
- Engineers spatial features (distance to the CBD, zone coordinates), temporal features (cyclical hour encoding, rush hour, weekend), and weather features.
- Trains a Poisson gradient-boosting model and compares it with a historical-average baseline on a chronological holdout.
- Serves predictions through a validated REST API.

## Architecture

```
 Raw trips (CSV / synthetic)
          │
          ▼
┌──────────────────────┐   invalid rows dropped + logged
│ preprocessing.py     │──────────────────────────────┐
│  validate_trips      │                              │
│  assign_grid_zones   │ (GeoPandas / Shapely)        │
│  aggregate_demand    │ zone x hour panel            │
│  engineer_features   │◄── shared by training & API  │
└──────────┬───────────┘                              │
           ▼                                          │
┌──────────────────────┐                              │
│ model.py             │                              │
│  time_based_split    │ chronological holdout        │
│  TimeSeriesSplit CV  │ no future leakage            │
│  baseline comparison │                              │
│  joblib artifact     │──► models/demand_model.joblib│
└──────────┬───────────┘     models/metrics.json      │
           │                 models/zone_demand.geojson
           ▼
┌──────────────────────┐
│ api.py (FastAPI)     │  POST /predict   GET /health
│  Pydantic validation │◄─ same feature code as training
└──────────────────────┘
```

## Key Technical Highlights

- **Leakage-safe evaluation:** chronological holdout plus `TimeSeriesSplit` instead of random splits.
- **Benchmarked against a baseline:** the model's MAE is compared with a per-zone, per-hour historical average.
- **Count-aware modelling:** Poisson loss for non-negative demand counts.
- **No training/serving skew:** `engineer_features` is reused by the API.
- **Spatial processing:** grid polygons built with Shapely, GeoPandas GeoDataFrame, GeoJSON export for mapping tools.
- **Defensive engineering:** type hints, docstrings, structured logging, explicit exceptions, strict Pydantic schemas (`extra="forbid"`, bounded ranges).
- **Tested:** Pytest suite covering preprocessing, edge cases, model behaviour, persistence, and API contracts.

## Results

Run training, then copy the numbers from `models/metrics.json` into this table:

| Metric (chronological holdout) | Historical-average baseline | Gradient boosting |
|---|---|---|
| MAE | _from metrics.json_ | _from metrics.json_ |
| RMSE | _from metrics.json_ | _from metrics.json_ |
| R² | _from metrics.json_ | _from metrics.json_ |

The file also reports `mae_improvement_vs_baseline_pct`, which you can quote on your resume.

## Setup & Installation

```bash
git clone https://github.com/<your-username>/spatial-temporal-demand-pipeline.git
cd spatial-temporal-demand-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Run the tests
pytest -v

# 2. Train (generates synthetic data; writes artifacts to ./models)
python -m src.model

#    Optional: train from a CSV file instead
python -m src.preprocessing --output data/raw/trips.csv
python -m src.model --data-path data/raw/trips.csv

# 3. Serve the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`. Set `MODEL_PATH` to load a model from a different location.

## API Usage

**Health check**
```bash
curl http://localhost:8000/health
```

**Predict demand for a zone**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "timestamp": "2024-02-01T08:00:00",
        "lat": 40.75,
        "lon": -73.98,
        "temperature_c": 4.5,
        "precipitation_mm": 1.2
      }'
```

Response shape (values depend on the trained model):
```json
{
  "zone_x": 4,
  "zone_y": 5,
  "zone_center_lat": 40.755,
  "zone_center_lon": -73.965,
  "predicted_trips_per_hour": 12.3
}
```

**Validation error example (HTTP 422)**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-02-01T08:00:00", "lat": 12.0, "lon": -73.98, "temperature_c": 4.5}'
```

## Future Improvements

- Replace synthetic data with public NYC TLC trip records.
- Add lag and rolling-demand features, which need a feature store at inference time.
- Add a `/predict/batch` endpoint and containerize with Docker.
- Track experiments with MLflow and add CI with GitHub Actions.