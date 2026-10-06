"""Central configuration: paths, spatial grid definition, and feature schema."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, List

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "trips.csv"
DEFAULT_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "demand_model.joblib"
DEFAULT_METRICS_PATH: Final[Path] = PROJECT_ROOT / "models" / "metrics.json"
DEFAULT_GEOJSON_PATH: Final[Path] = PROJECT_ROOT / "models" / "zone_demand.geojson"

# Service area bounding box (WGS84). Upper bounds are exclusive.
LAT_MIN: Final[float] = 40.60
LAT_MAX: Final[float] = 40.90
LON_MIN: Final[float] = -74.10
LON_MAX: Final[float] = -73.80

# Square grid cell size in degrees (~3.3 km in latitude).
CELL_SIZE_DEG: Final[float] = 0.03
N_ROWS: Final[int] = int(round((LAT_MAX - LAT_MIN) / CELL_SIZE_DEG))
N_COLS: Final[int] = int(round((LON_MAX - LON_MIN) / CELL_SIZE_DEG))

# Central business district reference point.
CBD_LAT: Final[float] = 40.75
CBD_LON: Final[float] = -73.98
EARTH_RADIUS_KM: Final[float] = 6371.0088

TARGET_COLUMN: Final[str] = "demand"

REQUIRED_TRIP_COLUMNS: Final[List[str]] = [
    "pickup_datetime",
    "pickup_lat",
    "pickup_lon",
    "temperature_c",
    "precipitation_mm",
]

FEATURE_COLUMNS: Final[List[str]] = [
    "zone_x",
    "zone_y",
    "dist_to_cbd_km",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "temperature_c",
    "precipitation_mm",
    "is_raining",
]

CONTINUOUS_FEATURES: Final[List[str]] = [
    "dist_to_cbd_km",
    "hour_sin",
    "hour_cos",
    "temperature_c",
    "precipitation_mm",
]

# Synthetic data generation parameters.
HOURLY_PROFILE: Final[np.ndarray] = np.array(
    [
        0.30, 0.20, 0.15, 0.10, 0.10, 0.20,
        0.50, 0.90, 1.20, 1.00, 0.80, 0.80,
        0.90, 0.90, 0.90, 1.00, 1.20, 1.50,
        1.40, 1.10, 1.00, 0.90, 0.70, 0.50,
    ],
    dtype=float,
)
# Hotspots: business district, transit hub, nightlife district.
HOTSPOT_LAT: Final[np.ndarray] = np.array([40.75, 40.65, 40.72], dtype=float)
HOTSPOT_LON: Final[np.ndarray] = np.array([-73.98, -73.85, -74.00], dtype=float)
HOTSPOT_SIGMA: Final[np.ndarray] = np.array([0.012, 0.010, 0.010], dtype=float)
DAY_HOTSPOT_PROBS: Final[np.ndarray] = np.array([0.55, 0.25, 0.20], dtype=float)
NIGHT_HOTSPOT_PROBS: Final[np.ndarray] = np.array([0.15, 0.20, 0.65], dtype=float)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a consistent, readable format.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )