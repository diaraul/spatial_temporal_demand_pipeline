"""Data ingestion, validation, spatial gridding, and feature engineering."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src import config

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def haversine_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> np.ndarray:
    """Compute great-circle distance in kilometres between coordinate pairs.

    Args:
        lat1: Latitude(s) of the first point in degrees.
        lon1: Longitude(s) of the first point in degrees.
        lat2: Latitude(s) of the second point in degrees.
        lon2: Longitude(s) of the second point in degrees.

    Returns:
        Distance(s) in kilometres (broadcast to the input shapes).
    """
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def zone_centers(
    zone_x: Union[int, np.ndarray], zone_y: Union[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (latitude, longitude) centre of grid zone(s).

    Args:
        zone_x: Column index (longitude axis) of the zone(s).
        zone_y: Row index (latitude axis) of the zone(s).

    Returns:
        Tuple of (center_lat, center_lon).
    """
    zx = np.asarray(zone_x, dtype=float)
    zy = np.asarray(zone_y, dtype=float)
    center_lat = config.LAT_MIN + (zy + 0.5) * config.CELL_SIZE_DEG
    center_lon = config.LON_MIN + (zx + 0.5) * config.CELL_SIZE_DEG
    return center_lat, center_lon


def generate_synthetic_trips(
    n_trips: int = 200_000,
    days: int = 28,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate realistic synthetic trip pickups with weather and spatial hotspots.

    Demand follows a daily profile, is lower on weekends, rises with rain, and
    concentrates around three hotspots whose mix shifts between day and night.

    Args:
        n_trips: Number of trip records to generate.
        days: Number of consecutive days to simulate.
        seed: Random seed for reproducibility.
        start: First day of the simulation (ISO date string).

    Returns:
        DataFrame with the columns listed in ``config.REQUIRED_TRIP_COLUMNS``.

    Raises:
        ValueError: If ``n_trips`` or ``days`` is not positive.
    """
    if n_trips <= 0 or days <= 0:
        raise ValueError("n_trips and days must both be positive integers.")

    rng = np.random.default_rng(seed)
    hours = pd.date_range(start=start, periods=days * 24, freq="h")
    n_hours = len(hours)
    hour_of_day = hours.hour.to_numpy()
    day_of_week = hours.dayofweek.to_numpy()

    raining = rng.random(n_hours) < 0.15
    precipitation = np.where(raining, rng.gamma(2.0, 1.5, n_hours), 0.0)
    temperature = (
        12.0
        + 8.0 * np.sin(2.0 * np.pi * (hour_of_day - 9) / 24.0)
        + rng.normal(0.0, 1.5, n_hours)
    )

    weights = (
        config.HOURLY_PROFILE[hour_of_day]
        * np.where(day_of_week >= 5, 0.85, 1.0)
        * (1.0 + 0.05 * np.minimum(precipitation, 5.0))
    )
    weights = weights / weights.sum()
    hour_idx = rng.choice(n_hours, size=n_trips, p=weights)

    trip_hours = hour_of_day[hour_idx]
    is_night = (trip_hours >= 21) | (trip_hours < 5)
    n_night = int(is_night.sum())
    spot = np.empty(n_trips, dtype=int)
    spot[is_night] = rng.choice(3, size=n_night, p=config.NIGHT_HOTSPOT_PROBS)
    spot[~is_night] = rng.choice(3, size=n_trips - n_night, p=config.DAY_HOTSPOT_PROBS)

    lat = config.HOTSPOT_LAT[spot] + rng.normal(0.0, 1.0, n_trips) * config.HOTSPOT_SIGMA[spot]
    lon = config.HOTSPOT_LON[spot] + rng.normal(0.0, 1.0, n_trips) * config.HOTSPOT_SIGMA[spot]
    lat = np.clip(lat, config.LAT_MIN, config.LAT_MAX - 1e-9)
    lon = np.clip(lon, config.LON_MIN, config.LON_MAX - 1e-9)

    seconds = rng.integers(0, 3600, size=n_trips)
    pickup_datetime = hours[hour_idx] + pd.to_timedelta(seconds, unit="s")

    trips = pd.DataFrame(
        {
            "pickup_datetime": pickup_datetime,
            "pickup_lat": lat,
            "pickup_lon": lon,
            "temperature_c": temperature[hour_idx],
            "precipitation_mm": precipitation[hour_idx],
        }
    )
    logger.info("Generated %d synthetic trips across %d days.", len(trips), days)
    return trips


def validate_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw trip records.

    Coerces types, drops unparseable or null rows, removes pickups outside the
    service area, and clips negative precipitation to zero.

    Args:
        df: Raw trips with the columns in ``config.REQUIRED_TRIP_COLUMNS``.

    Returns:
        Cleaned DataFrame with a fresh RangeIndex.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """
    missing = [c for c in config.REQUIRED_TRIP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[config.REQUIRED_TRIP_COLUMNS].copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"], errors="coerce")
    for col in ("pickup_lat", "pickup_lon", "temperature_c", "precipitation_mm"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    n_before = len(out)
    out = out.dropna()
    in_box = (
        out["pickup_lat"].ge(config.LAT_MIN)
        & out["pickup_lat"].lt(config.LAT_MAX)
        & out["pickup_lon"].ge(config.LON_MIN)
        & out["pickup_lon"].lt(config.LON_MAX)
    )
    out = out.loc[in_box].copy()
    out["precipitation_mm"] = out["precipitation_mm"].clip(lower=0.0)

    dropped = n_before - len(out)
    if dropped:
        logger.warning("Dropped %d of %d trip rows during validation.", dropped, n_before)
    if out.empty:
        raise ValueError("No valid trip records remain after validation.")
    return out.reset_index(drop=True)


def load_trips(path: PathLike) -> pd.DataFrame:
    """Load trips from CSV and validate them.

    Args:
        path: Path to a CSV file containing the required trip columns.

    Returns:
        Validated trips DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed or fails validation.
    """
    file_path = Path(path)
    try:
        raw = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("Trip data file not found: %s", file_path)
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.error("Could not parse trip data file %s: %s", file_path, exc)
        raise ValueError(f"Could not parse trip data file: {file_path}") from exc
    logger.info("Loaded %d raw rows from %s.", len(raw), file_path)
    return validate_trips(raw)


def assign_grid_zones(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
) -> pd.DataFrame:
    """Map coordinates to integer grid-zone indices.

    Args:
        df: DataFrame containing latitude and longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        Copy of ``df`` with added ``zone_x`` and ``zone_y`` integer columns.

    Raises:
        ValueError: If columns are missing, coordinates are null, or any point
            falls outside the service area.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and '{lon_col}' are required.")

    out = df.copy()
    coords = out[[lat_col, lon_col]].astype(float)
    if coords.isna().any().any():
        raise ValueError("Coordinates must not contain null values.")

    zone_x = np.floor((coords[lon_col].to_numpy() - config.LON_MIN) / config.CELL_SIZE_DEG).astype(int)
    zone_y = np.floor((coords[lat_col].to_numpy() - config.LAT_MIN) / config.CELL_SIZE_DEG).astype(int)
    invalid = (zone_x < 0) | (zone_x >= config.N_COLS) | (zone_y < 0) | (zone_y >= config.N_ROWS)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} coordinate(s) fall outside the service area."
        )
    out["zone_x"] = zone_x
    out["zone_y"] = zone_y
    return out


def build_zone_geodataframe() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of all grid-zone polygons in EPSG:4326.

    Returns:
        GeoDataFrame with columns zone_x, zone_y, center_lat, center_lon, geometry.
    """
    records = []
    for zone_x in range(config.N_COLS):
        for zone_y in range(config.N_ROWS):
            min_lon = config.LON_MIN + zone_x * config.CELL_SIZE_DEG
            min_lat = config.LAT_MIN + zone_y * config.CELL_SIZE_DEG
            polygon = box(
                min_lon,
                min_lat,
                min_lon + config.CELL_SIZE_DEG,
                min_lat + config.CELL_SIZE_DEG,
            )
            centroid = polygon.centroid
            records.append(
                {
                    "zone_x": zone_x,
                    "zone_y": zone_y,
                    "center_lat": centroid.y,
                    "center_lon": centroid.x,
                    "geometry": polygon,
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def aggregate_demand(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips into a complete, zero-filled zone x hour demand panel.

    Args:
        trips: Validated trips (see ``validate_trips``).

    Returns:
        Panel with columns timestamp, zone_x, zone_y, demand, temperature_c,
        precipitation_mm, sorted by timestamp then zone.

    Raises:
        ValueError: If required columns are missing or ``trips`` is empty.
    """
    missing = [c for c in config.REQUIRED_TRIP_COLUMNS if c not in trips.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if trips.empty:
        raise ValueError("Cannot aggregate an empty trips DataFrame.")

    df = assign_grid_zones(trips, lat_col="pickup_lat", lon_col="pickup_lon")
    df["timestamp"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")

    counts = df.groupby(["timestamp", "zone_x", "zone_y"]).size().rename(config.TARGET_COLUMN)
    weather = df.groupby("timestamp")[["temperature_c", "precipitation_mm"]].mean()

    hours = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    full_index = pd.MultiIndex.from_product(
        [hours, np.arange(config.N_COLS), np.arange(config.N_ROWS)],
        names=["timestamp", "zone_x", "zone_y"],
    )
    panel = counts.reindex(full_index, fill_value=0).reset_index()

    weather = weather.reindex(hours).ffill().bfill()
    panel = panel.merge(weather, left_on="timestamp", right_index=True, how="left")
    panel = panel.sort_values(["timestamp", "zone_x", "zone_y"]).reset_index(drop=True)
    logger.info(
        "Built demand panel: %d rows (%d hours x %d zones).",
        len(panel),
        len(hours),
        config.N_COLS * config.N_ROWS,
    )
    return panel


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from zone indices, timestamps, and weather.

    The same function is used for training and online inference, which prevents
    training/serving skew.

    Args:
        df: DataFrame with columns timestamp, zone_x, zone_y, temperature_c,
            precipitation_mm.

    Returns:
        DataFrame containing exactly ``config.FEATURE_COLUMNS`` (same index as input).

    Raises:
        ValueError: If columns are missing, zones are out of range, or any
            engineered feature is null.
    """
    required = ["timestamp", "zone_x", "zone_y", "temperature_c", "precipitation_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Cannot engineer features from an empty DataFrame.")

    zone_x = df["zone_x"].astype(int)
    zone_y = df["zone_y"].astype(int)
    out_of_range = (
        (zone_x < 0) | (zone_x >= config.N_COLS) | (zone_y < 0) | (zone_y >= config.N_ROWS)
    )
    if out_of_range.any():
        raise ValueError("Zone indices fall outside the configured grid.")

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour
    day_of_week = ts.dt.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)
    rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))) & (day_of_week < 5)
    center_lat, center_lon = zone_centers(zone_x.to_numpy(), zone_y.to_numpy())
    precipitation = df["precipitation_mm"].astype(float).clip(lower=0.0)

    feats = pd.DataFrame(index=df.index)
    feats["zone_x"] = zone_x
    feats["zone_y"] = zone_y
    feats["dist_to_cbd_km"] = haversine_km(center_lat, center_lon, config.CBD_LAT, config.CBD_LON)
    feats["hour"] = hour
    feats["day_of_week"] = day_of_week
    feats["month"] = ts.dt.month
    feats["is_weekend"] = is_weekend
    feats["is_rush_hour"] = rush.astype(int)
    feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    feats["temperature_c"] = df["temperature_c"].astype(float)
    feats["precipitation_mm"] = precipitation
    feats["is_raining"] = (precipitation > 0.0).astype(int)

    feats = feats[config.FEATURE_COLUMNS]
    if feats.isna().any().any():
        raise ValueError("Engineered features contain null values.")
    return feats


def build_inference_features(records: pd.DataFrame) -> pd.DataFrame:
    """Convert raw prediction requests into model-ready features.

    Args:
        records: DataFrame with columns timestamp, lat, lon, temperature_c,
            precipitation_mm.

    Returns:
        Feature DataFrame matching ``config.FEATURE_COLUMNS``.

    Raises:
        ValueError: If inputs are invalid or outside the service area.
    """
    with_zones = assign_grid_zones(records, lat_col="lat", lon_col="lon")
    return engineer_features(with_zones)


def time_based_split(
    panel: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a panel chronologically so the test set strictly follows training.

    Args:
        panel: Demand panel with a ``timestamp`` column.
        test_fraction: Fraction of unique timestamps reserved for testing.

    Returns:
        Tuple of (train_panel, test_panel).

    Raises:
        ValueError: If the fraction is invalid or there are too few timestamps.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if "timestamp" not in panel.columns:
        raise ValueError("Panel must contain a 'timestamp' column.")

    unique_ts = pd.DatetimeIndex(panel["timestamp"].drop_duplicates()).sort_values()
    if len(unique_ts) < 2:
        raise ValueError("At least two distinct timestamps are required to split.")

    cut_idx = int(len(unique_ts) * (1.0 - test_fraction))
    cut_idx = min(max(cut_idx, 1), len(unique_ts) - 1)
    cutoff = unique_ts[cut_idx]
    train = panel[panel["timestamp"] < cutoff]
    test = panel[panel["timestamp"] >= cutoff]
    logger.info("Time split at %s: %d train rows, %d test rows.", cutoff, len(train), len(test))
    return train, test


def export_zone_geojson(panel: pd.DataFrame, path: PathLike) -> Path:
    """Write zone polygons with average hourly demand to a GeoJSON file.

    Args:
        panel: Demand panel from ``aggregate_demand``.
        path: Destination file path.

    Returns:
        Path of the written GeoJSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    avg = (
        panel.groupby(["zone_x", "zone_y"], as_index=False)[config.TARGET_COLUMN]
        .mean()
        .rename(columns={config.TARGET_COLUMN: "avg_hourly_demand"})
    )
    zones = build_zone_geodataframe().merge(avg, on=["zone_x", "zone_y"], how="left")
    zones["avg_hourly_demand"] = zones["avg_hourly_demand"].fillna(0.0)

    out_path = Path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(zones.to_json(), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write GeoJSON to %s: %s", out_path, exc)
        raise
    logger.info("Wrote zone GeoJSON to %s.", out_path)
    return out_path


def main(argv: "list[str] | None" = None) -> int:
    """CLI: write a synthetic trips CSV so the file-ingestion path can be exercised.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    config.configure_logging()
    parser = argparse.ArgumentParser(description="Generate a synthetic trips CSV.")
    parser.add_argument("--n-trips", type=int, default=200_000)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    try:
        trips = generate_synthetic_trips(args.n_trips, args.days, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trips.to_csv(args.output, index=False)
    except (ValueError, OSError) as exc:
        logger.error("Data generation failed: %s", exc)
        return 1
    logger.info("Saved %d trips to %s.", len(trips), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())