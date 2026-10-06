"""Unit and integration tests for preprocessing, modelling, and the API."""

import json
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from src import config
from src.api import app
from src.model import (
    build_pipeline,
    compute_metrics,
    cross_validate_model,
    historical_average_baseline,
    load_model,
    predict_demand,
    save_artifact,
)
from src.preprocessing import (
    aggregate_demand,
    assign_grid_zones,
    build_inference_features,
    build_zone_geodataframe,
    engineer_features,
    export_zone_geojson,
    generate_synthetic_trips,
    haversine_km,
    time_based_split,
    validate_trips,
)

VALID_PAYLOAD = {
    "timestamp": "2024-01-10T08:00:00",
    "lat": 40.75,
    "lon": -73.98,
    "temperature_c": 6.5,
    "precipitation_mm": 0.0,
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Small, fast demand panel shared across tests."""
    trips = validate_trips(generate_synthetic_trips(n_trips=6000, days=7, seed=7))
    return aggregate_demand(trips)


@pytest.fixture(scope="module")
def fitted(panel: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Quickly fitted pipeline with its training data."""
    X = engineer_features(panel)
    y = panel[config.TARGET_COLUMN]
    pipeline = build_pipeline(random_state=0, max_iter=20)
    pipeline.fit(X, y)
    return pipeline, X, y


@pytest.fixture()
def client(fitted: Tuple[Pipeline, pd.DataFrame, pd.Series]) -> Iterator[TestClient]:
    """API client with a pre-loaded model (lifespan intentionally skipped)."""
    app.state.artifact = {
        "pipeline": fitted[0],
        "feature_columns": list(config.FEATURE_COLUMNS),
        "metrics": {},
    }
    yield TestClient(app)
    app.state.artifact = None


# ---------------------------------------------------------------- preprocessing
def test_haversine_known_distance() -> None:
    assert float(haversine_km(0.0, 0.0, 0.0, 1.0)) == pytest.approx(111.19, rel=0.01)
    assert float(haversine_km(40.75, -73.98, 40.75, -73.98)) == pytest.approx(0.0, abs=1e-9)


def test_assign_grid_zones_corners() -> None:
    df = pd.DataFrame(
        {
            "lat": [config.LAT_MIN + 0.001, config.LAT_MAX - 0.001],
            "lon": [config.LON_MIN + 0.001, config.LON_MAX - 0.001],
        }
    )
    out = assign_grid_zones(df)
    assert out[["zone_x", "zone_y"]].iloc[0].tolist() == [0, 0]
    assert out[["zone_x", "zone_y"]].iloc[1].tolist() == [config.N_COLS - 1, config.N_ROWS - 1]


def test_assign_grid_zones_rejects_out_of_bounds_and_nulls() -> None:
    with pytest.raises(ValueError):
        assign_grid_zones(pd.DataFrame({"lat": [41.5], "lon": [-73.9]}))
    with pytest.raises(ValueError):
        assign_grid_zones(pd.DataFrame({"lat": [np.nan], "lon": [-73.9]}))
    with pytest.raises(ValueError):
        assign_grid_zones(pd.DataFrame({"lat": [40.7]}))


def test_validate_trips_cleans_bad_rows() -> None:
    df = pd.DataFrame(
        {
            "pickup_datetime": ["2024-01-01 08:00", "2024-01-01 09:00", "not a date", "2024-01-01 10:00"],
            "pickup_lat": [40.75, 55.0, 40.7, np.nan],
            "pickup_lon": [-73.98, -73.98, -73.9, -73.9],
            "temperature_c": [5.0, 5.0, 5.0, 5.0],
            "precipitation_mm": [-2.0, 0.0, 0.0, 0.0],
        }
    )
    cleaned = validate_trips(df)
    assert len(cleaned) == 1
    assert cleaned["precipitation_mm"].iloc[0] == 0.0


def test_validate_trips_missing_columns_and_all_invalid() -> None:
    with pytest.raises(ValueError):
        validate_trips(pd.DataFrame({"pickup_lat": [40.7]}))
    bad = pd.DataFrame(
        {
            "pickup_datetime": ["2024-01-01"],
            "pickup_lat": [10.0],
            "pickup_lon": [10.0],
            "temperature_c": [1.0],
            "precipitation_mm": [0.0],
        }
    )
    with pytest.raises(ValueError):
        validate_trips(bad)


def test_generate_synthetic_trips_is_valid_and_reproducible() -> None:
    first = generate_synthetic_trips(n_trips=500, days=3, seed=1)
    second = generate_synthetic_trips(n_trips=500, days=3, seed=1)
    assert list(first.columns) == config.REQUIRED_TRIP_COLUMNS
    assert len(validate_trips(first)) == 500
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError):
        generate_synthetic_trips(n_trips=0)


def test_aggregate_demand_is_zero_filled_and_conserves_trips() -> None:
    trips = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:50", "2024-01-01 02:20"]),
            "pickup_lat": [40.75, 40.75, 40.62],
            "pickup_lon": [-73.98, -73.98, -74.08],
            "temperature_c": [5.0, 5.0, 4.0],
            "precipitation_mm": [0.0, 0.0, 1.0],
        }
    )
    result = aggregate_demand(trips)
    assert len(result) == 3 * config.N_ROWS * config.N_COLS
    assert result[config.TARGET_COLUMN].sum() == 3
    assert result[config.TARGET_COLUMN].max() == 2
    assert result[["temperature_c", "precipitation_mm"]].notna().all().all()


def test_aggregate_demand_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        aggregate_demand(pd.DataFrame({"pickup_lat": [40.7]}))


def test_engineer_features_flags_and_bounds() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-06 08:00", "2024-01-08 08:00"]),
            "zone_x": [0, 5],
            "zone_y": [0, 5],
            "temperature_c": [3.0, 8.0],
            "precipitation_mm": [0.0, 2.0],
        }
    )
    feats = engineer_features(df)
    assert list(feats.columns) == config.FEATURE_COLUMNS
    assert feats["is_weekend"].tolist() == [1, 0]  # Saturday, Monday
    assert feats["is_rush_hour"].tolist() == [0, 1]
    assert feats["is_raining"].tolist() == [0, 1]
    assert feats["hour_sin"].between(-1, 1).all() and feats["hour_cos"].between(-1, 1).all()
    assert (feats["dist_to_cbd_km"] >= 0).all()


def test_engineer_features_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        engineer_features(pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}))
    out_of_range = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "zone_x": [99],
            "zone_y": [0],
            "temperature_c": [1.0],
            "precipitation_mm": [0.0],
        }
    )
    with pytest.raises(ValueError):
        engineer_features(out_of_range)


def test_build_inference_features_extreme_weather_is_finite() -> None:
    record = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-07-04 00:00"),
                "lat": 40.61,
                "lon": -74.09,
                "temperature_c": -30.0,
                "precipitation_mm": 400.0,
            }
        ]
    )
    feats = build_inference_features(record)
    assert np.isfinite(feats.to_numpy(dtype=float)).all()


def test_time_based_split_is_chronological(panel: pd.DataFrame) -> None:
    train, test = time_based_split(panel, test_fraction=0.25)
    assert len(train) + len(test) == len(panel)
    assert train["timestamp"].max() < test["timestamp"].min()
    with pytest.raises(ValueError):
        time_based_split(panel, test_fraction=1.5)


def test_zone_geodataframe_and_geojson_export(panel: pd.DataFrame, tmp_path) -> None:
    zones = build_zone_geodataframe()
    assert len(zones) == config.N_ROWS * config.N_COLS
    assert zones.crs.to_epsg() == 4326
    first = zones[(zones["zone_x"] == 0) & (zones["zone_y"] == 0)].iloc[0]
    assert first["center_lat"] == pytest.approx(config.LAT_MIN + config.CELL_SIZE_DEG / 2)

    path = export_zone_geojson(panel, tmp_path / "zones.geojson")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["features"]) == config.N_ROWS * config.N_COLS


# ---------------------------------------------------------------------- model
def test_predictions_are_non_negative_and_finite(
    fitted: Tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    pipeline, X, _ = fitted
    preds = predict_demand(pipeline, X.head(50))
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_predict_demand_rejects_bad_input(
    fitted: Tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    pipeline, X, _ = fitted
    with pytest.raises(ValueError):
        predict_demand(pipeline, X.iloc[0:0])
    with pytest.raises(ValueError):
        predict_demand(pipeline, X.drop(columns=["hour"]))


def test_compute_metrics() -> None:
    perfect = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert perfect["mae"] == 0.0 and perfect["rmse"] == 0.0 and perfect["r2"] == 1.0
    off = compute_metrics(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert off["mae"] == pytest.approx(1.0) and off["rmse"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_cross_validation_returns_expected_keys(
    fitted: Tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    _, X, y = fitted
    summary = cross_validate_model(build_pipeline(random_state=0, max_iter=10), X, y, n_splits=3)
    expected = {f"cv_{m}_{s}" for m in ("mae", "rmse", "r2") for s in ("mean", "std")}
    assert expected == set(summary)
    assert summary["cv_mae_mean"] >= 0.0
    with pytest.raises(ValueError):
        cross_validate_model(build_pipeline(), X, y, n_splits=1)


def test_baseline_has_no_nans(fitted: Tuple[Pipeline, pd.DataFrame, pd.Series]) -> None:
    _, X, y = fitted
    baseline = historical_average_baseline(X.iloc[:8000], y.iloc[:8000], X.iloc[8000:8100])
    assert baseline.shape == (100,)
    assert not np.isnan(baseline).any()


def test_save_and_load_roundtrip(
    fitted: Tuple[Pipeline, pd.DataFrame, pd.Series], tmp_path
) -> None:
    pipeline, X, _ = fitted
    artifact = {"pipeline": pipeline, "feature_columns": list(config.FEATURE_COLUMNS), "metrics": {}}
    path = save_artifact(artifact, tmp_path / "nested" / "model.joblib")
    loaded = load_model(path)
    np.testing.assert_allclose(
        predict_demand(pipeline, X.head(20)), predict_demand(loaded["pipeline"], X.head(20))
    )


def test_load_model_failure_modes(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_text("this is not a model", encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(corrupt)


# ------------------------------------------------------------------------ API
def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_payload(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_trips_per_hour"] >= 0.0
    assert 0 <= body["zone_x"] < config.N_COLS
    assert 0 <= body["zone_y"] < config.N_ROWS


@pytest.mark.parametrize(
    "overrides",
    [
        {"lat": 12.0},
        {"lon": 100.0},
        {"precipitation_mm": -1.0},
        {"temperature_c": 500.0},
        {"timestamp": "not-a-date"},
        {"unexpected_field": 1},
    ],
)
def test_predict_rejects_invalid_payloads(client: TestClient, overrides: dict) -> None:
    payload = {**VALID_PAYLOAD, **overrides}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "lat"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_returns_503_without_model(client: TestClient) -> None:
    app.state.artifact = None
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503