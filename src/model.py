"""Model pipeline, time-series cross-validation, evaluation, and training CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing import (
    aggregate_demand,
    engineer_features,
    export_zone_geojson,
    generate_synthetic_trips,
    load_trips,
    time_based_split,
    validate_trips,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def build_pipeline(random_state: int = 42, max_iter: int = 200) -> Pipeline:
    """Create the preprocessing + Poisson gradient-boosting regression pipeline.

    Poisson loss is the natural choice for non-negative count targets such as
    trips per zone-hour.

    Args:
        random_state: Seed for reproducibility.
        max_iter: Number of boosting iterations.

    Returns:
        An unfitted scikit-learn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), config.CONTINUOUS_FEATURES)],
        remainder="passthrough",
    )
    regressor = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.1,
        max_iter=max_iter,
        max_depth=6,
        early_stopping=False,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def compute_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R^2.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys ``mae``, ``rmse``, and ``r2``.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    r2 = float(r2_score(true_arr, pred_arr)) if true_arr.size > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(true_arr, pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(true_arr, pred_arr))),
        "r2": r2,
    }


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 3
) -> Dict[str, float]:
    """Run leakage-safe time-series cross-validation.

    Rows must be sorted chronologically. Each fold trains on the past and
    validates on the following period.

    Args:
        pipeline: Unfitted pipeline (cloned internally by scikit-learn).
        X: Feature matrix sorted by time.
        y: Target series aligned with ``X``.
        n_splits: Number of time-series folds (at least 2).

    Returns:
        Mean and standard deviation of MAE, RMSE, and R^2 across folds.

    Raises:
        ValueError: If inputs are inconsistent or too small for the folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    try:
        scores = cross_validate(
            pipeline,
            X[config.FEATURE_COLUMNS],
            y,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
    except ValueError as exc:
        logger.error("Cross-validation failed: %s", exc)
        raise

    summary: Dict[str, float] = {}
    for name, sign in (("mae", -1.0), ("rmse", -1.0), ("r2", 1.0)):
        values = sign * scores[f"test_{name}"]
        summary[f"cv_{name}_mean"] = float(np.mean(values))
        summary[f"cv_{name}_std"] = float(np.std(values))
    return summary


def historical_average_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    """Predict demand as the historical mean for each (zone, hour-of-day).

    This is the benchmark any ML model must beat to justify its complexity.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.

    Returns:
        Baseline predictions aligned with the rows of ``X_test``.
    """
    keys = ["zone_x", "zone_y", "hour"]
    table = (
        X_train[keys]
        .assign(pred=y_train.to_numpy())
        .groupby(keys)["pred"]
        .mean()
        .reset_index()
    )
    merged = X_test[keys].merge(table, on=keys, how="left")
    return merged["pred"].fillna(float(y_train.mean())).to_numpy()


def predict_demand(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Generate non-negative demand predictions.

    Args:
        pipeline: Fitted pipeline.
        features: DataFrame containing ``config.FEATURE_COLUMNS``.

    Returns:
        Array of predicted trips per zone-hour (clipped at zero).

    Raises:
        ValueError: If features are empty or columns are missing.
    """
    if features.empty:
        raise ValueError("Cannot predict on an empty feature DataFrame.")
    missing = [c for c in config.FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    predictions = pipeline.predict(features[config.FEATURE_COLUMNS])
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)


def save_artifact(artifact: Dict[str, Any], path: PathLike) -> Path:
    """Persist a model artifact with joblib.

    Args:
        artifact: Dictionary containing at least ``pipeline`` and ``feature_columns``.
        path: Destination file path.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    out_path = Path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out_path)
    except OSError as exc:
        logger.error("Failed to save model to %s: %s", out_path, exc)
        raise
    logger.info("Saved model artifact to %s.", out_path)
    return out_path


def load_model(path: PathLike) -> Dict[str, Any]:
    """Load and validate a persisted model artifact.

    Args:
        path: Path to a joblib artifact.

    Returns:
        Artifact dictionary with ``pipeline`` and ``feature_columns``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupt or is not a valid artifact.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {file_path}")
    try:
        artifact = joblib.load(file_path)
    except Exception as exc:  # joblib/pickle raise many unrelated exception types
        logger.error("Could not load model from %s: %s", file_path, exc)
        raise ValueError(f"Could not load model artifact: {file_path}") from exc
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "feature_columns" not in artifact:
        raise ValueError("Model artifact is missing required keys 'pipeline'/'feature_columns'.")
    return artifact


def train_and_save(
    data_path: Optional[PathLike] = None,
    n_trips: int = 200_000,
    days: int = 28,
    seed: int = 42,
    model_path: PathLike = config.DEFAULT_MODEL_PATH,
    metrics_path: PathLike = config.DEFAULT_METRICS_PATH,
    geojson_path: PathLike = config.DEFAULT_GEOJSON_PATH,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Run the full training workflow and persist all artifacts.

    Steps: ingest -> aggregate to zone-hour panel -> features -> chronological
    holdout -> time-series CV -> holdout evaluation vs. baseline -> refit on all
    data -> save model, metrics, and zone GeoJSON.

    Args:
        data_path: Optional CSV of trips; synthetic data is generated if omitted.
        n_trips: Synthetic trip count (used only without ``data_path``).
        days: Synthetic day count (used only without ``data_path``).
        seed: Random seed.
        model_path: Output path for the model artifact.
        metrics_path: Output path for the metrics JSON.
        geojson_path: Output path for the zone demand GeoJSON.
        n_splits: Number of cross-validation folds.

    Returns:
        Dictionary of evaluation metrics.
    """
    if data_path is not None:
        trips = load_trips(data_path)
    else:
        trips = validate_trips(generate_synthetic_trips(n_trips=n_trips, days=days, seed=seed))

    panel = aggregate_demand(trips)
    train_panel, test_panel = time_based_split(panel, test_fraction=0.2)

    X_train = engineer_features(train_panel)
    y_train = train_panel[config.TARGET_COLUMN]
    X_test = engineer_features(test_panel)
    y_test = test_panel[config.TARGET_COLUMN]

    logger.info("Running %d-fold time-series cross-validation.", n_splits)
    cv_metrics = cross_validate_model(build_pipeline(random_state=seed), X_train, y_train, n_splits)

    holdout_pipeline = build_pipeline(random_state=seed)
    holdout_pipeline.fit(X_train[config.FEATURE_COLUMNS], y_train)
    model_metrics = compute_metrics(y_test, predict_demand(holdout_pipeline, X_test))
    baseline_metrics = compute_metrics(
        y_test, historical_average_baseline(X_train, y_train, X_test)
    )
    improvement = (
        100.0 * (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"]
        if baseline_metrics["mae"] > 0
        else 0.0
    )

    metrics: Dict[str, Any] = {
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "cross_validation": cv_metrics,
        "holdout_model": model_metrics,
        "holdout_baseline": baseline_metrics,
        "mae_improvement_vs_baseline_pct": float(improvement),
    }

    logger.info("Refitting final model on all %d rows.", len(panel))
    X_all = engineer_features(panel)
    y_all = panel[config.TARGET_COLUMN]
    final_pipeline = build_pipeline(random_state=seed)
    final_pipeline.fit(X_all[config.FEATURE_COLUMNS], y_all)

    save_artifact(
        {
            "pipeline": final_pipeline,
            "feature_columns": list(config.FEATURE_COLUMNS),
            "metrics": metrics,
        },
        model_path,
    )
    export_zone_geojson(panel, geojson_path)

    metrics_file = Path(metrics_path)
    try:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write metrics to %s: %s", metrics_file, exc)
        raise
    return metrics


def main(argv: "list[str] | None" = None) -> int:
    """CLI entry point for training.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    config.configure_logging()
    parser = argparse.ArgumentParser(description="Train the demand forecasting model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional trips CSV.")
    parser.add_argument("--n-trips", type=int, default=200_000)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        metrics = train_and_save(
            data_path=args.data_path,
            n_trips=args.n_trips,
            days=args.days,
            seed=args.seed,
            n_splits=args.n_splits,
        )
    except (ValueError, OSError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    logger.info("Training complete. Metrics:\n%s", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())