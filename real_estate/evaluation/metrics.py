"""
This module provides functions to calculate various prediction accuracy metrics:
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- RMSE: Root Mean Squared Error
- MdAPE: Median Absolute Percentage Error
- Accuracy@X%: Fraction of predictions within X% of actual
- R²: Coefficient of Determination

All percentage-based metrics are returned as decimals (0.0-1.0 scale).
For example, MAPE of 0.085 means 8.5% average error.

Scoring convention:
    Score = 1 - MAPE, so lower error = higher score.
    This abstraction exists to support future compound scoring systems
    where multiple metrics may be combined with different weights.
"""

from dataclasses import dataclass

import numpy as np

from .errors import EmptyDatasetError, MetricsError
from .models import PredictionMetrics


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for metrics calculation."""

    max_pct_error: float | None = None
    """Maximum percentage error cap (e.g., 2.0 = 200%). None = no capping."""

    accuracy_thresholds: tuple[float, ...] = (0.05, 0.10, 0.15)
    """Thresholds for accuracy metrics (as decimals). Default: 5%, 10%, 15%."""


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: MetricsConfig,
) -> PredictionMetrics:
    """
    Calculate all prediction metrics.

    Args:
        y_true: Ground truth prices (1D array)
        y_pred: Predicted prices (1D array)
        config: Metrics configuration.

    Returns:
        PredictionMetrics with all computed values

    Raises:
        MetricsError: If arrays have different lengths
        EmptyDatasetError: If no samples remain after filtering

    Example:
        >>> y_true = np.array([200_000, 500_000, 1_000_000])
        >>> y_pred = np.array([210_000, 450_000, 1_100_000])
        >>> metrics = calculate_metrics(y_true, y_pred, MetricsConfig())
        >>> print(f"MAPE: {metrics.mape:.4f}, Score: {metrics.score:.4f}")
        MAPE: 0.0833, Score: 0.9167
    """

    # Validate and convert inputs
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred):
        raise MetricsError(
            f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise EmptyDatasetError("Empty input arrays")

    # Calculate metrics
    mae = _calculate_mae(y_true, y_pred)
    mape = _calculate_mape(y_true, y_pred, config.max_pct_error)
    rmse = _calculate_rmse(y_true, y_pred)
    mdape = _calculate_mdape(y_true, y_pred)
    r2 = _calculate_r2(y_true, y_pred)

    # Calculate accuracy at each configured threshold
    accuracy = {
        threshold: _calculate_accuracy_at_threshold(y_true, y_pred, threshold)
        for threshold in config.accuracy_thresholds
    }

    return PredictionMetrics(
        mae=mae,
        mape=mape,
        rmse=rmse,
        mdape=mdape,
        accuracy=accuracy,
        r2=r2,
        n_samples=len(y_true),
    )


# --- Individual metric functions ---


def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in dollars."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _calculate_mape(
    y_true: np.ndarray, y_pred: np.ndarray, max_error: float | None = None
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Formula: MAPE = mean(|y_true - y_pred| / y_true)

    Args:
        y_true: Ground truth prices
        y_pred: Predicted prices
        max_error: Optional maximum percentage error cap (e.g., 2.0 = 200%). None = no capping.

    Returns:
        MAPE as decimal (e.g., 0.085 for 8.5%)
    """
    pct_errors = np.abs(y_true - y_pred) / y_true
    if max_error is not None:
        pct_errors = np.clip(pct_errors, 0, max_error)
    return float(np.mean(pct_errors))


def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in dollars."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _calculate_mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error as decimal."""
    pct_errors = np.abs(y_true - y_pred) / y_true
    return float(np.median(pct_errors))


def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination).

    Formula: R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)²)
    and SS_tot = sum((y_true - mean(y_true))²)

    Args:
        y_true: Ground truth prices
        y_pred: Predicted prices

    Returns:
        R² value (can be negative if model is worse than mean)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        # All ground truth values are identical
        # If predictions match, R² = 1; otherwise undefined (return 0)
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - (ss_res / ss_tot))


def _calculate_accuracy_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of predictions within threshold of actual (0.0-1.0)."""
    pct_errors = np.abs(y_true - y_pred) / y_true
    within_threshold = pct_errors < threshold
    return float(np.mean(within_threshold))


# --- Scoring functions ---


def mape_to_score(mape: float) -> float:
    """Convert MAPE to 0-1 score. See module docstring for scoring convention."""
    return max(0.0, 1.0 - mape)


def score_to_mape(score: float) -> float:
    """Convert score back to MAPE. See module docstring for scoring convention."""
    return 1.0 - score


def validate_predictions(
    predictions: np.ndarray,
    expected_length: int | None = None,
) -> np.ndarray:
    """
    Validate and normalize prediction array.

    Checks for:
    - Correct shape (1D or column vector)
    - No NaN or Inf values (these break metrics calculations)
    - Correct length if expected_length provided

    Args:
        predictions: Raw predictions from model
        expected_length: Expected number of predictions (optional)

    Returns:
        Validated 1D numpy array

    Raises:
        MetricsError: If predictions are invalid
    """
    predictions = np.asarray(predictions, dtype=np.float64)

    # Flatten if needed (handle (N,1) shape)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    elif predictions.ndim != 1:
        raise MetricsError(
            f"Invalid prediction shape: {predictions.shape}. Expected 1D or (N,1)."
        )

    # Check length
    if expected_length is not None and len(predictions) != expected_length:
        raise MetricsError(
            f"Prediction count mismatch: got {len(predictions)}, expected {expected_length}"
        )

    # Check for NaN/Inf (these break metrics calculations)
    if np.any(np.isnan(predictions)):
        nan_count = np.sum(np.isnan(predictions))
        raise MetricsError(f"Predictions contain {nan_count} NaN values")

    if np.any(np.isinf(predictions)):
        inf_count = np.sum(np.isinf(predictions))
        raise MetricsError(f"Predictions contain {inf_count} Inf values")

    # Note: We allow negative predictions - metrics will naturally penalize them heavily

    return predictions
