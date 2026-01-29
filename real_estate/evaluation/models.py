"""Data models for evaluation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionMetrics:
    """
    All computed metrics for a model's predictions.

    Contains both raw metrics and derived score.
    All percentage-based metrics are stored as decimals (e.g., 0.085 for 8.5%).
    """

    # Error metrics (lower is better)
    mae: float  # Mean Absolute Error ($)
    mape: float  # Mean Absolute Percentage Error (0.0-1.0+, e.g., 0.085 = 8.5%)
    rmse: float  # Root Mean Squared Error ($)
    mdape: float  # Median Absolute Percentage Error (0.0-1.0+)

    # Accuracy metrics (higher is better, 0.0-1.0)
    # Keys are thresholds as decimals (e.g., 0.05, 0.10, 0.15)
    # Values are fraction of predictions within that threshold
    accuracy: dict[float, float]

    # Explanatory metrics
    r2: float  # Coefficient of determination (-inf, 1]

    # Metadata
    n_samples: int  # Number of samples evaluated

    def get_accuracy(self, threshold: float) -> float | None:
        """Get accuracy at a specific threshold, or None if not computed."""
        return self.accuracy.get(threshold)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "mae": round(self.mae, 2),
            "mape": round(self.mape, 6),
            "rmse": round(self.rmse, 2),
            "mdape": round(self.mdape, 6),
            "r2": round(self.r2, 4),
            "n_samples": self.n_samples,
            "accuracy": {
                f"{threshold:.0%}": round(value, 4)
                for threshold, value in sorted(self.accuracy.items())
            },
        }
        return result

    @property
    def score(self) -> float:
        """
        Primary score for ranking models (0.0-1.0).

        Uses MAPE-based scoring: score = max(0, 1 - mape)
        A MAPE of 0.10 (10%) gives score of 0.90.
        """
        return max(0.0, 1.0 - self.mape)


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a single model.

    Contains predictions, metrics, and metadata.
    """

    hotkey: str
    predictions: np.ndarray | None = None
    metrics: PredictionMetrics | None = None
    error: Exception | None = None
    inference_time_ms: float | None = None

    # Model metadata (for context)
    model_hash: str | None = None

    @property
    def success(self) -> bool:
        """Success if we have predictions and metrics, and no error."""
        return (
            self.predictions is not None
            and self.metrics is not None
            and self.error is None
        )

    @property
    def score(self) -> float:
        """Get model score (0.0-1.0). Returns 0 if evaluation failed."""
        if self.metrics is None:
            return 0.0
        return self.metrics.score

    @property
    def error_message(self) -> str | None:
        """Get error message if evaluation failed (truncated to 50 chars)."""
        if self.error is None:
            return None
        msg = f"{type(self.error).__name__}: {self.error}"
        if len(msg) > 50:
            return msg[:47] + "..."
        return msg

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (excludes predictions array)."""
        result: dict[str, Any] = {
            "hotkey": self.hotkey,
            "success": self.success,
            "score": self.score,
        }

        if self.success:
            result["metrics"] = self.metrics.to_dict() if self.metrics else None
            result["inference_time_ms"] = self.inference_time_ms
            result["n_predictions"] = (
                len(self.predictions) if self.predictions is not None else 0
            )
        else:
            result["error"] = self.error_message

        if self.model_hash:
            result["model_hash"] = self.model_hash

        return result


@dataclass
class EvaluationBatch:
    """
    Results from evaluating multiple models on the same dataset.

    Contains summary statistics and individual results.
    """

    results: list[EvaluationResult] = field(default_factory=list)
    dataset_size: int = 0
    total_time_ms: float = 0.0

    @property
    def successful_count(self) -> int:
        """Number of models that evaluated successfully."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        """Number of models that failed evaluation."""
        return sum(1 for r in self.results if not r.success)

    @property
    def successful_results(self) -> list[EvaluationResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> list[EvaluationResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]

    def get_ranking(self) -> list[tuple[str, float]]:
        """
        Get models ranked by score (highest first).

        Returns:
            List of (hotkey, score) tuples, sorted by score descending.
        """
        scored = [(r.hotkey, r.score) for r in self.successful_results]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def get_best(self) -> EvaluationResult | None:
        """Get the highest-scoring result, or None if no successful evaluations."""
        if not self.successful_results:
            return None
        return max(self.successful_results, key=lambda r: r.score)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_size": self.dataset_size,
            "total_time_ms": round(self.total_time_ms, 2),
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "ranking": self.get_ranking(),
            "results": [r.to_dict() for r in self.results],
        }
