"""Prediction grouper for identifying models with identical predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .models import DuplicateGroup

if TYPE_CHECKING:
    from ..evaluation.models import EvaluationResult


@dataclass(frozen=True)
class GrouperConfig:
    """Configuration for prediction grouping."""

    similarity_threshold: float = 1e-6
    """
    Threshold for considering predictions identical.
    Predictions are rounded to this precision before comparison.
    Default 1e-6 means predictions within 0.000001 are considered identical.
    """


class PredictionGrouper:
    """
    Groups models with identical or near-identical predictions.

    Uses configurable precision threshold to handle floating-point comparison.
    Creates a hashable key from predictions by rounding to threshold precision.

    Usage:
        grouper = PredictionGrouper(GrouperConfig())
        groups = grouper.group_predictions(evaluation_results)
        # Returns only groups with 2+ models (actual duplicates)
    """

    def __init__(self, config: GrouperConfig | None = None):
        """
        Initialize grouper.

        Args:
            config: Grouper configuration. Uses defaults if None.
        """
        self._config = config or GrouperConfig()

    def group_predictions(
        self,
        results: list[EvaluationResult],
    ) -> list[DuplicateGroup]:
        """
        Group evaluation results by identical predictions.

        Args:
            results: List of evaluation results to analyze.
                     Only successful results (with predictions) are considered.

        Returns:
            List of DuplicateGroup objects, each containing 2+ hotkeys
            with identical predictions. Empty list if no duplicates found.

        Example:
            results = [
                EvaluationResult(hotkey="A", predictions=np.array([100, 200])),
                EvaluationResult(hotkey="B", predictions=np.array([100, 200])),
                EvaluationResult(hotkey="C", predictions=np.array([150, 250])),
            ]
            groups = grouper.group_predictions(results)
            # Returns [DuplicateGroup(hotkeys=("A", "B"))]
        """
        # Filter to successful results with predictions
        valid_results = [r for r in results if r.success and r.predictions is not None]

        # Group by prediction hash
        prediction_groups: dict[str, list[str]] = {}
        for result in valid_results:
            key = self._create_prediction_key(result.predictions)
            if key not in prediction_groups:
                prediction_groups[key] = []
            prediction_groups[key].append(result.hotkey)

        # Filter to groups with 2+ members (actual duplicates)
        duplicate_groups = [
            DuplicateGroup(hotkeys=tuple(hotkeys))
            for hotkeys in prediction_groups.values()
            if len(hotkeys) >= 2
        ]

        return duplicate_groups

    def _create_prediction_key(self, predictions: np.ndarray) -> str:
        """
        Create hashable key from predictions array.

        Rounds to threshold precision, then creates string representation.
        This handles floating-point comparison issues.

        Args:
            predictions: numpy array of predictions

        Returns:
            String key that can be used for dictionary grouping
        """
        # Calculate number of decimal places from threshold
        # e.g., 1e-6 -> 6 decimal places
        decimals = int(-np.log10(self._config.similarity_threshold))

        # Round predictions to threshold precision
        rounded = np.round(predictions.flatten(), decimals=decimals)

        # Convert to bytes hex for fast hashing
        return rounded.tobytes().hex()
