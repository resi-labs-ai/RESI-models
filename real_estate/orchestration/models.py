"""Data models for validation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from real_estate.data import FeatureLayout
    from real_estate.duplicate_detector import DuplicateDetectionResult
    from real_estate.evaluation import EvaluationBatch
    from real_estate.generalization_detector import GeneralizationDetectionResult
    from real_estate.incentives import IncentiveWeights, WinnerSelectionResult
    from real_estate.model_inspector import InspectionBatchResult


@dataclass(frozen=True)
class EncodedModels:
    """Result of per-model feature encoding."""

    model_paths: dict[str, Path]
    """Hotkey -> ONNX path (models that failed encoding are excluded)."""

    features: dict[str, np.ndarray]
    """Hotkey -> encoded feature matrix (N_samples x N_features)."""

    layouts: dict[str, FeatureLayout]
    """Hotkey -> feature layout (column indices for perturbation)."""


@dataclass
class ValidationResult:
    """Result of a complete validation round."""

    weights: IncentiveWeights
    """Final weight distribution for chain."""

    winner: WinnerSelectionResult
    """Winner selection details."""

    eval_batch: EvaluationBatch
    """All evaluation results."""

    duplicate_result: DuplicateDetectionResult
    """Duplicate detection results."""

    generalization_result: GeneralizationDetectionResult | None = None
    """Generalization detection results."""

    inspection_result: InspectionBatchResult | None = None
    """Pre-flight inspection results."""

    per_model_num_features: dict[str, int] | None = None
    """Hotkey -> number of features used by each model."""

    def to_dict(self) -> dict:
        """
        Serialize for state persistence.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        result = {
            "winner_hotkey": self.winner.winner_hotkey,
            "winner_score": round(self.winner.winner_score, 6),
            "results": {
                r.hotkey: {
                    "score": round(r.score, 6) if r.success else None,
                    "mape": round(r.metrics.mape, 6) if r.metrics else None,
                    "success": r.success,
                    "error": str(r.error) if r.error else None,
                }
                for r in self.eval_batch.results
            },
            "copiers": sorted(self.duplicate_result.copier_hotkeys),
            "weights": {
                hotkey: round(weight, 6)
                for hotkey, weight in self.weights.weights.items()
            },
        }

        if self.generalization_result is not None:
            result["memorizers"] = sorted(self.generalization_result.memorizer_hotkeys)

        if self.inspection_result is not None:
            result["rejected"] = sorted(self.inspection_result.rejected_hotkeys)

        return result
