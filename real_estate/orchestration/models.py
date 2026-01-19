"""Data models for validation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from real_estate.duplicate_detector import DuplicateDetectionResult
    from real_estate.evaluation import EvaluationBatch
    from real_estate.incentives import IncentiveWeights, WinnerSelectionResult


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

    def to_dict(self) -> dict:
        """
        Serialize for state persistence.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
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
