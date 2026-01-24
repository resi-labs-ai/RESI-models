"""Data models for miner CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluation.models import PredictionMetrics


@dataclass(frozen=True)
class EvaluateResult:
    """
    Result of local model evaluation.

    Contains metrics and pass/fail status for miner feedback.
    """

    model_path: str
    success: bool
    metrics: PredictionMetrics | None = None
    error_message: str | None = None
    inference_time_ms: float | None = None

    @property
    def score(self) -> float:
        """Get model score (0.0-1.0). Returns 0 if evaluation failed."""
        if self.metrics is None:
            return 0.0
        return self.metrics.score


@dataclass(frozen=True)
class SubmitResult:
    """
    Result of model submission to chain.

    Contains commitment details and transaction info.
    """

    model_path: str
    hf_repo_id: str
    model_hash: str
    success: bool
    error_message: str | None = None

    # Block at which submission was initiated (not necessarily inclusion block)
    submitted_at_block: int | None = None

    # Extrinsic ID in "block-index" format (set if scanning was successful)
    extrinsic_id: str | None = None

    def get_extrinsic_record(self, hotkey: str) -> dict | None:
        """
        Build extrinsic_record.json content for HuggingFace upload.

        Args:
            hotkey: The miner's hotkey SS58 address.

        Returns:
            Dict ready to be written as JSON, or None if extrinsic not found.
        """
        if self.extrinsic_id is None:
            return None

        return {
            "extrinsic": self.extrinsic_id,
            "hotkey": hotkey,
        }
