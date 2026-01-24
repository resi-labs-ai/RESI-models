"""Data models for observability and WandB logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MinerResultLog:
    """
    Log entry for a single miner's evaluation result.

    Used for per-miner tracking in WandB tables.
    """

    hotkey: str
    score: float
    success: bool

    # Metrics (only if successful)
    mape: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    accuracy_10pct: float | None = None  # % within 10% of ground truth

    # Metadata
    model_hash: str | None = None
    inference_time_ms: float | None = None
    is_winner: bool = False
    is_copier: bool = False

    # Error info (only if failed)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WandB logging."""
        return {
            "hotkey": self.hotkey,
            "score": round(self.score, 6) if self.score else 0.0,
            "success": self.success,
            "mape": round(self.mape, 6) if self.mape is not None else None,
            "mae": round(self.mae, 2) if self.mae is not None else None,
            "rmse": round(self.rmse, 2) if self.rmse is not None else None,
            "r2": round(self.r2, 4) if self.r2 is not None else None,
            "accuracy_10pct": (
                round(self.accuracy_10pct, 4)
                if self.accuracy_10pct is not None
                else None
            ),
            "model_hash": self.model_hash,
            "inference_time_ms": (
                round(self.inference_time_ms, 2)
                if self.inference_time_ms is not None
                else None
            ),
            "is_winner": self.is_winner,
            "is_copier": self.is_copier,
            "error": self.error,
        }


@dataclass
class PropertyPredictionLog:
    """
    Log entry for a single property prediction.

    Used for dashboard joining - allows matching predictions
    to properties in the validation set.
    """

    # Property identifier (for joining with validation data)
    property_id: str  # zpid or address

    # Miner info
    hotkey: str

    # Prediction data
    predicted_price: float
    ground_truth_price: float

    # Computed fields
    absolute_error: float | None = None
    percentage_error: float | None = None

    def __post_init__(self) -> None:
        """Compute derived fields."""
        if self.absolute_error is None:
            self.absolute_error = abs(self.predicted_price - self.ground_truth_price)
        if self.percentage_error is None and self.ground_truth_price > 0:
            self.percentage_error = self.absolute_error / self.ground_truth_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WandB table row."""
        return {
            "property_id": self.property_id,
            "hotkey": self.hotkey,
            "predicted_price": round(self.predicted_price, 2),
            "ground_truth_price": round(self.ground_truth_price, 2),
            "absolute_error": (
                round(self.absolute_error, 2) if self.absolute_error else None
            ),
            "percentage_error": (
                round(self.percentage_error, 6) if self.percentage_error else None
            ),
        }


@dataclass
class EvaluationLog:
    """
    Complete log entry for a validation round.

    Contains summary metrics and references to detailed tables.
    """

    # Timestamp
    timestamp: datetime
    evaluation_date: str  # YYYY-MM-DD format for easy filtering

    # Validator info
    validator_hotkey: str
    netuid: int

    # Dataset info
    dataset_size: int

    # Evaluation summary
    models_evaluated: int
    models_succeeded: int
    models_failed: int

    # Winner info
    winner_hotkey: str
    winner_score: float
    winner_mape: float | None = None
    winner_block: int | None = None

    # Anti-cheat summary
    duplicate_groups_found: int = 0
    copiers_detected: int = 0

    # Timing
    total_evaluation_time_ms: float = 0.0

    # Detailed results (logged as WandB table)
    miner_results: list[MinerResultLog] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for WandB scalar logging.

        Does not include miner_results - those go to a separate table.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "evaluation_date": self.evaluation_date,
            "validator_hotkey": self.validator_hotkey,
            "netuid": self.netuid,
            "dataset_size": self.dataset_size,
            "models_evaluated": self.models_evaluated,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "winner_hotkey": self.winner_hotkey,
            "winner_score": round(self.winner_score, 6),
            "winner_mape": (
                round(self.winner_mape, 6) if self.winner_mape is not None else None
            ),
            "winner_block": self.winner_block,
            "duplicate_groups_found": self.duplicate_groups_found,
            "copiers_detected": self.copiers_detected,
            "total_evaluation_time_ms": round(self.total_evaluation_time_ms, 2),
        }


@dataclass
class WandbConfig:
    """Configuration for WandB logging."""

    # Project settings
    project: str = "subnet-46-evaluations"
    entity: str | None = None  # WandB team/user, None = default

    # Authentication
    api_key: str | None = None  # WandB API key, or set WANDB_API_KEY env var

    # Run settings
    run_name: str | None = None  # Auto-generated if None
    tags: list[str] = field(default_factory=list)

    # Feature flags
    enabled: bool = True
    offline: bool = False  # Run in offline mode

    # What to log
    log_miner_table: bool = True  # Log per-miner results table
    log_predictions_table: bool = True  # Log per-property predictions

    # Prediction logging settings
    # Only log predictions for top N miners (to limit data volume)
    predictions_top_n_miners: int = 10
