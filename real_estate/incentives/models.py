"""Data models for incentives module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WinnerSelectionConfig:
    """Configuration for winner selection."""

    score_threshold: float = 0.005
    """Models within this threshold of best score are considered equivalent."""


@dataclass(frozen=True)
class DistributorConfig:
    """Configuration for incentive distribution."""

    winner_share: float = 0.99
    """Share of emissions allocated to the winner (0.0-1.0)."""

    @property
    def non_winner_share(self) -> float:
        """Share distributed among non-winners (remainder after winner)."""
        return 1.0 - self.winner_share


@dataclass(frozen=True)
class WinnerCandidate:
    """A model in the winner set (within threshold of best score)."""

    hotkey: str
    score: float
    block_number: int


@dataclass(frozen=True)
class WinnerSelectionResult:
    """Result of winner selection process."""

    winner_hotkey: str
    """Hotkey of the selected winner."""

    winner_score: float
    """Score of the winning model."""

    winner_block: int
    """Block number of winner's commitment."""

    candidates: tuple[WinnerCandidate, ...]
    """All candidates in the winner set (within threshold)."""

    best_score: float
    """Best score among all evaluated models."""

    threshold: float
    """Threshold used for winner set determination."""

    @property
    def winner_set_size(self) -> int:
        """Number of models in the winner set."""
        return len(self.candidates)

    @property
    def was_tie_broken_by_commit_time(self) -> bool:
        """True if there was more than one candidate and commit time decided."""
        return len(self.candidates) > 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "winner_hotkey": self.winner_hotkey,
            "winner_score": round(self.winner_score, 6),
            "winner_block": self.winner_block,
            "best_score": round(self.best_score, 6),
            "threshold": self.threshold,
            "winner_set_size": self.winner_set_size,
            "tie_broken_by_commit_time": self.was_tie_broken_by_commit_time,
            "candidates": [
                {
                    "hotkey": c.hotkey,
                    "score": round(c.score, 6),
                    "block_number": c.block_number,
                }
                for c in self.candidates
            ],
        }


@dataclass(frozen=True)
class IncentiveWeights:
    """
    Final weight distribution for all miners.

    Weights are normalized to sum to 1.0 (or 0.0 if no valid models).
    """

    weights: dict[str, float] = field(default_factory=dict)
    """Mapping of hotkey -> weight (0.0-1.0). Sum should be 1.0."""

    winner_hotkey: str | None = None
    """Hotkey of the winner, if any."""

    winner_score: float | None = None
    """Score of the winner, if any."""

    @property
    def total(self) -> float:
        """Total weight allocated (should be 1.0 or 0.0)."""
        return sum(self.weights.values())

    @property
    def hotkeys(self) -> list[str]:
        """All hotkeys with assigned weights."""
        return list(self.weights.keys())

    def get_weight(self, hotkey: str) -> float:
        """Get weight for a specific hotkey (0.0 if not found)."""
        return self.weights.get(hotkey, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "winner_hotkey": self.winner_hotkey,
            "winner_score": (
                round(self.winner_score, 6) if self.winner_score is not None else None
            ),
            "total": round(self.total, 6),
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
        }
