"""Data models for generalization detection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeneralizationConfig:
    """Configuration for generalization testing (perturbation parameters)."""

    global_noise_pct: float = 0.01
    """Multiplicative noise level for global perturbation (0.01 = +/-1%)."""

    global_threshold: float = 0.70
    """Minimum global robustness ratio to pass."""

    seed: int | None = None
    """Random seed for perturbation. None = random each call."""

    num_numeric_features: int = 52
    """Number of numeric features (for global perturbation range)."""


@dataclass(frozen=True)
class GeneralizationTestResult:
    """Per-model generalization test result."""

    hotkey: str
    original_score: float
    perturbed_score: float
    global_ratio: float
    is_memorizer: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hotkey": self.hotkey,
            "original_score": round(self.original_score, 4),
            "perturbed_score": round(self.perturbed_score, 4),
            "global_ratio": round(self.global_ratio, 4),
            "is_memorizer": self.is_memorizer,
        }


@dataclass(frozen=True)
class GeneralizationDetectionResult:
    """Aggregate result of generalization detection across all models."""

    memorizer_hotkeys: frozenset[str]
    """Hotkeys flagged as memorizers — should receive 0 weight."""

    test_results: tuple[GeneralizationTestResult, ...] = ()
    """Per-model test results for logging/analysis."""

    def is_memorizer(self, hotkey: str) -> bool:
        """Check if a hotkey is flagged as a memorizer."""
        return hotkey in self.memorizer_hotkeys

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "memorizer_hotkeys": sorted(self.memorizer_hotkeys),
            "memorizer_count": len(self.memorizer_hotkeys),
            "test_results": [r.to_dict() for r in self.test_results],
            "tested_count": len(self.test_results),
        }
