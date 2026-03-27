"""Data models for model inspection module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RejectionReason(str, Enum):
    """Why a model was rejected during inspection."""

    LOOKUP_PATTERN = "Lookup pattern detected"
    UNUSED_INITIALIZERS = "Model contains unused initializers"
    ZERO_PADDING = "Model contains significant zero-valued padding"
    PRICES_IN_WEIGHTS = "Suspicious amount of price-like values found in model weights"
    INSPECTION_FAILED = "Inspection container failed"


@dataclass(frozen=True)
class InspectionConfig:
    """Configuration for pre-flight model inspection."""

    price_count_threshold: int = 50_000
    """Minimum price-like values in weights to flag."""

    reject_unused_initializers: bool = True
    """Reject models with any unused initializers."""

    zero_padding_bytes_threshold: int = 20_000_000
    """Reject models with more than this many bytes of all-zero initializers."""

    memory_limit: str = "2g"
    """Docker memory limit for inspection container."""

    cpu_limit: float = 1.0
    """Docker CPU limit (1.0 = 1 core)."""

    timeout_seconds: int = 120
    """Inspection container timeout."""

    max_concurrent: int = 4
    """Maximum concurrent inspection containers."""

    image: str = "resi-onnx-runner:latest"
    """Docker image (same as inference — needs onnx + numpy)."""


@dataclass(frozen=True)
class ModelInspectionResult:
    """Inspection result for a single model."""

    hotkey: str
    has_lookup_pattern: bool
    has_unused_initializers: bool
    has_zero_padding: bool
    price_like_values: int
    zero_padding_bytes: int
    total_params: int
    rejection_reason: RejectionReason | None = None
    """Why the model was rejected, or None if it passed."""

    error: Exception | None = None
    """Exception if inspection failed (Docker error, timeout, etc.)."""

    @property
    def is_rejected(self) -> bool:
        return self.rejection_reason is not None

    @property
    def error_message(self) -> str | None:
        """Error message if inspection failed (truncated to 50 chars)."""
        if self.error is None:
            return None
        msg = f"{type(self.error).__name__}: {self.error}"
        if len(msg) > 50:
            return msg[:47] + "..."
        return msg

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hotkey": self.hotkey,
            "has_lookup_pattern": self.has_lookup_pattern,
            "has_unused_initializers": self.has_unused_initializers,
            "has_zero_padding": self.has_zero_padding,
            "price_like_values": self.price_like_values,
            "zero_padding_bytes": self.zero_padding_bytes,
            "total_params": self.total_params,
            "is_rejected": self.is_rejected,
            "rejection_reason": self.rejection_reason,
            "error": self.error_message,
        }


@dataclass(frozen=True)
class InspectionBatchResult:
    """Aggregate result of inspecting multiple models."""

    results: tuple[ModelInspectionResult, ...]
    """Per-model inspection results."""

    @property
    def rejected_hotkeys(self) -> frozenset[str]:
        """Hotkeys rejected by inspection — should not be evaluated."""
        return frozenset(r.hotkey for r in self.results if r.is_rejected)

    def is_rejected(self, hotkey: str) -> bool:
        """Check if a hotkey was rejected by inspection."""
        return hotkey in self.rejected_hotkeys

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        rejected = self.rejected_hotkeys
        return {
            "rejected_hotkeys": sorted(rejected),
            "rejected_count": len(rejected),
            "results": [r.to_dict() for r in self.results],
        }
