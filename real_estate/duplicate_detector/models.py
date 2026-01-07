"""Data models for duplicate detection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DuplicateGroup:
    """
    A group of models with identical or near-identical predictions.

    Immutable to prevent accidental modification after detection.
    """

    hotkeys: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate group has at least 2 members."""
        if len(self.hotkeys) < 2:
            raise ValueError("DuplicateGroup must have at least 2 hotkeys")

    @property
    def size(self) -> int:
        """Number of models in this duplicate group."""
        return len(self.hotkeys)

    def contains(self, hotkey: str) -> bool:
        """Check if a hotkey is in this group."""
        return hotkey in self.hotkeys

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hotkeys": list(self.hotkeys),
            "size": self.size,
        }


@dataclass(frozen=True)
class DuplicateDetectionResult:
    """
    Complete result of duplicate detection analysis.

    Main output for incentivization: copier_hotkeys (models to zero-score).
    Additional data (groups, pioneers) preserved for logging/analysis.
    """

    copier_hotkeys: frozenset[str]
    """Hotkeys that copied another model - should receive 0 score."""

    pioneer_hotkeys: frozenset[str] = frozenset()
    """Hotkeys that were first to submit (within duplicate groups)."""

    groups: tuple[DuplicateGroup, ...] = ()
    """Duplicate groups for logging/analysis."""

    skipped_hotkeys: tuple[str, ...] = ()
    """Hotkeys skipped due to missing chain metadata."""

    @property
    def total_duplicates(self) -> int:
        """Total number of models involved in duplication."""
        return sum(g.size for g in self.groups)

    def is_copier(self, hotkey: str) -> bool:
        """Check if a hotkey is a copier (for incentivization filtering)."""
        return hotkey in self.copier_hotkeys

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "copier_hotkeys": sorted(self.copier_hotkeys),
            "pioneer_hotkeys": sorted(self.pioneer_hotkeys),
            "groups": [g.to_dict() for g in self.groups],
            "skipped_hotkeys": list(self.skipped_hotkeys),
            "total_duplicates": self.total_duplicates,
            "copier_count": len(self.copier_hotkeys),
            "pioneer_count": len(self.pioneer_hotkeys),
        }
