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

    Contains duplicate groups and pioneer information.
    """

    groups: tuple[DuplicateGroup, ...]
    pioneers: dict[str, bool]  # {hotkey: is_pioneer}
    skipped_hotkeys: tuple[str, ...] = ()  # Hotkeys skipped due to missing metadata

    @property
    def total_duplicates(self) -> int:
        """Total number of models involved in duplication."""
        return sum(g.size for g in self.groups)

    @property
    def pioneer_hotkeys(self) -> list[str]:
        """List of hotkeys that are pioneers."""
        return [hk for hk, is_pioneer in self.pioneers.items() if is_pioneer]

    @property
    def copier_hotkeys(self) -> list[str]:
        """List of hotkeys that are copiers (not pioneers)."""
        return [hk for hk, is_pioneer in self.pioneers.items() if not is_pioneer]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "groups": [g.to_dict() for g in self.groups],
            "pioneers": self.pioneers,
            "skipped_hotkeys": list(self.skipped_hotkeys),
            "total_duplicates": self.total_duplicates,
            "pioneer_count": len(self.pioneer_hotkeys),
            "copier_count": len(self.copier_hotkeys),
        }
