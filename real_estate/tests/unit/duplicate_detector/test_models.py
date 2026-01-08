"""Unit tests for duplicate detection models."""

import pytest

from real_estate.duplicate_detector import DuplicateDetectionResult, DuplicateGroup


class TestDuplicateGroup:
    """Tests for DuplicateGroup dataclass."""

    def test_single_hotkey_raises_error(self) -> None:
        """Single hotkey raises ValueError - need 2+ for a duplicate."""
        with pytest.raises(ValueError, match="at least 2"):
            DuplicateGroup(hotkeys=("only_one",))

    def test_empty_hotkeys_raises_error(self) -> None:
        """Empty hotkeys raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            DuplicateGroup(hotkeys=())

    def test_to_dict_serialization(self) -> None:
        """to_dict produces expected format for logging/API."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        result = group.to_dict()
        assert result == {"hotkeys": ["A", "B"], "size": 2}


class TestDuplicateDetectionResult:
    """Tests for DuplicateDetectionResult dataclass."""

    def test_total_duplicates_sums_group_sizes(self) -> None:
        """total_duplicates sums all group sizes."""
        groups = (
            DuplicateGroup(hotkeys=("A", "B")),
            DuplicateGroup(hotkeys=("C", "D", "E")),
        )
        result = DuplicateDetectionResult(
            copier_hotkeys=frozenset({"B", "D", "E"}),
            pioneer_hotkeys=frozenset({"A", "C"}),
            groups=groups,
        )
        assert result.total_duplicates == 5

    def test_is_copier_for_incentivization(self) -> None:
        """is_copier returns True for copiers, False otherwise."""
        result = DuplicateDetectionResult(
            copier_hotkeys=frozenset({"B", "C"}),
            pioneer_hotkeys=frozenset({"A"}),
        )
        assert result.is_copier("B") is True
        assert result.is_copier("C") is True
        assert result.is_copier("A") is False
        assert result.is_copier("unknown") is False

    def test_to_dict_serialization(self) -> None:
        """to_dict produces expected format for logging/API."""
        groups = (DuplicateGroup(hotkeys=("A", "B")),)
        result = DuplicateDetectionResult(
            copier_hotkeys=frozenset({"B"}),
            pioneer_hotkeys=frozenset({"A"}),
            groups=groups,
            skipped_hotkeys=("C",),
        )
        serialized = result.to_dict()

        assert serialized["copier_hotkeys"] == ["B"]
        assert serialized["pioneer_hotkeys"] == ["A"]
        assert serialized["skipped_hotkeys"] == ["C"]
        assert serialized["total_duplicates"] == 2
        assert serialized["copier_count"] == 1
        assert serialized["pioneer_count"] == 1
