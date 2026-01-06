"""Unit tests for duplicate detection models."""

import pytest

from real_estate.duplicate_detection import DuplicateDetectionResult, DuplicateGroup


class TestDuplicateGroup:
    """Tests for DuplicateGroup dataclass."""

    def test_valid_group_creation(self) -> None:
        """Group with 2+ hotkeys is created successfully."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        assert group.hotkeys == ("A", "B")

    def test_three_hotkeys(self) -> None:
        """Group with 3 hotkeys is valid."""
        group = DuplicateGroup(hotkeys=("A", "B", "C"))
        assert group.size == 3

    def test_single_hotkey_raises_error(self) -> None:
        """Single hotkey raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            DuplicateGroup(hotkeys=("only_one",))

    def test_empty_hotkeys_raises_error(self) -> None:
        """Empty hotkeys raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            DuplicateGroup(hotkeys=())

    def test_size_property(self) -> None:
        """size property returns number of hotkeys."""
        group = DuplicateGroup(hotkeys=("A", "B", "C"))
        assert group.size == 3

    def test_contains_method_true(self) -> None:
        """contains returns True for hotkey in group."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        assert group.contains("A") is True
        assert group.contains("B") is True

    def test_contains_method_false(self) -> None:
        """contains returns False for hotkey not in group."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        assert group.contains("C") is False

    def test_to_dict_serialization(self) -> None:
        """to_dict produces serializable dict."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        result = group.to_dict()
        assert result == {"hotkeys": ["A", "B"], "size": 2}

    def test_frozen_immutable(self) -> None:
        """Frozen dataclass is immutable."""
        group = DuplicateGroup(hotkeys=("A", "B"))
        with pytest.raises(AttributeError):
            group.hotkeys = ("C", "D")  # type: ignore


class TestDuplicateDetectionResult:
    """Tests for DuplicateDetectionResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result has zero counts."""
        result = DuplicateDetectionResult(groups=(), pioneers={})
        assert result.total_duplicates == 0
        assert result.pioneer_hotkeys == []
        assert result.copier_hotkeys == []

    def test_total_duplicates_property(self) -> None:
        """total_duplicates sums all group sizes."""
        groups = (
            DuplicateGroup(hotkeys=("A", "B")),
            DuplicateGroup(hotkeys=("C", "D", "E")),
        )
        result = DuplicateDetectionResult(
            groups=groups,
            pioneers={"A": True, "B": False, "C": True, "D": False, "E": False},
        )
        assert result.total_duplicates == 5

    def test_pioneer_hotkeys_property(self) -> None:
        """pioneer_hotkeys returns only pioneers."""
        result = DuplicateDetectionResult(
            groups=(DuplicateGroup(hotkeys=("A", "B", "C")),),
            pioneers={"A": True, "B": False, "C": False},
        )
        assert result.pioneer_hotkeys == ["A"]

    def test_copier_hotkeys_property(self) -> None:
        """copier_hotkeys returns non-pioneers."""
        result = DuplicateDetectionResult(
            groups=(DuplicateGroup(hotkeys=("A", "B", "C")),),
            pioneers={"A": True, "B": False, "C": False},
        )
        assert set(result.copier_hotkeys) == {"B", "C"}

    def test_to_dict_serialization(self) -> None:
        """to_dict produces complete serializable dict."""
        groups = (DuplicateGroup(hotkeys=("A", "B")),)
        result = DuplicateDetectionResult(
            groups=groups, pioneers={"A": True, "B": False}
        )
        serialized = result.to_dict()

        assert serialized["total_duplicates"] == 2
        assert serialized["pioneer_count"] == 1
        assert serialized["copier_count"] == 1
        assert serialized["pioneers"] == {"A": True, "B": False}
        assert serialized["skipped_hotkeys"] == []
        assert len(serialized["groups"]) == 1

    def test_skipped_hotkeys_default_empty(self) -> None:
        """skipped_hotkeys defaults to empty tuple."""
        result = DuplicateDetectionResult(groups=(), pioneers={})
        assert result.skipped_hotkeys == ()

    def test_skipped_hotkeys_included_in_to_dict(self) -> None:
        """skipped_hotkeys is included in serialization."""
        groups = (DuplicateGroup(hotkeys=("A", "B")),)
        result = DuplicateDetectionResult(
            groups=groups,
            pioneers={"A": True, "B": False},
            skipped_hotkeys=("C", "D"),
        )
        serialized = result.to_dict()
        assert serialized["skipped_hotkeys"] == ["C", "D"]
