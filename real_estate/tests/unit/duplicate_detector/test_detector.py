"""Unit tests for PioneerDetector."""

from unittest.mock import MagicMock

import pytest

from real_estate.duplicate_detector import (
    DuplicateGroup,
    PioneerDetectionResult,
    PioneerDetector,
)


def _create_mock_metadata(hotkey: str, block_number: int) -> MagicMock:
    """Create a mock ChainModelMetadata for testing."""
    metadata = MagicMock()
    metadata.hotkey = hotkey
    metadata.block_number = block_number
    return metadata


class TestPioneerDetectorDetectPioneers:
    """Tests for PioneerDetector.detect_pioneers method."""

    @pytest.fixture
    def detector(self) -> PioneerDetector:
        """Create pioneer detector."""
        return PioneerDetector()

    def test_pioneer_is_lowest_block_number(self, detector: PioneerDetector) -> None:
        """Model with lowest block_number is marked as pioneer."""
        groups = [DuplicateGroup(hotkeys=("A", "B", "C"))]
        metadata = {
            "A": _create_mock_metadata("A", 1000),
            "B": _create_mock_metadata("B", 900),  # Lowest - pioneer
            "C": _create_mock_metadata("C", 1100),
        }

        result = detector.detect_pioneers(groups, metadata)

        assert isinstance(result, PioneerDetectionResult)
        assert "B" in result.pioneer_hotkeys
        assert "A" in result.copier_hotkeys
        assert "C" in result.copier_hotkeys
        assert result.skipped_hotkeys == []

    def test_single_group_one_pioneer(self, detector: PioneerDetector) -> None:
        """Each group has exactly one pioneer."""
        groups = [DuplicateGroup(hotkeys=("A", "B"))]
        metadata = {
            "A": _create_mock_metadata("A", 500),  # Pioneer
            "B": _create_mock_metadata("B", 600),
        }

        result = detector.detect_pioneers(groups, metadata)

        assert len(result.pioneer_hotkeys) == 1

    def test_multiple_groups_multiple_pioneers(self, detector: PioneerDetector) -> None:
        """Multiple groups have multiple pioneers (one each)."""
        groups = [
            DuplicateGroup(hotkeys=("A", "B")),
            DuplicateGroup(hotkeys=("C", "D")),
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),  # Pioneer of group 1
            "B": _create_mock_metadata("B", 200),
            "C": _create_mock_metadata("C", 400),
            "D": _create_mock_metadata("D", 300),  # Pioneer of group 2
        }

        result = detector.detect_pioneers(groups, metadata)

        assert result.pioneer_hotkeys == frozenset({"A", "D"})
        assert result.copier_hotkeys == frozenset({"B", "C"})

    def test_empty_groups_returns_empty_result(self, detector: PioneerDetector) -> None:
        """Empty groups list returns empty sets."""
        result = detector.detect_pioneers([], {})
        assert result.pioneer_hotkeys == frozenset()
        assert result.copier_hotkeys == frozenset()
        assert result.skipped_hotkeys == []

    def test_all_hotkeys_in_result(self, detector: PioneerDetector) -> None:
        """All hotkeys from all groups appear in result."""
        groups = [
            DuplicateGroup(hotkeys=("A", "B")),
            DuplicateGroup(hotkeys=("C", "D", "E")),
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
            "C": _create_mock_metadata("C", 300),
            "D": _create_mock_metadata("D", 400),
            "E": _create_mock_metadata("E", 500),
        }

        result = detector.detect_pioneers(groups, metadata)

        all_hotkeys = result.pioneer_hotkeys | result.copier_hotkeys
        assert all_hotkeys == {"A", "B", "C", "D", "E"}


class TestPioneerDetectorMissingMetadata:
    """Tests for PioneerDetector handling of missing metadata."""

    @pytest.fixture
    def detector(self) -> PioneerDetector:
        """Create pioneer detector."""
        return PioneerDetector()

    def test_missing_metadata_skips_hotkey(self, detector: PioneerDetector) -> None:
        """Missing metadata for hotkey results in skip, not error."""
        groups = [DuplicateGroup(hotkeys=("A", "B", "C"))]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
            # C is missing
        }

        result = detector.detect_pioneers(groups, metadata)

        # A and B are processed, C is skipped
        assert "A" in result.pioneer_hotkeys  # Pioneer (lower block)
        assert "B" in result.copier_hotkeys
        assert "C" not in result.pioneer_hotkeys
        assert "C" not in result.copier_hotkeys
        assert result.skipped_hotkeys == ["C"]

    def test_skipped_hotkeys_are_sorted(self, detector: PioneerDetector) -> None:
        """Skipped hotkeys list is sorted alphabetically."""
        groups = [DuplicateGroup(hotkeys=("A", "D", "B", "C"))]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            # B, C, D are missing
        }

        result = detector.detect_pioneers(groups, metadata)

        # Group skipped (only 1 hotkey with metadata)
        assert result.pioneer_hotkeys == frozenset()
        assert result.copier_hotkeys == frozenset()
        assert result.skipped_hotkeys == ["B", "C", "D"]

    def test_group_with_one_remaining_hotkey_skipped(
        self, detector: PioneerDetector
    ) -> None:
        """Group with <2 hotkeys after filtering is skipped entirely."""
        groups = [DuplicateGroup(hotkeys=("A", "B"))]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            # B is missing - group now has only 1 hotkey
        }

        result = detector.detect_pioneers(groups, metadata)

        # No pioneers because group was skipped
        assert result.pioneer_hotkeys == frozenset()
        assert result.copier_hotkeys == frozenset()
        assert result.skipped_hotkeys == ["B"]

    def test_partial_metadata_processes_available(
        self, detector: PioneerDetector
    ) -> None:
        """Partial metadata processes hotkeys with metadata, skips others."""
        groups = [
            DuplicateGroup(hotkeys=("A", "B")),
            DuplicateGroup(hotkeys=("C", "D", "E")),
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
            "C": _create_mock_metadata("C", 300),
            # D and E are missing
        }

        result = detector.detect_pioneers(groups, metadata)

        # Group 1 processed (both have metadata)
        assert "A" in result.pioneer_hotkeys
        assert "B" in result.copier_hotkeys
        # Group 2 skipped (only C has metadata, <2 remaining)
        assert "C" not in result.pioneer_hotkeys
        assert "C" not in result.copier_hotkeys
        assert set(result.skipped_hotkeys) == {"D", "E"}

    def test_empty_metadata_with_groups_skips_all(
        self, detector: PioneerDetector
    ) -> None:
        """Empty metadata dict skips all hotkeys."""
        groups = [DuplicateGroup(hotkeys=("A", "B"))]
        metadata: dict = {}

        result = detector.detect_pioneers(groups, metadata)

        assert result.pioneer_hotkeys == frozenset()
        assert result.copier_hotkeys == frozenset()
        assert result.skipped_hotkeys == ["A", "B"]
