"""Unit tests for PioneerDetector."""

from unittest.mock import MagicMock

import pytest

from real_estate.duplicate_detection import (
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
        assert result.pioneers["A"] is False
        assert result.pioneers["B"] is True  # Pioneer
        assert result.pioneers["C"] is False
        assert result.skipped_hotkeys == []

    def test_single_group_one_pioneer(self, detector: PioneerDetector) -> None:
        """Each group has exactly one pioneer."""
        groups = [DuplicateGroup(hotkeys=("A", "B"))]
        metadata = {
            "A": _create_mock_metadata("A", 500),  # Pioneer
            "B": _create_mock_metadata("B", 600),
        }

        result = detector.detect_pioneers(groups, metadata)

        pioneer_count = sum(1 for is_pioneer in result.pioneers.values() if is_pioneer)
        assert pioneer_count == 1

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

        assert result.pioneers["A"] is True
        assert result.pioneers["B"] is False
        assert result.pioneers["C"] is False
        assert result.pioneers["D"] is True

    def test_tie_breaking_alphabetical(self, detector: PioneerDetector) -> None:
        """Tie on block_number uses alphabetical hotkey order."""
        groups = [DuplicateGroup(hotkeys=("C", "A", "B"))]
        metadata = {
            "A": _create_mock_metadata("A", 500),  # Same block, alphabetically first
            "B": _create_mock_metadata("B", 500),  # Same block
            "C": _create_mock_metadata("C", 500),  # Same block
        }

        result = detector.detect_pioneers(groups, metadata)

        assert result.pioneers["A"] is True  # Alphabetically first
        assert result.pioneers["B"] is False
        assert result.pioneers["C"] is False

    def test_empty_groups_returns_empty_result(self, detector: PioneerDetector) -> None:
        """Empty groups list returns empty pioneers dict."""
        result = detector.detect_pioneers([], {})
        assert result.pioneers == {}
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

        assert set(result.pioneers.keys()) == {"A", "B", "C", "D", "E"}


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
        assert result.pioneers["A"] is True  # Pioneer (lower block)
        assert result.pioneers["B"] is False
        assert "C" not in result.pioneers
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
        assert result.pioneers == {}
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
        assert result.pioneers == {}
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
        assert result.pioneers["A"] is True
        assert result.pioneers["B"] is False
        # Group 2 skipped (only C has metadata, <2 remaining)
        assert "C" not in result.pioneers
        assert set(result.skipped_hotkeys) == {"D", "E"}

    def test_empty_metadata_with_groups_skips_all(
        self, detector: PioneerDetector
    ) -> None:
        """Empty metadata dict skips all hotkeys."""
        groups = [DuplicateGroup(hotkeys=("A", "B"))]
        metadata: dict = {}

        result = detector.detect_pioneers(groups, metadata)

        assert result.pioneers == {}
        assert result.skipped_hotkeys == ["A", "B"]
