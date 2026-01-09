"""Unit tests for duplicate detection factory."""

from unittest.mock import MagicMock

import numpy as np

from real_estate.duplicate_detector import (
    DuplicateDetectionResult,
    DuplicateDetector,
    create_duplicate_detector,
)


def _create_mock_result(
    hotkey: str,
    predictions: np.ndarray | None = None,
    success: bool = True,
) -> MagicMock:
    """Create a mock EvaluationResult for testing."""
    result = MagicMock()
    result.hotkey = hotkey
    result.predictions = predictions
    result.success = success
    return result


def _create_mock_metadata(hotkey: str, block_number: int) -> MagicMock:
    """Create a mock ChainModelMetadata for testing."""
    metadata = MagicMock()
    metadata.hotkey = hotkey
    metadata.block_number = block_number
    return metadata


class TestCreateDuplicateDetector:
    """Tests for create_duplicate_detector factory."""

    def test_returns_duplicate_detector(self) -> None:
        """Factory returns DuplicateDetector instance."""
        detector = create_duplicate_detector()
        assert isinstance(detector, DuplicateDetector)

    def test_default_threshold(self) -> None:
        """Default threshold is 1e-6."""
        detector = create_duplicate_detector()
        # Verify by testing behavior - exact match should detect duplicates
        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
        }
        result = detector.detect(results, metadata)
        assert len(result.groups) == 1

    def test_custom_threshold(self) -> None:
        """Custom threshold is applied."""
        # Loose threshold should group slightly different predictions
        detector = create_duplicate_detector(similarity_threshold=1e-1)

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.05, 200.05])),  # Within 0.1
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
        }
        result = detector.detect(results, metadata)
        assert len(result.groups) == 1


class TestDuplicateDetectorDetect:
    """Tests for DuplicateDetector.detect method."""

    def test_full_pipeline_with_duplicates(self) -> None:
        """Full detection pipeline identifies duplicates and pioneers."""
        detector = create_duplicate_detector()

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),  # Duplicate of A
            _create_mock_result("C", np.array([300.0, 400.0])),  # Unique
        ]
        metadata = {
            "A": _create_mock_metadata("A", 1000),
            "B": _create_mock_metadata("B", 500),  # Earlier - pioneer
            "C": _create_mock_metadata("C", 800),
        }

        result = detector.detect(results, metadata)

        assert isinstance(result, DuplicateDetectionResult)
        assert len(result.groups) == 1
        assert set(result.groups[0].hotkeys) == {"A", "B"}
        assert "A" in result.copier_hotkeys
        assert "B" in result.pioneer_hotkeys  # Pioneer (earlier block)

    def test_no_duplicates_returns_empty_result(self) -> None:
        """No duplicates returns empty result."""
        detector = create_duplicate_detector()

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([300.0, 400.0])),
            _create_mock_result("C", np.array([500.0, 600.0])),
        ]
        metadata = {}  # Not needed since no duplicates

        result = detector.detect(results, metadata)

        assert result.groups == ()
        assert result.copier_hotkeys == frozenset()
        assert result.pioneer_hotkeys == frozenset()
        assert result.total_duplicates == 0

    def test_missing_metadata_results_in_skipped_hotkeys(self) -> None:
        """Missing metadata results in skipped hotkeys, not error."""
        detector = create_duplicate_detector()

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),
            _create_mock_result("C", np.array([100.0, 200.0])),
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
            # C is missing
        }

        result = detector.detect(results, metadata)

        assert "A" in result.pioneer_hotkeys
        assert "B" in result.copier_hotkeys
        assert "C" not in result.pioneer_hotkeys
        assert "C" not in result.copier_hotkeys
        assert result.skipped_hotkeys == ("C",)

    def test_multiple_duplicate_groups(self) -> None:
        """Multiple duplicate groups are all detected."""
        detector = create_duplicate_detector()

        results = [
            _create_mock_result("A", np.array([100.0])),
            _create_mock_result("B", np.array([100.0])),  # Group 1
            _create_mock_result("C", np.array([200.0])),
            _create_mock_result("D", np.array([200.0])),  # Group 2
        ]
        metadata = {
            "A": _create_mock_metadata("A", 100),  # Pioneer group 1
            "B": _create_mock_metadata("B", 200),
            "C": _create_mock_metadata("C", 400),
            "D": _create_mock_metadata("D", 300),  # Pioneer group 2
        }

        result = detector.detect(results, metadata)

        assert len(result.groups) == 2
        assert result.total_duplicates == 4
        assert len(result.pioneer_hotkeys) == 2
        assert len(result.copier_hotkeys) == 2


