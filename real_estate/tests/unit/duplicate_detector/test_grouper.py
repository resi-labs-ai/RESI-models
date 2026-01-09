"""Unit tests for PredictionGrouper."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from real_estate.duplicate_detector import GrouperConfig, PredictionGrouper


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


class TestPredictionGrouperGroupPredictions:
    """Tests for PredictionGrouper.group_predictions method."""

    @pytest.fixture
    def grouper(self) -> PredictionGrouper:
        """Create default grouper."""
        return PredictionGrouper()

    def test_no_duplicates_returns_empty_list(self, grouper: PredictionGrouper) -> None:
        """When all predictions are unique, returns empty list."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0, 300.0])),
            _create_mock_result("B", np.array([150.0, 250.0, 350.0])),
            _create_mock_result("C", np.array([200.0, 300.0, 400.0])),
        ]
        groups = grouper.group_predictions(results)
        assert groups == []

    def test_exact_duplicates_grouped(self, grouper: PredictionGrouper) -> None:
        """Models with identical predictions are grouped together."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0, 300.0])),
            _create_mock_result("B", np.array([100.0, 200.0, 300.0])),  # Same as A
            _create_mock_result("C", np.array([150.0, 250.0, 350.0])),
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
        assert set(groups[0].hotkeys) == {"A", "B"}

    def test_multiple_duplicate_groups(self, grouper: PredictionGrouper) -> None:
        """Multiple separate duplicate groups are detected."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),  # Same as A
            _create_mock_result("C", np.array([300.0, 400.0])),
            _create_mock_result("D", np.array([300.0, 400.0])),  # Same as C
            _create_mock_result("E", np.array([500.0, 600.0])),  # Unique
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 2
        hotkey_sets = {frozenset(g.hotkeys) for g in groups}
        assert frozenset({"A", "B"}) in hotkey_sets
        assert frozenset({"C", "D"}) in hotkey_sets

    def test_three_way_duplicate(self, grouper: PredictionGrouper) -> None:
        """Three models with same predictions form one group."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),
            _create_mock_result("C", np.array([100.0, 200.0])),
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
        assert set(groups[0].hotkeys) == {"A", "B", "C"}

    def test_failed_results_ignored(self, grouper: PredictionGrouper) -> None:
        """Results without predictions (failed) are skipped."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),
            _create_mock_result("C", predictions=None, success=False),  # Failed
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
        assert set(groups[0].hotkeys) == {"A", "B"}
        assert "C" not in groups[0].hotkeys

    def test_success_false_ignored(self, grouper: PredictionGrouper) -> None:
        """Results with success=False are skipped even if predictions exist."""
        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0]), success=False),
        ]
        groups = grouper.group_predictions(results)

        # Only A is valid, no duplicates
        assert groups == []

    def test_single_model_not_grouped(self, grouper: PredictionGrouper) -> None:
        """Single model is not returned as a group."""
        results = [_create_mock_result("A", np.array([100.0, 200.0]))]
        groups = grouper.group_predictions(results)
        assert groups == []

    def test_empty_results_returns_empty_list(self, grouper: PredictionGrouper) -> None:
        """Empty input returns empty list."""
        groups = grouper.group_predictions([])
        assert groups == []

    def test_2d_predictions_flattened(self, grouper: PredictionGrouper) -> None:
        """2D prediction arrays are flattened for comparison."""
        results = [
            _create_mock_result("A", np.array([[100.0], [200.0]])),
            _create_mock_result("B", np.array([[100.0], [200.0]])),
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
        assert set(groups[0].hotkeys) == {"A", "B"}


class TestPredictionGrouperConfig:
    """Tests for GrouperConfig and threshold behavior."""

    def test_default_threshold(self) -> None:
        """Default threshold is 1e-6."""
        config = GrouperConfig()
        assert config.similarity_threshold == 1e-6

    def test_custom_threshold_applied(self) -> None:
        """Custom threshold is stored."""
        config = GrouperConfig(similarity_threshold=1e-3)
        assert config.similarity_threshold == 1e-3

    def test_near_duplicates_within_threshold(self) -> None:
        """Predictions within threshold are considered identical."""
        # 1e-3 threshold = 3 decimal places
        grouper = PredictionGrouper(GrouperConfig(similarity_threshold=1e-3))

        results = [
            _create_mock_result("A", np.array([100.0001, 200.0001])),
            _create_mock_result("B", np.array([100.0002, 200.0002])),  # Within 1e-3
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
        assert set(groups[0].hotkeys) == {"A", "B"}

    def test_near_duplicates_outside_threshold(self) -> None:
        """Predictions outside threshold are not grouped."""
        # 1e-6 threshold = 6 decimal places (stricter)
        grouper = PredictionGrouper(GrouperConfig(similarity_threshold=1e-6))

        results = [
            _create_mock_result("A", np.array([100.0001, 200.0001])),
            _create_mock_result(
                "B", np.array([100.0002, 200.0002])
            ),  # Differs at 4th decimal
        ]
        groups = grouper.group_predictions(results)

        assert groups == []  # Not considered duplicates

    def test_strict_threshold_1e9(self) -> None:
        """Very strict threshold requires very close match."""
        grouper = PredictionGrouper(GrouperConfig(similarity_threshold=1e-9))

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.0, 200.0])),  # Exact match
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1

    def test_loose_threshold_1e1(self) -> None:
        """Loose threshold groups predictions within 0.1."""
        grouper = PredictionGrouper(GrouperConfig(similarity_threshold=1e-1))

        results = [
            _create_mock_result("A", np.array([100.0, 200.0])),
            _create_mock_result("B", np.array([100.05, 200.05])),  # Within 0.1
        ]
        groups = grouper.group_predictions(results)

        assert len(groups) == 1
