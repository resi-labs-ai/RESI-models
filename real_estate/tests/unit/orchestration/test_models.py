"""Unit tests for orchestration models."""

from unittest.mock import MagicMock

import numpy as np

from real_estate.evaluation.models import EvaluationBatch, EvaluationResult, PredictionMetrics
from real_estate.orchestration.models import ValidationResult


def _create_eval_result(
    hotkey: str,
    mape: float = 0.1,
    success: bool = True,
) -> EvaluationResult:
    """Create an EvaluationResult for testing."""
    metrics = PredictionMetrics(
        mape=mape,
        mae=10000.0,
        rmse=15000.0,
        mdape=mape,
        accuracy={0.05: 0.3, 0.10: 0.6, 0.15: 0.8},
        r2=0.9,
        n_samples=10,
    ) if success else None

    return EvaluationResult(
        hotkey=hotkey,
        predictions=np.array([100.0, 200.0]) if success else None,
        metrics=metrics,
        error=None if success else Exception("Eval failed"),
        inference_time_ms=100.0,
        model_hash="abc123",
    )


class TestValidationResultToDict:
    """Tests for ValidationResult.to_dict method."""

    def test_basic_serialization(self) -> None:
        """to_dict returns expected structure."""
        # Create mocks
        mock_weights = MagicMock()
        mock_weights.weights = {"A": 0.99, "B": 0.01}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.95

        eval_results = [
            _create_eval_result("A", mape=0.05),
            _create_eval_result("B", mape=0.10),
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()

        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=eval_batch,
            duplicate_result=mock_duplicates,
        )

        output = result.to_dict()

        assert output["winner_hotkey"] == "A"
        assert output["winner_score"] == 0.95
        assert "A" in output["results"]
        assert "B" in output["results"]
        assert output["copiers"] == []
        assert output["weights"] == {"A": 0.99, "B": 0.01}

    def test_results_include_score_and_mape(self) -> None:
        """Individual results include score and mape."""
        mock_weights = MagicMock()
        mock_weights.weights = {"A": 1.0}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.9

        eval_results = [_create_eval_result("A", mape=0.1)]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()

        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=eval_batch,
            duplicate_result=mock_duplicates,
        )

        output = result.to_dict()

        assert output["results"]["A"]["score"] == 0.9  # 1 - mape
        assert output["results"]["A"]["mape"] == 0.1
        assert output["results"]["A"]["success"] is True
        assert output["results"]["A"]["error"] is None

    def test_failed_result_has_null_score_and_error(self) -> None:
        """Failed evaluation has null score and error message."""
        mock_weights = MagicMock()
        mock_weights.weights = {"A": 1.0}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.9

        eval_results = [
            _create_eval_result("A", mape=0.1),
            _create_eval_result("B", success=False),
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()

        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=eval_batch,
            duplicate_result=mock_duplicates,
        )

        output = result.to_dict()

        assert output["results"]["B"]["score"] is None
        assert output["results"]["B"]["mape"] is None
        assert output["results"]["B"]["success"] is False
        assert output["results"]["B"]["error"] == "Eval failed"

    def test_copiers_are_sorted(self) -> None:
        """Copiers list is sorted alphabetically."""
        mock_weights = MagicMock()
        mock_weights.weights = {}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.9

        eval_batch = EvaluationBatch(results=[], dataset_size=10)

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset({"C", "A", "B"})

        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=eval_batch,
            duplicate_result=mock_duplicates,
        )

        output = result.to_dict()

        assert output["copiers"] == ["A", "B", "C"]

    def test_scores_are_rounded(self) -> None:
        """Scores and weights are rounded to 6 decimal places."""
        mock_weights = MagicMock()
        mock_weights.weights = {"A": 0.123456789}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.987654321

        eval_results = [_create_eval_result("A", mape=0.123456789)]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()

        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=eval_batch,
            duplicate_result=mock_duplicates,
        )

        output = result.to_dict()

        assert output["winner_score"] == 0.987654
        assert output["weights"]["A"] == 0.123457
        assert output["results"]["A"]["mape"] == 0.123457
