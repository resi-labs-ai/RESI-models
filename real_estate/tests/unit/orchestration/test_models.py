"""Unit tests for orchestration models."""

from unittest.mock import MagicMock

from real_estate.orchestration.models import ValidationResult

from .conftest import (
    create_duplicate_result,
    create_eval_batch,
    create_eval_result,
    create_weights,
    create_winner_result,
)


class TestValidationResultToDict:
    """Tests for ValidationResult.to_dict method."""

    def test_failed_result_has_null_score_and_error(self) -> None:
        """Failed evaluation has null score and error message."""
        result = ValidationResult(
            weights=create_weights({"A": 1.0}),
            winner=create_winner_result("A", 0.9),
            eval_batch=create_eval_batch([
                create_eval_result("A", score=0.9),
                create_eval_result("B", success=False),
            ]),
            duplicate_result=create_duplicate_result(),
        )

        output = result.to_dict()

        assert output["results"]["B"]["score"] is None
        assert output["results"]["B"]["mape"] is None
        assert output["results"]["B"]["success"] is False
        assert output["results"]["B"]["error"] == "Eval failed"

    def test_copiers_are_sorted(self) -> None:
        """Copiers list is sorted alphabetically."""
        result = ValidationResult(
            weights=create_weights({}),
            winner=create_winner_result("A", 0.9),
            eval_batch=create_eval_batch([]),
            duplicate_result=create_duplicate_result(frozenset({"C", "A", "B"})),
        )

        output = result.to_dict()

        assert output["copiers"] == ["A", "B", "C"]

    def test_scores_are_rounded(self) -> None:
        """Scores and weights are rounded to 6 decimal places."""
        # Need to use MagicMock directly for precise control over values
        mock_weights = MagicMock()
        mock_weights.weights = {"A": 0.123456789}

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.987654321

        # score = 1 - mape, so mape = 1 - 0.876543211 = 0.123456789
        result = ValidationResult(
            weights=mock_weights,
            winner=mock_winner,
            eval_batch=create_eval_batch([create_eval_result("A", score=0.876543211)]),
            duplicate_result=create_duplicate_result(),
        )

        output = result.to_dict()

        assert output["winner_score"] == 0.987654
        assert output["weights"]["A"] == 0.123457
        assert output["results"]["A"]["mape"] == 0.123457
