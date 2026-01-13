"""Tests for WinnerSelector."""

import numpy as np
import pytest

from real_estate.chain.models import ChainModelMetadata
from real_estate.evaluation.models import EvaluationResult, PredictionMetrics
from real_estate.incentives import (
    NoValidModelsError,
    WinnerSelector,
)


def make_metrics(mape: float = 0.1) -> PredictionMetrics:
    """Create test metrics with specified MAPE."""
    return PredictionMetrics(
        mae=10000.0,
        mape=mape,
        rmse=15000.0,
        mdape=mape,
        accuracy={0.05: 0.4, 0.10: 0.7, 0.15: 0.85},
        r2=0.8,
        n_samples=100,
    )


def make_result(hotkey: str, score: float) -> EvaluationResult:
    """Create test evaluation result with specified score (via MAPE)."""
    # score = 1 - mape, so mape = 1 - score
    mape = 1.0 - score
    return EvaluationResult(
        hotkey=hotkey,
        predictions=np.array([100000.0] * 10),
        metrics=make_metrics(mape),
    )


def make_failed_result(hotkey: str) -> EvaluationResult:
    """Create a failed evaluation result."""
    return EvaluationResult(
        hotkey=hotkey,
        error=ValueError("Model failed"),
    )


def make_metadata(hotkey: str, block_number: int) -> ChainModelMetadata:
    """Create test chain metadata."""
    return ChainModelMetadata(
        hotkey=hotkey,
        hf_repo_id="test/model",
        model_hash="abc12345",
        block_number=block_number,
        timestamp=1700000000,
    )


class TestWinnerSelector:
    """Tests for WinnerSelector."""

    def test_select_winner_single_model(self):
        """Single model should win."""
        results = [make_result("hotkey_a", score=0.90)]
        metadata = {"hotkey_a": make_metadata("hotkey_a", block_number=1000)}

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        assert result.winner_hotkey == "hotkey_a"
        assert result.winner_score == pytest.approx(0.90, abs=1e-6)
        assert result.winner_set_size == 1
        assert not result.was_tie_broken_by_commit_time

    def test_select_winner_highest_score_wins(self):
        """Highest score wins when outside threshold."""
        results = [
            make_result("hotkey_a", score=0.90),
            make_result("hotkey_b", score=0.80),  # More than 0.005 below
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),  # Later commit
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),  # Earlier commit
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        # hotkey_a wins because score difference (0.10) > threshold (0.005)
        assert result.winner_hotkey == "hotkey_a"
        assert result.winner_set_size == 1

    def test_select_winner_within_threshold_earlier_commit_wins(self):
        """Within threshold, earlier commit wins."""
        results = [
            make_result("hotkey_a", score=0.900),
            make_result("hotkey_b", score=0.898),  # Within 0.005 threshold
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),  # Later commit
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),  # Earlier commit
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        # hotkey_b wins despite lower score (earlier commit)
        assert result.winner_hotkey == "hotkey_b"
        assert result.winner_score == pytest.approx(0.898, abs=1e-6)
        assert result.winner_set_size == 2
        assert result.was_tie_broken_by_commit_time

    def test_select_winner_exact_threshold_boundary(self):
        """Models exactly at threshold boundary are included."""
        results = [
            make_result("hotkey_a", score=0.900),
            make_result(
                "hotkey_b", score=0.895
            ),  # Exactly at threshold (0.900 - 0.005)
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        # Both should be in winner set (0.895 >= 0.895)
        assert result.winner_hotkey == "hotkey_b"
        assert result.winner_set_size == 2

    def test_select_winner_just_outside_threshold(self):
        """Models just outside threshold are excluded."""
        results = [
            make_result("hotkey_a", score=0.900),
            make_result("hotkey_b", score=0.8949),  # Just below threshold
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        # Only hotkey_a in winner set
        assert result.winner_hotkey == "hotkey_a"
        assert result.winner_set_size == 1

    def test_select_winner_custom_threshold(self):
        """Custom threshold affects winner set."""
        results = [
            make_result("hotkey_a", score=0.90),
            make_result("hotkey_b", score=0.85),  # 0.05 below
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),
        }

        # With default threshold (0.005), only hotkey_a is in winner set
        selector_default = WinnerSelector(0.005)
        result_default = selector_default.select_winner(results, metadata)
        assert result_default.winner_hotkey == "hotkey_a"
        assert result_default.winner_set_size == 1

        # With larger threshold (0.10), both are in winner set
        selector_large = WinnerSelector(0.10)
        result_large = selector_large.select_winner(results, metadata)
        assert result_large.winner_hotkey == "hotkey_b"  # Earlier commit
        assert result_large.winner_set_size == 2

    def test_select_winner_skips_failed_results(self):
        """Failed results are filtered out."""
        results = [
            make_failed_result("hotkey_a"),
            make_result("hotkey_b", score=0.80),
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=1000),
            "hotkey_b": make_metadata("hotkey_b", block_number=2000),
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        assert result.winner_hotkey == "hotkey_b"

    def test_select_winner_missing_metadata_raises(self):
        """Missing metadata raises ValueError (programming error)."""
        results = [
            make_result("hotkey_a", score=0.90),
            make_result("hotkey_b", score=0.90),
        ]
        metadata = {
            "hotkey_b": make_metadata("hotkey_b", block_number=2000),
            # hotkey_a missing from metadata - this is a bug
        }

        selector = WinnerSelector(0.005)
        with pytest.raises(ValueError, match="Missing chain metadata"):
            selector.select_winner(results, metadata)

    def test_select_winner_no_valid_results_raises(self):
        """No valid results raises NoValidModelsError."""
        results = [make_failed_result("hotkey_a")]
        metadata = {"hotkey_a": make_metadata("hotkey_a", block_number=1000)}

        selector = WinnerSelector(0.005)
        with pytest.raises(NoValidModelsError, match="No successful evaluation"):
            selector.select_winner(results, metadata)

    def test_select_winner_empty_results_raises(self):
        """Empty results raises NoValidModelsError."""
        selector = WinnerSelector(0.005)
        with pytest.raises(NoValidModelsError, match="No successful evaluation"):
            selector.select_winner([], {})

    def test_select_winner_all_missing_metadata_raises(self):
        """All models missing metadata raises ValueError (programming error)."""
        results = [
            make_result("hotkey_a", score=0.90),
            make_result("hotkey_b", score=0.85),
        ]
        metadata = {}  # No metadata - this is a bug

        selector = WinnerSelector(0.005)
        with pytest.raises(ValueError, match="Missing chain metadata"):
            selector.select_winner(results, metadata)

    def test_result_to_dict(self):
        """WinnerSelectionResult serializes correctly."""
        results = [
            make_result("hotkey_a", score=0.90),
            make_result("hotkey_b", score=0.898),
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=2000),
            "hotkey_b": make_metadata("hotkey_b", block_number=1000),
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)
        result_dict = result.to_dict()

        assert result_dict["winner_hotkey"] == "hotkey_b"
        assert result_dict["winner_set_size"] == 2
        assert result_dict["tie_broken_by_commit_time"] is True
        assert len(result_dict["candidates"]) == 2

    def test_threshold_property(self):
        """Threshold property returns configured value."""
        selector = WinnerSelector(0.01)
        assert selector.threshold == 0.01

    def test_many_candidates_in_winner_set(self):
        """Multiple candidates within threshold compete by commit time."""
        results = [
            make_result("hotkey_a", score=0.900),
            make_result("hotkey_b", score=0.899),
            make_result("hotkey_c", score=0.898),
            make_result("hotkey_d", score=0.897),
            make_result("hotkey_e", score=0.896),  # All within 0.005
        ]
        metadata = {
            "hotkey_a": make_metadata("hotkey_a", block_number=5000),
            "hotkey_b": make_metadata("hotkey_b", block_number=4000),
            "hotkey_c": make_metadata("hotkey_c", block_number=1000),  # Earliest
            "hotkey_d": make_metadata("hotkey_d", block_number=3000),
            "hotkey_e": make_metadata("hotkey_e", block_number=2000),
        }

        selector = WinnerSelector(0.005)
        result = selector.select_winner(results, metadata)

        assert result.winner_hotkey == "hotkey_c"  # Earliest commit
        assert result.winner_set_size == 5
