"""Tests for incentives models."""

import pytest

from real_estate.incentives import (
    DistributorConfig,
    IncentiveWeights,
    WinnerCandidate,
    WinnerSelectionResult,
)


class TestDistributorConfig:
    """Tests for DistributorConfig."""

    def test_non_winner_share_computed(self):
        """non_winner_share is computed from winner_share."""
        config = DistributorConfig(winner_share=0.50)
        assert config.non_winner_share == 0.50

        config2 = DistributorConfig(winner_share=0.80)
        assert config2.non_winner_share == pytest.approx(0.20)


class TestWinnerSelectionResult:
    """Tests for WinnerSelectionResult."""

    def test_was_tie_broken_by_commit_time(self):
        """was_tie_broken_by_commit_time indicates multiple candidates."""
        single = WinnerSelectionResult(
            winner_hotkey="a",
            winner_score=0.90,
            winner_block=1000,
            candidates=(WinnerCandidate("a", 0.90, 1000),),
            best_score=0.90,
            threshold=0.005,
        )
        assert single.was_tie_broken_by_commit_time is False

        multiple = WinnerSelectionResult(
            winner_hotkey="a",
            winner_score=0.90,
            winner_block=1000,
            candidates=(
                WinnerCandidate("a", 0.90, 1000),
                WinnerCandidate("b", 0.89, 2000),
            ),
            best_score=0.90,
            threshold=0.005,
        )
        assert multiple.was_tie_broken_by_commit_time is True

    def test_to_dict(self):
        """to_dict serializes with rounding."""
        candidates = (
            WinnerCandidate("hotkey_a", 0.9001234, 2000),
            WinnerCandidate("hotkey_b", 0.8981234, 1000),
        )
        result = WinnerSelectionResult(
            winner_hotkey="hotkey_b",
            winner_score=0.8981234,
            winner_block=1000,
            candidates=candidates,
            best_score=0.9001234,
            threshold=0.005,
        )
        result_dict = result.to_dict()

        # Verify rounding to 6 decimals
        assert result_dict["winner_score"] == 0.898123
        assert result_dict["best_score"] == 0.900123


class TestIncentiveWeights:
    """Tests for IncentiveWeights."""

    def test_get_weight_missing_returns_zero(self):
        """get_weight returns 0 for missing hotkey."""
        weights = IncentiveWeights(weights={"a": 0.99})
        assert weights.get_weight("missing") == 0.0

    def test_to_dict(self):
        """to_dict serializes with rounding."""
        weights = IncentiveWeights(
            weights={"a": 0.991234567, "b": 0.008765433},
            winner_hotkey="a",
            winner_score=0.901234567,
        )
        weights_dict = weights.to_dict()

        # Verify rounding to 6 decimals
        assert weights_dict["winner_score"] == 0.901235
        assert weights_dict["weights"]["a"] == 0.991235
        assert weights_dict["weights"]["b"] == 0.008765
