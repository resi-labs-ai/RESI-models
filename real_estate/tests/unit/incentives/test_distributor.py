"""Tests for IncentiveDistributor."""

import numpy as np
import pytest

from real_estate.evaluation.models import EvaluationResult, PredictionMetrics
from real_estate.incentives import (
    DistributorConfig,
    IncentiveDistributor,
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
    """Create test evaluation result with specified score."""
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


class TestIncentiveDistributor:
    """Tests for IncentiveDistributor."""

    def test_single_winner_gets_99_percent(self):
        """Single winner with no non-winners gets 99%."""
        results = [make_result("winner", score=0.90)]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.winner_hotkey == "winner"
        assert weights.get_weight("winner") == pytest.approx(0.99)
        assert weights.total == pytest.approx(0.99)

    def test_winner_and_one_non_winner(self):
        """Winner gets 99%, single non-winner gets 1%."""
        results = [
            make_result("winner", score=0.90),
            make_result("runner_up", score=0.80),
        ]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.get_weight("winner") == pytest.approx(0.99)
        assert weights.get_weight("runner_up") == pytest.approx(0.01)
        assert weights.total == pytest.approx(1.0)

    def test_non_winners_share_proportionally(self):
        """Non-winners share 1% proportionally by score."""
        results = [
            make_result("winner", score=0.90),
            make_result("non_winner_a", score=0.60),  # 60% of non-winner total
            make_result("non_winner_b", score=0.40),  # 40% of non-winner total
        ]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.get_weight("winner") == pytest.approx(0.99)
        # Non-winners split 1% proportionally (0.6/1.0 and 0.4/1.0)
        assert weights.get_weight("non_winner_a") == pytest.approx(0.006)  # 0.01 * 0.6
        assert weights.get_weight("non_winner_b") == pytest.approx(0.004)  # 0.01 * 0.4
        assert weights.total == pytest.approx(1.0)

    def test_cheaters_get_zero(self):
        """Cheaters get 0% weight."""
        results = [
            make_result("winner", score=0.90),
            make_result("cheater_a", score=0.85),
            make_result("cheater_b", score=0.80),
            make_result("honest", score=0.70),
        ]
        cheaters = frozenset(["cheater_a", "cheater_b"])

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
            cheater_hotkeys=cheaters,
        )

        assert weights.get_weight("winner") == pytest.approx(0.99)
        assert weights.get_weight("honest") == pytest.approx(0.01)
        assert weights.get_weight("cheater_a") == 0.0
        assert weights.get_weight("cheater_b") == 0.0
        assert weights.total == pytest.approx(1.0)

    def test_failed_results_excluded(self):
        """Failed results don't participate in distribution."""
        results = [
            make_result("winner", score=0.90),
            make_failed_result("failed"),
            make_result("honest", score=0.80),
        ]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert "failed" not in weights.weights
        assert weights.get_weight("winner") == pytest.approx(0.99)
        assert weights.get_weight("honest") == pytest.approx(0.01)

    def test_custom_distribution_config(self):
        """Custom distribution config changes shares."""
        results = [
            make_result("winner", score=0.90),
            make_result("runner_up", score=0.80),
        ]

        config = DistributorConfig(winner_share=0.50)
        distributor = IncentiveDistributor(config)
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.get_weight("winner") == pytest.approx(0.50)
        assert weights.get_weight("runner_up") == pytest.approx(0.50)
        assert weights.total == pytest.approx(1.0)

    def test_all_non_winners_zero_score(self):
        """Non-winners with 0 scores get 0 weight (Pylon normalizes)."""
        results = [
            make_result("winner", score=0.90),
            make_result("zero_a", score=0.0),
            make_result("zero_b", score=0.0),
        ]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.get_weight("winner") == pytest.approx(0.99)
        assert weights.get_weight("zero_a") == 0.0
        assert weights.get_weight("zero_b") == 0.0

    def test_weights_to_dict(self):
        """IncentiveWeights serializes correctly."""
        results = [
            make_result("winner", score=0.90),
            make_result("runner_up", score=0.80),
        ]

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )
        weights_dict = weights.to_dict()

        assert weights_dict["winner_hotkey"] == "winner"
        assert weights_dict["winner_score"] == pytest.approx(0.90, abs=1e-5)
        assert weights_dict["total"] == pytest.approx(1.0)
        assert "winner" in weights_dict["weights"]
        assert "runner_up" in weights_dict["weights"]

    def test_properties(self):
        """Distributor properties return config values."""
        config = DistributorConfig(winner_share=0.80)
        distributor = IncentiveDistributor(config)

        assert distributor.winner_share == 0.80
        assert distributor.non_winner_share == pytest.approx(0.20)

    def test_empty_cheaters_set(self):
        """None and empty set for cheaters work the same."""
        results = [
            make_result("winner", score=0.90),
            make_result("runner_up", score=0.80),
        ]

        distributor = IncentiveDistributor()

        weights_none = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
            cheater_hotkeys=None,
        )

        weights_empty = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
            cheater_hotkeys=frozenset(),
        )

        assert weights_none.weights == weights_empty.weights

    def test_many_non_winners_proportional(self):
        """Many non-winners split 1% proportionally."""
        results = [make_result("winner", score=0.90)]
        results.extend([make_result(f"nw_{i}", score=0.10) for i in range(10)])

        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=results,
            winner_hotkey="winner",
            winner_score=0.90,
        )

        assert weights.get_weight("winner") == pytest.approx(0.99)
        # 10 non-winners with equal score (0.10 each) share 1% equally
        for i in range(10):
            assert weights.get_weight(f"nw_{i}") == pytest.approx(0.001)
        assert weights.total == pytest.approx(1.0)
