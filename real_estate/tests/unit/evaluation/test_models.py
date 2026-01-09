"""Unit tests for evaluation models."""

import numpy as np
import pytest

from real_estate.evaluation.models import (
    EvaluationBatch,
    EvaluationResult,
    PredictionMetrics,
)


class TestPredictionMetrics:
    """Tests for PredictionMetrics dataclass."""

    def test_score_clamped_at_zero(self) -> None:
        """MAPE > 1.0 should give score of 0.0 (not negative)."""
        metrics = PredictionMetrics(
            mae=500000, mape=1.5, rmse=600000, mdape=1.2, accuracy={}, r2=-0.5, n_samples=10
        )
        assert metrics.score == 0.0

    def test_to_dict_accuracy_keys_formatted(self) -> None:
        """to_dict formats accuracy keys as percentages."""
        metrics = PredictionMetrics(
            mae=50000.0,
            mape=0.10,
            rmse=65000.0,
            mdape=0.08,
            accuracy={0.05: 0.3, 0.10: 0.6, 0.15: 0.85},
            r2=0.92,
            n_samples=100,
        )
        result = metrics.to_dict()

        assert "5%" in result["accuracy"]
        assert "10%" in result["accuracy"]
        assert "15%" in result["accuracy"]
        assert result["accuracy"]["10%"] == 0.6


class TestEvaluationBatch:
    """Tests for EvaluationBatch dataclass."""

    @pytest.fixture
    def mixed_batch(self) -> EvaluationBatch:
        """Create batch with mixed success/failure results."""
        return EvaluationBatch(
            results=[
                EvaluationResult(
                    hotkey="hotkey1",
                    predictions=np.array([100000]),
                    metrics=PredictionMetrics(
                        mae=5000, mape=0.05, rmse=6000, mdape=0.04,
                        accuracy={}, r2=0.98, n_samples=1
                    ),
                ),
                EvaluationResult(
                    hotkey="hotkey2",
                    predictions=np.array([200000]),
                    metrics=PredictionMetrics(
                        mae=20000, mape=0.10, rmse=25000, mdape=0.08,
                        accuracy={}, r2=0.90, n_samples=1
                    ),
                ),
                EvaluationResult(
                    hotkey="hotkey3",
                    error=RuntimeError("Docker failed"),
                ),
            ],
            dataset_size=100,
            total_time_ms=500.0,
        )

    def test_get_ranking(self, mixed_batch: EvaluationBatch) -> None:
        """get_ranking returns sorted by score descending."""
        ranking = mixed_batch.get_ranking()

        assert len(ranking) == 2
        assert ranking[0] == ("hotkey1", 0.95)  # 1 - 0.05
        assert ranking[1] == ("hotkey2", 0.90)  # 1 - 0.10

    def test_get_best(self, mixed_batch: EvaluationBatch) -> None:
        """get_best returns highest scoring result."""
        best = mixed_batch.get_best()

        assert best is not None
        assert best.hotkey == "hotkey1"
        assert best.score == 0.95

    def test_get_best_empty_batch(self) -> None:
        """get_best returns None for empty batch."""
        batch = EvaluationBatch()
        assert batch.get_best() is None

    def test_get_best_all_failed(self) -> None:
        """get_best returns None when all failed."""
        batch = EvaluationBatch(
            results=[
                EvaluationResult(hotkey="h1", error=Exception("fail")),
                EvaluationResult(hotkey="h2", error=Exception("fail")),
            ]
        )
        assert batch.get_best() is None
