"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from real_estate.evaluation import (
    EmptyDatasetError,
    MetricsConfig,
    MetricsError,
    calculate_metrics,
    mape_to_score,
    score_to_mape,
    validate_predictions,
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have 0 error metrics and score 1.0."""
        y_true = np.array([100_000, 200_000, 500_000])
        y_pred = np.array([100_000, 200_000, 500_000])

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert metrics.mae == 0.0
        assert metrics.mape == 0.0
        assert metrics.rmse == 0.0
        assert metrics.mdape == 0.0
        assert metrics.r2 == 1.0
        assert metrics.accuracy[0.05] == 1.0
        assert metrics.accuracy[0.10] == 1.0
        assert metrics.accuracy[0.15] == 1.0
        assert metrics.score == 1.0

    def test_mae_calculation(self) -> None:
        """MAE should be mean of absolute errors in dollars."""
        y_true = np.array([200_000, 500_000, 1_000_000])
        y_pred = np.array([210_000, 450_000, 1_100_000])
        # Errors: 10K, 50K, 100K -> MAE = 53,333.33

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert pytest.approx(metrics.mae, rel=0.01) == 53_333.33

    def test_mape_calculation(self) -> None:
        """MAPE should be mean of percentage errors."""
        y_true = np.array([200_000, 500_000, 1_000_000])
        y_pred = np.array([210_000, 450_000, 1_100_000])
        # Pct errors: 5%, 10%, 10% -> MAPE = 8.33% = 0.0833

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert pytest.approx(metrics.mape, rel=0.01) == 0.0833

    def test_mape_no_capping_by_default(self) -> None:
        """MAPE should not cap errors by default."""
        y_true = np.array([100_000, 100_000])
        y_pred = np.array([100_000, 500_000])  # 0% and 400% error

        # Without capping, MAPE = (0 + 4.0) / 2 = 2.0
        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert pytest.approx(metrics.mape, rel=0.01) == 2.0

    def test_mape_capping_when_enabled(self) -> None:
        """MAPE should cap extreme errors when max_pct_error is set."""
        y_true = np.array([100_000, 100_000])
        y_pred = np.array([100_000, 500_000])  # 0% and 400% error

        config = MetricsConfig(max_pct_error=2.0)  # Cap at 200%
        metrics = calculate_metrics(y_true, y_pred, config=config)

        # With 200% cap, MAPE = (0 + 2.0) / 2 = 1.0
        assert pytest.approx(metrics.mape, rel=0.01) == 1.0

    def test_custom_mape_cap(self) -> None:
        """Custom max_pct_error should be respected."""
        y_true = np.array([100_000, 100_000])
        y_pred = np.array([100_000, 500_000])  # 0% and 400% error

        config = MetricsConfig(max_pct_error=1.0)  # Cap at 100%
        metrics = calculate_metrics(y_true, y_pred, config=config)

        # With 100% cap, MAPE = (0 + 1.0) / 2 = 0.5
        assert pytest.approx(metrics.mape, rel=0.01) == 0.5

    def test_rmse_calculation(self) -> None:
        """RMSE should penalize large errors more than MAE."""
        y_true = np.array([200_000, 500_000, 1_000_000])
        y_pred = np.array([210_000, 450_000, 1_100_000])
        # Errors: 10K, 50K, 100K
        # MSE = (100M + 2.5B + 10B) / 3 = 4.2B
        # RMSE = sqrt(4.2B) ≈ 64,807

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert metrics.rmse > metrics.mae  # RMSE >= MAE always
        assert pytest.approx(metrics.rmse, rel=0.01) == 64_807.41

    def test_mdape_calculation(self) -> None:
        """MdAPE should be median of percentage errors."""
        y_true = np.array([100_000, 200_000, 500_000, 1_000_000, 1_000_000])
        y_pred = np.array([105_000, 220_000, 450_000, 900_000, 2_000_000])
        # Pct errors: 5%, 10%, 10%, 10%, 100%
        # Median = 10% = 0.10

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert metrics.mdape == 0.10

    def test_r2_calculation(self) -> None:
        """R² should measure explained variance."""
        y_true = np.array([100_000, 200_000, 300_000, 400_000, 500_000])
        y_pred = np.array([110_000, 190_000, 310_000, 390_000, 510_000])
        # Small errors around true values should give high R²

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert metrics.r2 > 0.95  # Should be very close to 1

    def test_r2_negative_for_bad_model(self) -> None:
        """R² can be negative if model is worse than mean prediction."""
        y_true = np.array([100_000, 200_000, 300_000])
        # Predictions that are systematically wrong
        y_pred = np.array([500_000, 500_000, 100_000])

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert metrics.r2 < 0  # Worse than predicting mean

    def test_accuracy_thresholds(self) -> None:
        """Accuracy at different thresholds should be correctly calculated."""
        y_true = np.array([100_000] * 10)
        # Predictions with various error percentages
        y_pred = np.array([
            100_000,  # 0% - within 5%, 10%, 15%
            103_000,  # 3% - within 5%, 10%, 15%
            105_000,  # 5% - NOT within 5% (strict <), within 10%, 15%
            107_000,  # 7% - within 10%, 15%
            110_000,  # 10% - NOT within 10% (strict <), within 15%
            112_000,  # 12% - within 15%
            114_000,  # 14% - within 15%
            116_000,  # 16% - NOT within 15%
            120_000,  # 20% - outside all
            130_000,  # 30% - outside all
        ])

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        # Within 5%: 0%, 3% = 2/10 = 0.2
        assert metrics.accuracy[0.05] == 0.2
        # Within 10%: 0%, 3%, 5%, 7% = 4/10 = 0.4
        assert metrics.accuracy[0.10] == 0.4
        # Within 15%: 0%, 3%, 5%, 7%, 10%, 12%, 14% = 7/10 = 0.7
        assert metrics.accuracy[0.15] == 0.7

    def test_custom_accuracy_thresholds(self) -> None:
        """Custom accuracy thresholds should be respected."""
        y_true = np.array([100_000] * 4)
        y_pred = np.array([100_000, 102_000, 108_000, 125_000])
        # Errors: 0%, 2%, 8%, 25%

        config = MetricsConfig(accuracy_thresholds=(0.03, 0.10, 0.20))
        metrics = calculate_metrics(y_true, y_pred, config=config)

        # Within 3%: 0%, 2% = 2/4 = 0.5
        assert metrics.accuracy[0.03] == 0.5
        # Within 10%: 0%, 2%, 8% = 3/4 = 0.75
        assert metrics.accuracy[0.10] == 0.75
        # Within 20%: 0%, 2%, 8% = 3/4 = 0.75 (25% is outside)
        assert metrics.accuracy[0.20] == 0.75
        # Default thresholds should not be present
        assert 0.05 not in metrics.accuracy
        assert 0.15 not in metrics.accuracy

    def test_score_from_mape(self) -> None:
        """Score should be 1 - MAPE."""
        y_true = np.array([100_000, 200_000])
        y_pred = np.array([110_000, 220_000])  # 10% error

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())

        assert pytest.approx(metrics.mape, rel=0.01) == 0.10
        assert pytest.approx(metrics.score, rel=0.01) == 0.90

    def test_to_dict_serialization(self) -> None:
        """to_dict should return serializable dictionary."""
        y_true = np.array([100_000, 200_000, 300_000])
        y_pred = np.array([110_000, 190_000, 310_000])

        metrics = calculate_metrics(y_true, y_pred, MetricsConfig())
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert "mae" in result
        assert "mape" in result
        assert "rmse" in result
        assert "mdape" in result
        assert "r2" in result
        assert "accuracy" in result
        assert isinstance(result["accuracy"], dict)
        assert "5%" in result["accuracy"]
        assert "10%" in result["accuracy"]
        assert "15%" in result["accuracy"]
        assert "n_samples" in result
        assert result["n_samples"] == 3


class TestCalculateMetricsErrors:
    """Tests for error handling in calculate_metrics."""

    def test_length_mismatch_raises_error(self) -> None:
        """Mismatched array lengths should raise MetricsError."""
        y_true = np.array([100_000, 200_000])
        y_pred = np.array([100_000])

        with pytest.raises(MetricsError, match="length mismatch"):
            calculate_metrics(y_true, y_pred, MetricsConfig())

    def test_empty_arrays_raise_error(self) -> None:
        """Empty arrays should raise EmptyDatasetError."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(EmptyDatasetError):
            calculate_metrics(y_true, y_pred, MetricsConfig())


class TestMapeToScore:
    """Tests for MAPE to score conversion."""

    def test_perfect_mape(self) -> None:
        """MAPE of 0 should give score of 1.0."""
        assert mape_to_score(0.0) == 1.0

    def test_ten_percent_mape(self) -> None:
        """MAPE of 0.10 (10%) should give score of 0.90."""
        assert mape_to_score(0.10) == 0.90

    def test_fifty_percent_mape(self) -> None:
        """MAPE of 0.50 (50%) should give score of 0.50."""
        assert mape_to_score(0.50) == 0.50

    def test_hundred_percent_mape(self) -> None:
        """MAPE of 1.0 (100%) should give score of 0."""
        assert mape_to_score(1.0) == 0.0

    def test_over_hundred_percent_mape(self) -> None:
        """MAPE over 100% should be clamped to score of 0."""
        assert mape_to_score(1.5) == 0.0
        assert mape_to_score(2.0) == 0.0


class TestScoreToMape:
    """Tests for score to MAPE conversion."""

    def test_perfect_score(self) -> None:
        """Score of 1.0 should give MAPE of 0."""
        assert score_to_mape(1.0) == 0.0

    def test_ninety_score(self) -> None:
        """Score of 0.90 should give MAPE of 0.10."""
        assert pytest.approx(score_to_mape(0.90)) == 0.10

    def test_roundtrip(self) -> None:
        """Converting MAPE to score and back should return original."""
        for mape in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]:
            assert pytest.approx(score_to_mape(mape_to_score(mape))) == mape


class TestValidatePredictions:
    """Tests for prediction validation."""

    def test_valid_1d_array(self) -> None:
        """Valid 1D array should pass validation."""
        predictions = np.array([100_000, 200_000, 300_000])
        result = validate_predictions(predictions)

        assert np.array_equal(result, predictions)

    def test_valid_column_vector(self) -> None:
        """Column vector (N,1) should be flattened."""
        predictions = np.array([[100_000], [200_000], [300_000]])
        result = validate_predictions(predictions)

        assert result.shape == (3,)
        assert np.array_equal(result, [100_000, 200_000, 300_000])

    def test_expected_length_match(self) -> None:
        """Correct length should pass validation."""
        predictions = np.array([100_000, 200_000, 300_000])
        result = validate_predictions(predictions, expected_length=3)

        assert len(result) == 3

    def test_expected_length_mismatch(self) -> None:
        """Wrong length should raise MetricsError."""
        predictions = np.array([100_000, 200_000])

        with pytest.raises(MetricsError, match="count mismatch"):
            validate_predictions(predictions, expected_length=5)

    def test_nan_values_raise_error(self) -> None:
        """NaN values should raise MetricsError."""
        predictions = np.array([100_000, np.nan, 300_000])

        with pytest.raises(MetricsError, match="NaN"):
            validate_predictions(predictions)

    def test_inf_values_raise_error(self) -> None:
        """Inf values should raise MetricsError."""
        predictions = np.array([100_000, np.inf, 300_000])

        with pytest.raises(MetricsError, match="Inf"):
            validate_predictions(predictions)

    def test_negative_values_allowed(self) -> None:
        """Negative values are allowed (metrics will penalize them)."""
        predictions = np.array([100_000, -50_000, 300_000])

        result = validate_predictions(predictions)

        assert np.array_equal(result, predictions)

    def test_invalid_shape_raises_error(self) -> None:
        """2D array with multiple columns should raise MetricsError."""
        predictions = np.array([[100_000, 200_000], [300_000, 400_000]])

        with pytest.raises(MetricsError, match="Invalid prediction shape"):
            validate_predictions(predictions)
