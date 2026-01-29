"""Tests for observability data models."""

import pytest
from datetime import datetime, UTC

from real_estate.observability.models import (
    EvaluationLog,
    MinerResultLog,
    PropertyPredictionLog,
)


class TestPropertyPredictionLog:
    """Tests for PropertyPredictionLog computed properties."""

    def test_computes_absolute_error(self) -> None:
        """Test absolute_error is computed from prices."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FTest",
            predicted_price=500000.0,
            ground_truth_price=480000.0,
        )

        assert log.absolute_error == 20000.0

    def test_computes_percentage_error(self) -> None:
        """Test percentage_error is computed correctly."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FTest",
            predicted_price=550000.0,
            ground_truth_price=500000.0,
        )

        # (550000 - 500000) / 500000 = 0.10 = 10%
        assert log.percentage_error == pytest.approx(0.10, rel=1e-6)

    def test_percentage_error_underprediction(self) -> None:
        """Test percentage_error for underprediction (absolute value)."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FTest",
            predicted_price=450000.0,
            ground_truth_price=500000.0,
        )

        # abs(450000 - 500000) / 500000 = 0.10 = 10%
        assert log.percentage_error == pytest.approx(0.10, rel=1e-6)

    def test_explicit_errors_not_overwritten(self) -> None:
        """Test that explicitly provided error values are used."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FTest",
            predicted_price=500000.0,
            ground_truth_price=480000.0,
            absolute_error=15000.0,  # Different from computed
            percentage_error=0.05,
        )

        assert log.absolute_error == 15000.0
        assert log.percentage_error == 0.05

    def test_to_dict_rounds_prices(self) -> None:
        """Test that to_dict rounds prices to 2 decimal places."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FTest",
            predicted_price=500000.123456,
            ground_truth_price=480000.789012,
        )

        result = log.to_dict()

        assert result["predicted_price"] == 500000.12
        assert result["ground_truth_price"] == 480000.79


class TestMinerResultLog:
    """Tests for MinerResultLog serialization."""

    def test_to_dict_includes_all_metrics(self) -> None:
        """Test that to_dict includes all metric fields."""
        log = MinerResultLog(
            hotkey="5FTest",
            score=0.90,
            success=True,
            mape=0.10,
            mae=25000.0,
            rmse=35000.0,
            r2=0.85,
            accuracy=0.75,
            model_hash="abc123",
            inference_time_ms=1500.0,
            is_winner=True,
            is_copier=False,
        )

        result = log.to_dict()

        assert result["mape"] == 0.10
        assert result["mae"] == 25000.0
        assert result["rmse"] == 35000.0
        assert result["r2"] == 0.85
        assert result["accuracy"] == 0.75
        assert result["is_winner"] is True
        assert result["is_copier"] is False

    def test_to_dict_failed_miner_has_error(self) -> None:
        """Test that failed miner includes error message."""
        log = MinerResultLog(
            hotkey="5FTest",
            score=0.0,
            success=False,
            error="ModelCorruptedError: Invalid ONNX format",
        )

        result = log.to_dict()

        assert result["success"] is False
        assert result["error"] == "ModelCorruptedError: Invalid ONNX format"
        assert result["mape"] is None


class TestEvaluationLog:
    """Tests for EvaluationLog serialization."""

    def test_to_summary_dict_excludes_miner_results(self) -> None:
        """Test that miner_results are excluded from summary dict."""
        log = EvaluationLog(
            timestamp=datetime.now(UTC),
            evaluation_date="2025-01-15",
            validator_hotkey="5FValidator",
            netuid=46,
            dataset_size=1000,
            models_evaluated=50,
            models_succeeded=45,
            models_failed=5,
            winner_hotkey="5FWinner",
            winner_score=0.90,
            miner_results=[
                MinerResultLog(hotkey="5FA", score=0.90, success=True),
                MinerResultLog(hotkey="5FB", score=0.85, success=True),
            ],
        )

        result = log.to_summary_dict()

        assert "miner_results" not in result
        assert result["models_evaluated"] == 50
        assert result["winner_hotkey"] == "5FWinner"

    def test_to_summary_dict_formats_timestamp(self) -> None:
        """Test that timestamp is formatted as ISO string."""
        now = datetime(2025, 1, 15, 16, 0, 0, tzinfo=UTC)

        log = EvaluationLog(
            timestamp=now,
            evaluation_date="2025-01-15",
            validator_hotkey="5FValidator",
            netuid=46,
            dataset_size=1000,
            models_evaluated=50,
            models_succeeded=45,
            models_failed=5,
            winner_hotkey="5FWinner",
            winner_score=0.90,
        )

        result = log.to_summary_dict()

        assert result["timestamp"] == "2025-01-15T16:00:00+00:00"
