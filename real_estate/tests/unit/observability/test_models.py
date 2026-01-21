"""Tests for observability data models."""

import pytest

from real_estate.observability.models import (
    EvaluationLog,
    MinerResultLog,
    PropertyPredictionLog,
    WandbConfig,
)
from datetime import datetime, UTC


class TestMinerResultLog:
    """Tests for MinerResultLog dataclass."""

    def test_successful_miner_to_dict(self) -> None:
        """Test serialization of successful miner result."""
        log = MinerResultLog(
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            score=0.912345,
            success=True,
            mape=0.087655,
            mae=25000.5,
            rmse=35000.0,
            r2=0.85,
            accuracy_10pct=0.75,
            model_hash="abc123",
            inference_time_ms=1500.5,
            is_winner=True,
            is_copier=False,
        )

        result = log.to_dict()

        assert result["hotkey"] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        assert result["score"] == 0.912345
        assert result["success"] is True
        assert result["mape"] == 0.087655
        assert result["mae"] == 25000.5
        assert result["rmse"] == 35000.0
        assert result["r2"] == 0.85
        assert result["accuracy_10pct"] == 0.75
        assert result["model_hash"] == "abc123"
        assert result["inference_time_ms"] == 1500.5
        assert result["is_winner"] is True
        assert result["is_copier"] is False
        assert result["error"] is None

    def test_failed_miner_to_dict(self) -> None:
        """Test serialization of failed miner result."""
        log = MinerResultLog(
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            score=0.0,
            success=False,
            error="ModelCorruptedError: Invalid ONNX format",
        )

        result = log.to_dict()

        assert result["success"] is False
        assert result["error"] == "ModelCorruptedError: Invalid ONNX format"
        assert result["mape"] is None
        assert result["mae"] is None

    def test_copier_miner_to_dict(self) -> None:
        """Test serialization of copier miner result."""
        log = MinerResultLog(
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            score=0.90,
            success=True,
            mape=0.10,
            is_copier=True,
        )

        result = log.to_dict()

        assert result["is_copier"] is True
        assert result["is_winner"] is False


class TestPropertyPredictionLog:
    """Tests for PropertyPredictionLog dataclass."""

    def test_basic_prediction(self) -> None:
        """Test basic prediction log creation."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            predicted_price=500000.0,
            ground_truth_price=480000.0,
        )

        assert log.absolute_error == 20000.0
        assert log.percentage_error == pytest.approx(0.041666, rel=1e-3)

    def test_to_dict_rounding(self) -> None:
        """Test that to_dict rounds appropriately."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            predicted_price=500000.123456,
            ground_truth_price=480000.789012,
        )

        result = log.to_dict()

        assert result["predicted_price"] == 500000.12
        assert result["ground_truth_price"] == 480000.79

    def test_explicit_errors(self) -> None:
        """Test with explicitly provided error values."""
        log = PropertyPredictionLog(
            property_id="zpid-12345",
            hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            predicted_price=500000.0,
            ground_truth_price=480000.0,
            absolute_error=15000.0,  # Different from computed
            percentage_error=0.05,
        )

        # Explicit values should be used (not overwritten)
        assert log.absolute_error == 15000.0
        assert log.percentage_error == 0.05


class TestEvaluationLog:
    """Tests for EvaluationLog dataclass."""

    def test_to_summary_dict(self) -> None:
        """Test summary dict serialization."""
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
            winner_score=0.912345,
            winner_mape=0.087655,
            winner_block=12345678,
            duplicate_groups_found=3,
            copiers_detected=5,
            total_evaluation_time_ms=120000.5,
        )

        result = log.to_summary_dict()

        assert result["timestamp"] == "2025-01-15T16:00:00+00:00"
        assert result["evaluation_date"] == "2025-01-15"
        assert result["validator_hotkey"] == "5FValidator"
        assert result["netuid"] == 46
        assert result["dataset_size"] == 1000
        assert result["models_evaluated"] == 50
        assert result["models_succeeded"] == 45
        assert result["models_failed"] == 5
        assert result["winner_hotkey"] == "5FWinner"
        assert result["winner_score"] == 0.912345
        assert result["winner_mape"] == 0.087655
        assert result["winner_block"] == 12345678
        assert result["duplicate_groups_found"] == 3
        assert result["copiers_detected"] == 5
        assert result["total_evaluation_time_ms"] == 120000.5

    def test_miner_results_not_in_summary(self) -> None:
        """Test that miner_results are not in summary dict."""
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


class TestWandbConfig:
    """Tests for WandbConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = WandbConfig()

        assert config.project == "resi-subnet"
        assert config.entity is None
        assert config.enabled is True
        assert config.offline is False
        assert config.log_miner_table is True
        assert config.log_predictions_table is True
        assert config.predictions_top_n_miners == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WandbConfig(
            project="my-project",
            entity="my-team",
            enabled=False,
            offline=True,
            predictions_top_n_miners=5,
        )

        assert config.project == "my-project"
        assert config.entity == "my-team"
        assert config.enabled is False
        assert config.offline is True
        assert config.predictions_top_n_miners == 5
