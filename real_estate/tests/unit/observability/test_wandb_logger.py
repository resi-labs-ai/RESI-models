"""Tests for WandB logger."""

from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from real_estate.observability import WandbConfig, WandbLogger, create_wandb_logger


class TestWandbLogger:
    """Tests for WandbLogger class."""

    @pytest.fixture
    def config(self) -> WandbConfig:
        """Create test config."""
        return WandbConfig(
            project="test-project",
            entity="test-entity",
            enabled=True,
        )

    @pytest.fixture
    def logger(self, config: WandbConfig) -> WandbLogger:
        """Create test logger."""
        return WandbLogger(
            config=config,
            validator_hotkey="5FValidatorHotkey123",
            netuid=46,
        )

    def test_init(self, logger: WandbLogger) -> None:
        """Test logger initialization."""
        assert logger._validator_hotkey == "5FValidatorHotkey123"
        assert logger._netuid == 46
        assert logger._run is None

    def test_is_enabled(self, logger: WandbLogger) -> None:
        """Test is_enabled property."""
        assert logger.is_enabled is True

    def test_is_enabled_disabled_config(self) -> None:
        """Test is_enabled when config disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config, "5FTest", 46)
        assert logger.is_enabled is False

    def test_is_running_not_started(self, logger: WandbLogger) -> None:
        """Test is_running when not started."""
        assert logger.is_running is False

    def test_start_run(self, logger: WandbLogger) -> None:
        """Test starting a WandB run."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.url = "https://wandb.ai/test"
        mock_wandb.init.return_value = mock_run

        # Inject mock
        logger._wandb = mock_wandb

        logger.start_run(run_name="test-run-name")

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["entity"] == "test-entity"
        assert call_kwargs["name"] == "test-run-name"
        assert "netuid-46" in call_kwargs["tags"]
        assert logger.is_running is True

    def test_start_run_disabled(self) -> None:
        """Test start_run when disabled does nothing."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config, "5FTest", 46)

        # Should not raise
        logger.start_run()

        assert logger.is_running is False

    def test_finish(self, logger: WandbLogger) -> None:
        """Test finishing a run."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        logger._wandb = mock_wandb

        logger.start_run()
        logger.finish()

        mock_run.finish.assert_called_once()
        assert logger.is_running is False

    def test_log_evaluation_not_started(self, logger: WandbLogger) -> None:
        """Test log_evaluation warns when run not started."""
        # Create mock result
        mock_result = MagicMock()

        with patch.object(logger, "_run", None):
            # Should not raise, just warn
            logger.log_evaluation(mock_result, MagicMock())

    def test_log_evaluation_disabled(self) -> None:
        """Test log_evaluation does nothing when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config, "5FTest", 46)

        # Should not raise
        logger.log_evaluation(MagicMock(), MagicMock())


class TestWandbLoggerBuildEvaluationLog:
    """Tests for _build_evaluation_log method."""

    @pytest.fixture
    def logger(self) -> WandbLogger:
        """Create test logger."""
        config = WandbConfig(enabled=True)
        return WandbLogger(config, "5FValidatorHotkey123", 46)

    @pytest.fixture
    def mock_validation_result(self) -> MagicMock:
        """Create mock validation result."""
        # Mock evaluation results
        eval_result_1 = MagicMock()
        eval_result_1.hotkey = "5FWinner"
        eval_result_1.score = 0.90
        eval_result_1.success = True
        eval_result_1.metrics = MagicMock()
        eval_result_1.metrics.mape = 0.10
        eval_result_1.metrics.mae = 25000.0
        eval_result_1.metrics.rmse = 35000.0
        eval_result_1.metrics.r2 = 0.85
        eval_result_1.metrics.accuracy = {0.10: 0.75}
        eval_result_1.model_hash = "abc123"
        eval_result_1.inference_time_ms = 1500.0
        eval_result_1.error_message = None

        eval_result_2 = MagicMock()
        eval_result_2.hotkey = "5FLoser"
        eval_result_2.score = 0.0
        eval_result_2.success = False
        eval_result_2.metrics = None
        eval_result_2.model_hash = None
        eval_result_2.inference_time_ms = None
        eval_result_2.error_message = "ModelCorruptedError: bad model"

        # Mock eval batch
        eval_batch = MagicMock()
        eval_batch.results = [eval_result_1, eval_result_2]
        eval_batch.dataset_size = 1000
        eval_batch.successful_count = 1
        eval_batch.failed_count = 1
        eval_batch.total_time_ms = 5000.0

        # Mock winner
        winner = MagicMock()
        winner.winner_hotkey = "5FWinner"
        winner.winner_score = 0.90
        winner.winner_block = 12345

        # Mock duplicate result
        duplicate_result = MagicMock()
        duplicate_result.groups = []
        duplicate_result.copier_hotkeys = frozenset()

        # Mock weights
        weights = MagicMock()
        weights.weights = {"5FWinner": 0.99, "5FLoser": 0.01}

        # Assemble result
        result = MagicMock()
        result.eval_batch = eval_batch
        result.winner = winner
        result.duplicate_result = duplicate_result
        result.weights = weights

        return result

    def test_build_evaluation_log(
        self, logger: WandbLogger, mock_validation_result: MagicMock
    ) -> None:
        """Test building evaluation log from validation result."""
        log = logger._build_evaluation_log(mock_validation_result)

        assert log.validator_hotkey == "5FValidatorHotkey123"
        assert log.netuid == 46
        assert log.dataset_size == 1000
        assert log.models_evaluated == 2
        assert log.models_succeeded == 1
        assert log.models_failed == 1
        assert log.winner_hotkey == "5FWinner"
        assert log.winner_score == 0.90
        assert log.winner_mape == 0.10
        assert log.winner_block == 12345

    def test_build_evaluation_log_miner_results(
        self, logger: WandbLogger, mock_validation_result: MagicMock
    ) -> None:
        """Test that miner results are correctly built."""
        log = logger._build_evaluation_log(mock_validation_result)

        assert len(log.miner_results) == 2

        # Winner result
        winner_log = next(m for m in log.miner_results if m.hotkey == "5FWinner")
        assert winner_log.success is True
        assert winner_log.score == 0.90
        assert winner_log.is_winner is True
        assert winner_log.is_copier is False

        # Failed result
        failed_log = next(m for m in log.miner_results if m.hotkey == "5FLoser")
        assert failed_log.success is False
        assert failed_log.error == "ModelCorruptedError: bad model"
        assert failed_log.is_winner is False

    def test_build_evaluation_log_with_copiers(
        self, logger: WandbLogger, mock_validation_result: MagicMock
    ) -> None:
        """Test that copiers are correctly identified."""
        mock_validation_result.duplicate_result.copier_hotkeys = frozenset(["5FLoser"])
        mock_validation_result.duplicate_result.groups = [MagicMock()]

        log = logger._build_evaluation_log(mock_validation_result)

        assert log.copiers_detected == 1
        assert log.duplicate_groups_found == 1

        copier_log = next(m for m in log.miner_results if m.hotkey == "5FLoser")
        assert copier_log.is_copier is True


class TestCreateWandbLogger:
    """Tests for create_wandb_logger factory function."""

    def test_creates_logger(self) -> None:
        """Test factory creates logger with correct config."""
        logger = create_wandb_logger(
            project="test-project",
            entity="test-entity",
            validator_hotkey="5FTest",
            netuid=99,
            enabled=True,
            offline=True,
        )

        assert isinstance(logger, WandbLogger)
        assert logger._config.project == "test-project"
        assert logger._config.entity == "test-entity"
        assert logger._config.enabled is True
        assert logger._config.offline is True
        assert logger._validator_hotkey == "5FTest"
        assert logger._netuid == 99

    def test_default_values(self) -> None:
        """Test factory with default values."""
        logger = create_wandb_logger()

        assert logger._config.project == "resi-subnet"
        assert logger._config.entity is None
        assert logger._config.enabled is True
        assert logger._config.offline is False
        assert logger._netuid == 46
