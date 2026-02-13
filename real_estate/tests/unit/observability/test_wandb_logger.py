"""Tests for WandB logger."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from real_estate.observability import WandbConfig, WandbLogger, create_wandb_logger


def create_mock_eval_result(
    hotkey: str,
    score: float,
    success: bool = True,
    mape: float = 0.10,
    predictions: np.ndarray | None = None,
) -> MagicMock:
    """Helper to create mock evaluation result."""
    result = MagicMock()
    result.hotkey = hotkey
    result.score = score
    result.success = success
    result.predictions = predictions
    result.model_hash = f"hash-{hotkey[:8]}"
    result.inference_time_ms = 1500.0
    result.error_message = None if success else "ModelError"

    if success:
        result.metrics = MagicMock()
        result.metrics.mape = mape
        result.metrics.mae = 25000.0
        result.metrics.rmse = 35000.0
        result.metrics.r2 = 0.85
        result.metrics.accuracy = {0.10: 0.75}
    else:
        result.metrics = None

    return result


def create_mock_validation_result(
    miners: list[tuple[str, float, bool]],  # (hotkey, score, success)
    winner_hotkey: str,
    copier_hotkeys: frozenset[str] | None = None,
    predictions: dict[str, np.ndarray] | None = None,
) -> MagicMock:
    """Helper to create mock validation result."""
    eval_results = []
    for hotkey, score, success in miners:
        preds = predictions.get(hotkey) if predictions else None
        eval_results.append(create_mock_eval_result(hotkey, score, success, predictions=preds))

    eval_batch = MagicMock()
    eval_batch.results = eval_results
    eval_batch.dataset_size = 100
    eval_batch.successful_count = sum(1 for _, _, s in miners if s)
    eval_batch.failed_count = sum(1 for _, _, s in miners if not s)
    eval_batch.total_time_ms = 5000.0

    winner = MagicMock()
    winner.winner_hotkey = winner_hotkey
    winner.winner_score = next(s for h, s, _ in miners if h == winner_hotkey)
    winner.winner_block = 12345

    duplicate_result = MagicMock()
    duplicate_result.copier_hotkeys = copier_hotkeys or frozenset()
    duplicate_result.groups = []

    weights = MagicMock()
    weights.weights = {h: 0.5 for h, _, _ in miners}

    result = MagicMock()
    result.eval_batch = eval_batch
    result.winner = winner
    result.duplicate_result = duplicate_result
    result.weights = weights

    return result


def create_mock_dataset(n_properties: int = 10) -> MagicMock:
    """Helper to create mock validation dataset."""
    properties = [
        {"zpid": f"zpid-{i}", "price": 300000.0 + i * 10000}
        for i in range(n_properties)
    ]

    dataset = MagicMock()
    dataset.properties = properties
    dataset.ground_truth = [p["price"] for p in properties]
    dataset.__len__ = lambda self: len(properties)

    return dataset


class TestWandbLoggerDisabledBehavior:
    """Tests for disabled logger behavior."""

    def test_start_run_does_nothing_when_disabled(self) -> None:
        """Test start_run is no-op when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config, "5FTest", 46)

        logger.start_run()

        assert logger._run is None

    def test_log_evaluation_does_nothing_when_disabled(self) -> None:
        """Test log_evaluation is no-op when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config, "5FTest", 46)

        # Should not raise
        logger.log_evaluation(MagicMock(), MagicMock())

    def test_log_evaluation_warns_when_not_started(self) -> None:
        """Test log_evaluation warns when run not started."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config, "5FTest", 46)

        # Should not raise, just warn (no run started)
        logger.log_evaluation(MagicMock(), MagicMock())

        assert logger._run is None


class TestWandbLoggerStartRun:
    """Tests for start_run method."""

    def test_start_run_calls_wandb_init_with_config(self) -> None:
        """Test that start_run calls wandb.init with correct parameters."""
        config = WandbConfig(
            project="test-project",
            entity="test-entity",
            enabled=True,
        )
        logger = WandbLogger(config, "5FValidatorHotkey123", 46)

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.url = "https://wandb.ai/test"
        mock_wandb.init.return_value = mock_run
        logger._wandb = mock_wandb

        logger.start_run(run_name="my-custom-run")

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["entity"] == "test-entity"
        assert call_kwargs["name"] == "my-custom-run"
        assert "netuid-46" in call_kwargs["tags"]
        assert "validator-5FValida" in call_kwargs["tags"]

    def test_start_run_generates_run_name_if_not_provided(self) -> None:
        """Test that start_run generates a run name if not provided."""
        config = WandbConfig(project="test-project", enabled=True)
        logger = WandbLogger(config, "5FValidatorHotkey123", 46)

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        logger._wandb = mock_wandb

        logger.start_run()

        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["name"].startswith("validator-5FValida-")


class TestWandbLoggerBuildEvaluationLog:
    """Tests for _build_evaluation_log method."""

    @pytest.fixture
    def logger(self) -> WandbLogger:
        """Create test logger."""
        config = WandbConfig(enabled=True)
        return WandbLogger(config, "5FValidatorHotkey123", 46)

    def test_builds_log_with_correct_metrics(self, logger: WandbLogger) -> None:
        """Test that evaluation log contains correct aggregated metrics."""
        result = create_mock_validation_result(
            miners=[
                ("5FWinner", 0.90, True),
                ("5FLoser", 0.0, False),
            ],
            winner_hotkey="5FWinner",
        )

        log = logger._build_evaluation_log(result)

        assert log.validator_hotkey == "5FValidatorHotkey123"
        assert log.netuid == 46
        assert log.dataset_size == 100
        assert log.models_evaluated == 2
        assert log.models_succeeded == 1
        assert log.models_failed == 1
        assert log.winner_hotkey == "5FWinner"
        assert log.winner_score == 0.90

    def test_builds_miner_results_with_winner_flag(self, logger: WandbLogger) -> None:
        """Test that winner is correctly flagged in miner results."""
        result = create_mock_validation_result(
            miners=[
                ("5FWinner", 0.90, True),
                ("5FSecond", 0.85, True),
            ],
            winner_hotkey="5FWinner",
        )

        log = logger._build_evaluation_log(result)

        winner_log = next(m for m in log.miner_results if m.hotkey == "5FWinner")
        second_log = next(m for m in log.miner_results if m.hotkey == "5FSecond")

        assert winner_log.is_winner is True
        assert second_log.is_winner is False

    def test_builds_miner_results_with_copier_flag(self, logger: WandbLogger) -> None:
        """Test that copiers are correctly flagged in miner results."""
        result = create_mock_validation_result(
            miners=[
                ("5FWinner", 0.90, True),
                ("5FCopier", 0.90, True),
            ],
            winner_hotkey="5FWinner",
            copier_hotkeys=frozenset(["5FCopier"]),
        )
        result.duplicate_result.groups = [MagicMock()]

        log = logger._build_evaluation_log(result)

        assert log.copiers_detected == 1
        assert log.duplicate_groups_found == 1

        copier_log = next(m for m in log.miner_results if m.hotkey == "5FCopier")
        assert copier_log.is_copier is True

    def test_builds_failed_miner_with_error(self, logger: WandbLogger) -> None:
        """Test that failed miners include error message."""
        result = create_mock_validation_result(
            miners=[
                ("5FWinner", 0.90, True),
                ("5FFailed", 0.0, False),
            ],
            winner_hotkey="5FWinner",
        )

        log = logger._build_evaluation_log(result)

        failed_log = next(m for m in log.miner_results if m.hotkey == "5FFailed")
        assert failed_log.success is False
        assert failed_log.error == "ModelError"


class TestWandbLoggerMinerTable:
    """Tests for _log_miner_table method."""

    def test_creates_table_with_correct_columns(self) -> None:
        """Test that miner table has correct column structure."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config, "5FTest", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        from real_estate.observability.models import MinerResultLog
        miner_results = [
            MinerResultLog(hotkey="5FA", score=0.90, success=True, mape=0.10, is_winner=True),
            MinerResultLog(hotkey="5FB", score=0.85, success=True, mape=0.15, is_winner=False),
        ]

        logger._log_miner_table(miner_results)

        # Check table was created with correct columns
        mock_wandb.Table.assert_called_once()
        columns = mock_wandb.Table.call_args.kwargs["columns"]
        assert "hotkey" in columns
        assert "score" in columns
        assert "mape" in columns
        assert "is_winner" in columns
        assert "is_copier" in columns

        # Check data was added
        assert mock_table.add_data.call_count == 2

        # Check table was logged
        mock_run.log.assert_called_once()
        log_args = mock_run.log.call_args[0][0]
        assert "miner_results" in log_args


class TestWandbLoggerPredictionsTable:
    """Tests for _log_predictions_table method."""

    def test_creates_predictions_for_top_n_miners(self) -> None:
        """Test that predictions table only includes top N miners."""
        config = WandbConfig(enabled=True, predictions_top_n_miners=2)
        logger = WandbLogger(config, "5FTest", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        # 3 miners but only top 2 should be logged
        predictions = {
            "5FFirst": np.array([300000.0, 310000.0]),
            "5FSecond": np.array([305000.0, 315000.0]),
            "5FThird": np.array([350000.0, 360000.0]),
        }
        result = create_mock_validation_result(
            miners=[
                ("5FFirst", 0.95, True),
                ("5FSecond", 0.90, True),
                ("5FThird", 0.80, True),
            ],
            winner_hotkey="5FFirst",
            predictions=predictions,
        )
        # Set predictions on eval results
        for eval_result in result.eval_batch.results:
            eval_result.predictions = predictions.get(eval_result.hotkey)

        dataset = create_mock_dataset(n_properties=2)

        logger._log_predictions_table(result, dataset, "zpid")

        # Should have 2 miners x 2 properties = 4 rows
        assert mock_table.add_data.call_count == 4

    def test_logs_all_miners_when_top_n_is_none(self) -> None:
        """Test that all successful miners are logged when top_n is None."""
        config = WandbConfig(enabled=True, predictions_top_n_miners=None)
        logger = WandbLogger(config, "5FTest", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        # 3 miners, all should be logged
        predictions = {
            "5FFirst": np.array([300000.0, 310000.0]),
            "5FSecond": np.array([305000.0, 315000.0]),
            "5FThird": np.array([350000.0, 360000.0]),
        }
        result = create_mock_validation_result(
            miners=[
                ("5FFirst", 0.95, True),
                ("5FSecond", 0.90, True),
                ("5FThird", 0.80, True),
            ],
            winner_hotkey="5FFirst",
            predictions=predictions,
        )
        for eval_result in result.eval_batch.results:
            eval_result.predictions = predictions.get(eval_result.hotkey)

        dataset = create_mock_dataset(n_properties=2)

        logger._log_predictions_table(result, dataset, "zpid")

        # Should have 3 miners x 2 properties = 6 rows
        assert mock_table.add_data.call_count == 6

    def test_skips_miners_without_predictions(self) -> None:
        """Test that miners with no predictions are skipped."""
        config = WandbConfig(enabled=True, predictions_top_n_miners=None)
        logger = WandbLogger(config, "5FTest", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[
                ("5FWithPreds", 0.90, True),
                ("5FNoPreds", 0.85, True),
            ],
            winner_hotkey="5FWithPreds",
        )
        # Only first miner has predictions
        result.eval_batch.results[0].predictions = np.array([300000.0, 310000.0])
        result.eval_batch.results[1].predictions = None

        dataset = create_mock_dataset(n_properties=2)

        logger._log_predictions_table(result, dataset, "zpid")

        # Should have 1 miner x 2 properties = 2 rows
        assert mock_table.add_data.call_count == 2

    def test_uses_fallback_property_id(self) -> None:
        """Test that fallback property ID is used when zpid missing."""
        config = WandbConfig(enabled=True, predictions_top_n_miners=None)
        logger = WandbLogger(config, "5FTest", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[("5FMiner", 0.90, True)],
            winner_hotkey="5FMiner",
        )
        result.eval_batch.results[0].predictions = np.array([300000.0])

        # Dataset without zpid but with address
        dataset = MagicMock()
        dataset.properties = [{"address": "123 Main St", "price": 300000.0}]
        dataset.ground_truth = [300000.0]

        logger._log_predictions_table(result, dataset, "zpid")

        # Should use address as fallback
        add_data_call = mock_table.add_data.call_args[0]
        assert add_data_call[0] == "123 Main St"  # property_id


class TestWandbLoggerLogEvaluation:
    """Tests for log_evaluation full flow."""

    def test_logs_summary_miner_table_and_predictions(self) -> None:
        """Test that log_evaluation logs all three components."""
        config = WandbConfig(
            enabled=True,
            log_miner_table=True,
            log_predictions_table=True,
            predictions_top_n_miners=None,
        )
        logger = WandbLogger(config, "5FValidator", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[("5FWinner", 0.90, True)],
            winner_hotkey="5FWinner",
        )
        result.eval_batch.results[0].predictions = np.array([300000.0, 310000.0])

        dataset = create_mock_dataset(n_properties=2)

        logger.log_evaluation(result, dataset)

        # Should have logged: summary dict, miner_results table, property_predictions table
        assert mock_run.log.call_count == 3

    def test_skips_miner_table_when_disabled(self) -> None:
        """Test that miner table is skipped when disabled in config."""
        config = WandbConfig(
            enabled=True,
            log_miner_table=False,
            log_predictions_table=True,
            predictions_top_n_miners=None,
        )
        logger = WandbLogger(config, "5FValidator", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[("5FWinner", 0.90, True)],
            winner_hotkey="5FWinner",
        )
        result.eval_batch.results[0].predictions = np.array([300000.0])

        dataset = create_mock_dataset(n_properties=1)

        logger.log_evaluation(result, dataset)

        # Should have logged: summary dict, property_predictions table (no miner_results)
        assert mock_run.log.call_count == 2
        log_calls = [c[0][0] for c in mock_run.log.call_args_list]
        assert not any("miner_results" in c for c in log_calls)

    def test_handles_exception_gracefully(self) -> None:
        """Test that log_evaluation catches exceptions and doesn't raise."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config, "5FValidator", 46)

        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("WandB error")
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[("5FWinner", 0.90, True)],
            winner_hotkey="5FWinner",
        )
        dataset = create_mock_dataset(n_properties=1)

        # Should not raise
        logger.log_evaluation(result, dataset)


class TestWandbLoggerEdgeCases:
    """Tests for edge cases."""

    def test_handles_all_miners_failed(self) -> None:
        """Test handling when all miners failed evaluation."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config, "5FValidator", 46)

        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        # All miners failed - need a valid winner for the mock
        result = create_mock_validation_result(
            miners=[
                ("5FFailed1", 0.0, False),
                ("5FFailed2", 0.0, False),
            ],
            winner_hotkey="5FFailed1",  # Even though failed, still need a winner
        )

        dataset = create_mock_dataset(n_properties=2)

        # Should not raise
        logger.log_evaluation(result, dataset)

    def test_handles_empty_predictions_table(self) -> None:
        """Test handling when no miners have predictions."""
        config = WandbConfig(enabled=True, predictions_top_n_miners=None)
        logger = WandbLogger(config, "5FValidator", 46)

        mock_wandb = MagicMock()
        logger._wandb = mock_wandb

        mock_run = MagicMock()
        logger._run = mock_run

        result = create_mock_validation_result(
            miners=[("5FMiner", 0.90, True)],
            winner_hotkey="5FMiner",
        )
        # No predictions
        result.eval_batch.results[0].predictions = None

        dataset = create_mock_dataset(n_properties=2)

        # Should not raise
        logger._log_predictions_table(result, dataset, "zpid")


class TestCreateWandbLogger:
    """Tests for factory function."""

    def test_creates_logger_with_all_params(self) -> None:
        """Test factory creates logger with all parameters."""
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
        assert logger._config.offline is True
        assert logger._validator_hotkey == "5FTest"
        assert logger._netuid == 99

    def test_creates_logger_with_predictions_top_n(self) -> None:
        """Test factory wires predictions_top_n_miners."""
        logger = create_wandb_logger(
            validator_hotkey="5FTest",
            predictions_top_n_miners=5,
        )

        assert logger._config.predictions_top_n_miners == 5
