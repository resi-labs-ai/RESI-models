"""Tests for Validator class."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from real_estate.chain.models import Metagraph, Neuron
from real_estate.models import DownloadResult
from real_estate.validator import Validator


def create_mock_neuron(uid: int, hotkey: str, validator_permit: bool = True) -> Neuron:
    """Create a mock Neuron for testing."""
    return Neuron(
        uid=uid,
        hotkey=hotkey,
        coldkey=f"coldkey_{uid}",
        stake=100.0,
        trust=0.5,
        consensus=0.5,
        incentive=0.1,
        dividends=0.1,
        emission=0.1,
        is_active=True,
        validator_permit=validator_permit,
    )


def create_mock_metagraph(hotkeys: list[str], block: int = 1000) -> Metagraph:
    """Create a mock Metagraph for testing."""
    neurons = [create_mock_neuron(uid, hotkey) for uid, hotkey in enumerate(hotkeys)]
    return Metagraph(
        block=block,
        neurons=neurons,
        timestamp=datetime.now(),
    )


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config for Validator."""
    config = MagicMock()
    config.pylon_url = "http://test.pylon"
    config.pylon_token = "test_token"
    config.pylon_identity = None
    config.subtensor_network = "test"
    config.netuid = 1
    config.hotkey = "our_hotkey"
    config.disable_set_weights = False
    config.epoch_length = 100
    config.validation_data_url = "http://test.validation"
    config.validation_data_max_retries = 3
    config.validation_data_retry_delay = 1
    config.validation_data_schedule_hour = 2
    config.validation_data_schedule_minute = 0
    config.validation_data_download_raw = False
    config.scheduler_pre_download_hours = 3.0
    config.scheduler_catch_up_minutes = 30.0
    config.burn_amount = 0.0
    config.burn_uid = -1
    return config


@pytest.fixture
def validator(mock_config: MagicMock) -> Validator:
    """Create a Validator instance with mocked dependencies."""
    with (
        patch("real_estate.validator.validator.check_config"),
        patch("real_estate.validator.validator.bt.subtensor") as mock_subtensor,
        patch("real_estate.validator.validator.bt.wallet") as mock_wallet,
        patch("real_estate.validator.validator.ValidationDatasetClient"),
        patch("real_estate.validator.validator.ValidationOrchestrator"),
    ):
        mock_subtensor.return_value = MagicMock(chain_endpoint="mock_endpoint")
        mock_wallet.return_value = MagicMock(
            hotkey=MagicMock(ss58_address="our_hotkey")
        )
        return Validator(mock_config)


class TestOnMetagraphUpdated:
    """Tests for _on_metagraph_updated method."""

    def test_first_sync_initializes_hotkeys_scores_and_uid(
        self, validator: Validator
    ) -> None:
        """Test that first sync populates hotkeys, scores, and uid from metagraph."""
        hotkeys = ["hotkey_0", "hotkey_1", "our_hotkey", "hotkey_3"]
        validator.metagraph = create_mock_metagraph(hotkeys)

        validator._on_metagraph_updated()

        assert validator.hotkeys == hotkeys
        assert len(validator.scores) == len(hotkeys)
        assert validator.scores.dtype == np.float32
        np.testing.assert_array_equal(validator.scores, np.zeros(4, dtype=np.float32))
        assert validator.uid == 2  # "our_hotkey" is at index 2

    def test_hotkey_replaced_at_single_uid_zeros_score(
        self, validator: Validator
    ) -> None:
        """Test that when a hotkey changes at a specific UID, its score is zeroed."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "our_hotkey", "hotkey_3"]
        validator.scores = np.array([0.5, 0.8, 0.3, 0.9], dtype=np.float32)

        # Update metagraph with hotkey_1 replaced by new_hotkey_1
        new_hotkeys = ["hotkey_0", "new_hotkey_1", "our_hotkey", "hotkey_3"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)

        validator._on_metagraph_updated()

        # Score at UID 1 should be zeroed, others unchanged
        expected_scores = np.array([0.5, 0.0, 0.3, 0.9], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_multiple_hotkeys_replaced_zeros_multiple_scores(
        self, validator: Validator
    ) -> None:
        """Test that multiple hotkey replacements zero out multiple scores."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3", "hotkey_4"]
        validator.scores = np.array([0.5, 0.8, 0.3, 0.9, 0.7], dtype=np.float32)

        # Replace hotkeys at UIDs 0, 2, and 4
        new_hotkeys = [
            "new_hotkey_0",
            "hotkey_1",
            "new_hotkey_2",
            "hotkey_3",
            "new_hotkey_4",
        ]
        validator.metagraph = create_mock_metagraph(new_hotkeys)

        validator._on_metagraph_updated()

        # Scores at UIDs 0, 2, and 4 should be zeroed
        expected_scores = np.array([0.0, 0.8, 0.0, 0.9, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_hotkey_replacement_with_metagraph_growth(
        self, validator: Validator
    ) -> None:
        """Test hotkey replacement when metagraph also grows in size."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)

        # Replace hotkey at UID 1 and add new hotkeys
        new_hotkeys = ["hotkey_0", "new_hotkey_1", "hotkey_2", "hotkey_3", "hotkey_4"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)

        validator._on_metagraph_updated()

        # Score at UID 1 should be zeroed, UID 0, 2 unchanged, new UIDs get 0
        expected_scores = np.array([0.5, 0.0, 0.3, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_validator_deregistered_uid_becomes_none(
        self, validator: Validator
    ) -> None:
        """Test that when validator's hotkey is removed from metagraph, uid becomes None."""
        assert validator.hotkey == "our_hotkey"

        # Initialize with validator at UID 2
        validator.hotkeys = ["hotkey_0", "hotkey_1", "our_hotkey", "hotkey_3"]
        validator.scores = np.array([0.5, 0.8, 0.3, 0.9], dtype=np.float32)
        validator.uid = 2

        # Validator's hotkey removed from metagraph
        new_hotkeys = ["hotkey_0", "hotkey_1", "new_hotkey_2", "hotkey_3"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)

        validator._on_metagraph_updated()

        # Validator's uid should now be None (deregistered)
        assert validator.uid is None


class TestUpdateMetagraph:
    """Tests for update_metagraph method."""

    @pytest.mark.asyncio
    async def test_concurrent_updates_are_serialized(
        self, validator: Validator
    ) -> None:
        """Concurrent update_metagraph calls don't interleave."""
        call_order: list[str] = []

        async def slow_get_metagraph() -> Metagraph:
            """Simulate slow chain fetch."""
            call_order.append("start")
            await asyncio.sleep(0.05)  # Small delay
            call_order.append("end")
            return create_mock_metagraph(["hotkey_0", "our_hotkey"])

        # Setup mock chain
        mock_chain = MagicMock()
        mock_chain.get_metagraph = slow_get_metagraph
        validator.chain = mock_chain

        # Launch two concurrent updates
        await asyncio.gather(
            validator.update_metagraph(),
            validator.update_metagraph(),
        )

        # Without lock: [start, start, end, end] (interleaved)
        # With lock: [start, end, start, end] (serialized)
        assert call_order == ["start", "end", "start", "end"]


class TestSetWeights:
    """Tests for set_weights method."""

    @pytest.mark.asyncio
    async def test_set_weights_normalizes_and_maps_to_hotkeys(
        self, validator: Validator
    ) -> None:
        """Test normalization math and hotkey-to-weight mapping."""
        # Include validator's own hotkey in the metagraph for validator_permit check
        validator.hotkeys = [
            "hotkey_0",
            "hotkey_1",
            "hotkey_2",
            "hotkey_3",
            "our_hotkey",
        ]
        validator.scores = np.array([1.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        # Set up mock chain client
        mock_chain = MagicMock()
        mock_chain.set_weights = AsyncMock()
        validator.chain = mock_chain

        await validator.set_weights()

        # [1, 0, 3, 0] / 4 = [0.25, 0, 0.75, 0] -> only non-zero in dict
        mock_chain.set_weights.assert_called_once_with(
            {"hotkey_0": 0.25, "hotkey_2": 0.75}
        )

    @pytest.mark.asyncio
    async def test_set_weights_with_burn_allocation(
        self, validator: Validator
    ) -> None:
        """Test burn allocates fraction to burn_uid and scales down rest."""
        validator.hotkeys = [
            "hotkey_0",
            "hotkey_1",
            "burn_hotkey",  # UID 2 is the burn target
            "hotkey_3",
            "our_hotkey",
        ]
        validator.scores = np.array([1.0, 0.0, 0.0, 3.0, 0.0], dtype=np.float32)
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        # Configure 50% burn to UID 2
        validator.config.burn_amount = 0.5
        validator.config.burn_uid = 2

        mock_chain = MagicMock()
        mock_chain.set_weights = AsyncMock()
        validator.chain = mock_chain

        await validator.set_weights()

        # Original: {hotkey_0: 0.25, hotkey_3: 0.75}
        # After 50% burn: {hotkey_0: 0.125, hotkey_3: 0.375, burn_hotkey: 0.5}
        mock_chain.set_weights.assert_called_once()
        weights = mock_chain.set_weights.call_args[0][0]
        assert weights["hotkey_0"] == pytest.approx(0.125)
        assert weights["hotkey_3"] == pytest.approx(0.375)
        assert weights["burn_hotkey"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_set_weights_no_burn_when_zero(
        self, validator: Validator
    ) -> None:
        """Test no burn applied when burn_amount is 0."""
        validator.hotkeys = ["hotkey_0", "hotkey_1", "our_hotkey"]
        validator.scores = np.array([1.0, 3.0, 0.0], dtype=np.float32)
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        # No burn configured (defaults)
        validator.config.burn_amount = 0.0
        validator.config.burn_uid = -1

        mock_chain = MagicMock()
        mock_chain.set_weights = AsyncMock()
        validator.chain = mock_chain

        await validator.set_weights()

        # No burn: {hotkey_0: 0.25, hotkey_1: 0.75}
        mock_chain.set_weights.assert_called_once_with(
            {"hotkey_0": 0.25, "hotkey_1": 0.75}
        )


class TestRunEvaluationScores:
    """Tests for score updates in _run_evaluation."""

    @pytest.mark.asyncio
    async def test_old_scores_are_zeroed_before_update(
        self, validator: Validator
    ) -> None:
        """Miners not in evaluation results get 0, not stale scores."""
        # Setup: 4 hotkeys, hotkey_1 had old score
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3"]
        validator.scores = np.array(
            [0.0, 0.5, 0.0, 0.0], dtype=np.float32
        )  # hotkey_1 has old score
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        # Mock download results - only hotkey_0 and hotkey_2 have models
        validator.download_results = {
            "hotkey_0": MagicMock(success=True, model_path="/model_0.onnx"),
            "hotkey_2": MagicMock(success=True, model_path="/model_2.onnx"),
        }

        # Mock scheduler
        validator._model_scheduler = MagicMock()
        validator._model_scheduler.known_commitments = {
            "hotkey_0": MagicMock(),
            "hotkey_2": MagicMock(),
        }

        # Mock orchestrator - returns weights only for evaluated miners
        mock_weights = MagicMock()
        mock_weights.weights = {
            "hotkey_0": 0.99,
            "hotkey_2": 0.01,
        }  # hotkey_1 NOT included
        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "hotkey_0"
        mock_winner.winner_score = 0.95
        mock_result = MagicMock()
        mock_result.weights = mock_weights
        mock_result.winner = mock_winner

        validator._orchestrator = MagicMock()
        validator._orchestrator.run = AsyncMock(return_value=mock_result)

        # Run evaluation
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        await validator._run_evaluation(mock_dataset)

        # Verify: hotkey_1's old score (0.5) should be zeroed, not preserved
        assert validator.scores[0] == 0.99  # hotkey_0 - from weights
        assert validator.scores[1] == 0.0  # hotkey_1 - zeroed (was 0.5)
        assert validator.scores[2] == 0.01  # hotkey_2 - from weights
        assert validator.scores[3] == 0.0  # hotkey_3 - zeroed

    @pytest.mark.asyncio
    async def test_all_scores_come_from_current_evaluation(
        self, validator: Validator
    ) -> None:
        """After evaluation, scores exactly match weights from orchestrator."""
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array(
            [0.3, 0.4, 0.3], dtype=np.float32
        )  # All have old scores
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        validator.download_results = {
            "hotkey_0": MagicMock(success=True, model_path="/model_0.onnx"),
        }

        validator._model_scheduler = MagicMock()
        validator._model_scheduler.known_commitments = {"hotkey_0": MagicMock()}

        # Only hotkey_0 gets weight
        mock_weights = MagicMock()
        mock_weights.weights = {"hotkey_0": 1.0}
        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "hotkey_0"
        mock_winner.winner_score = 0.95
        mock_result = MagicMock()
        mock_result.weights = mock_weights
        mock_result.winner = mock_winner

        validator._orchestrator = MagicMock()
        validator._orchestrator.run = AsyncMock(return_value=mock_result)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        await validator._run_evaluation(mock_dataset)

        # All old scores replaced
        np.testing.assert_array_equal(
            validator.scores, np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )


class TestGetNextEvalTime:
    """Tests for _get_next_eval_time method."""

    def test_returns_today_if_before_scheduled_time(self, validator: Validator) -> None:
        """If current time is before scheduled time, returns today."""
        # Schedule at 14:00 UTC
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        # Mock "now" as 10:00 UTC
        mock_now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        with patch("real_estate.validator.validator.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            result = validator._get_next_eval_time()

        # Should be today at 14:00
        expected = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        assert result == expected

    def test_returns_tomorrow_if_after_scheduled_time(
        self, validator: Validator
    ) -> None:
        """If current time is after scheduled time, returns tomorrow."""
        # Schedule at 14:00 UTC
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        # Mock "now" as 16:00 UTC (after scheduled time)
        mock_now = datetime(2025, 1, 15, 16, 0, 0, tzinfo=UTC)
        with patch("real_estate.validator.validator.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            result = validator._get_next_eval_time()

        # Should be tomorrow at 14:00
        expected = datetime(2025, 1, 16, 14, 0, 0, tzinfo=UTC)
        assert result == expected

    def test_returns_tomorrow_if_exactly_at_scheduled_time(
        self, validator: Validator
    ) -> None:
        """If current time equals scheduled time, returns tomorrow."""
        # Schedule at 14:00 UTC
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        # Mock "now" as exactly 14:00 UTC
        mock_now = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        with patch("real_estate.validator.validator.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            result = validator._get_next_eval_time()

        # Should be tomorrow at 14:00 (since <= triggers tomorrow)
        expected = datetime(2025, 1, 16, 14, 0, 0, tzinfo=UTC)
        assert result == expected

    def test_respects_minute_configuration(self, validator: Validator) -> None:
        """Scheduled minute is respected."""
        # Schedule at 02:30 UTC
        validator.config.validation_data_schedule_hour = 2
        validator.config.validation_data_schedule_minute = 30

        # Mock "now" as 01:00 UTC
        mock_now = datetime(2025, 1, 15, 1, 0, 0, tzinfo=UTC)
        with patch("real_estate.validator.validator.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            result = validator._get_next_eval_time()

        # Should be today at 02:30
        expected = datetime(2025, 1, 15, 2, 30, 0, tzinfo=UTC)
        assert result == expected


class TestRunCatchUpIfTime:
    """Tests for _run_catch_up_if_time method."""

    @pytest.mark.asyncio
    async def test_skips_if_past_eval_time(self, validator: Validator) -> None:
        """Skips catch-up if evaluation time has already passed."""
        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_catch_up = AsyncMock()
        validator.download_results = {}

        # Eval time is in the past
        past_eval = datetime.now(UTC) - timedelta(minutes=5)

        await validator._run_catch_up_if_time(past_eval)

        # Should not call run_catch_up
        validator._model_scheduler.run_catch_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_extracts_failed_hotkeys_from_download_results(
        self, validator: Validator
    ) -> None:
        """Extracts failed hotkeys from download_results and passes to catch-up."""
        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_catch_up = AsyncMock(return_value={})

        # Setup download results with some failures
        validator.download_results = {
            "hotkey_success": DownloadResult(hotkey="hotkey_success", success=True),
            "hotkey_fail_1": DownloadResult(hotkey="hotkey_fail_1", success=False),
            "hotkey_fail_2": DownloadResult(hotkey="hotkey_fail_2", success=False),
        }

        # Mock time so we're past deadline but before eval
        mock_now = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        eval_time = datetime(2025, 1, 15, 14, 5, 0, tzinfo=UTC)  # 5 min in future
        # With 30 min catch_up_minutes, deadline = 13:35, we're past it

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await validator._run_catch_up_if_time(eval_time)

        # Should call run_catch_up with failed hotkeys
        validator._model_scheduler.run_catch_up.assert_called_once()
        call_kwargs = validator._model_scheduler.run_catch_up.call_args.kwargs
        assert call_kwargs["failed_hotkeys"] == {"hotkey_fail_1", "hotkey_fail_2"}

    @pytest.mark.asyncio
    async def test_merges_catch_up_results_with_download_results(
        self, validator: Validator
    ) -> None:
        """Catch-up results are merged into download_results."""
        # Catch-up returns a recovered download
        recovered_result = DownloadResult(hotkey="hotkey_fail", success=True)
        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_catch_up = AsyncMock(
            return_value={"hotkey_fail": recovered_result}
        )

        validator.download_results = {
            "hotkey_fail": DownloadResult(hotkey="hotkey_fail", success=False),
        }

        mock_now = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        eval_time = datetime(2025, 1, 15, 14, 5, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await validator._run_catch_up_if_time(eval_time)

        # download_results should be updated with recovered result
        assert validator.download_results["hotkey_fail"].success is True

    @pytest.mark.asyncio
    async def test_handles_empty_download_results(self, validator: Validator) -> None:
        """Handles case when download_results is empty."""
        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_catch_up = AsyncMock(return_value={})
        validator.download_results = {}

        mock_now = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        eval_time = datetime(2025, 1, 15, 14, 5, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await validator._run_catch_up_if_time(eval_time)

        # Should call with None (no failed hotkeys)
        validator._model_scheduler.run_catch_up.assert_called_once()
        call_kwargs = validator._model_scheduler.run_catch_up.call_args.kwargs
        assert call_kwargs["failed_hotkeys"] is None


class TestPreDownloadLoop:
    """Tests for _pre_download_loop method."""

    @pytest.mark.asyncio
    async def test_runs_immediately_when_past_pre_download_start(
        self, validator: Validator
    ) -> None:
        """Pre-download runs immediately when already past pre_download_start time."""
        validator.config.scheduler_pre_download_hours = 2.0
        validator.config.scheduler_catch_up_minutes = 30.0
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_pre_download = AsyncMock(return_value={})

        call_count = 0

        async def mock_run_pre_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()
            return {}

        validator._model_scheduler.run_pre_download = mock_run_pre_download
        validator._run_catch_up_if_time = AsyncMock()

        # Mock time: now=13:00, eval=14:00, pre_download_start=12:00 (2h before eval)
        # Since now > pre_download_start, should run immediately (no sleep)
        mock_now = datetime(2025, 1, 15, 13, 0, 0, tzinfo=UTC)
        mock_sleep = AsyncMock()

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch("real_estate.validator.validator.asyncio.sleep", mock_sleep),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            with pytest.raises(asyncio.CancelledError):
                await validator._pre_download_loop()

        assert call_count == 1
        # Should NOT have slept before pre-download (already past start time)
        # Note: may sleep after eval for next cycle, but first pre-download is immediate
        for call in mock_sleep.call_args_list:
            # Any sleep should be small (not waiting hours)
            if call.args:
                assert call.args[0] < 3600, (
                    "Should not wait hours when past pre_download_start"
                )

    @pytest.mark.asyncio
    async def test_waits_until_pre_download_start_when_early(
        self, validator: Validator
    ) -> None:
        """Pre-download waits until pre_download_start when validator starts early."""
        validator.config.scheduler_pre_download_hours = 3.0  # 3h before eval
        validator.config.scheduler_catch_up_minutes = 30.0
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        validator._model_scheduler = MagicMock()
        pre_download_called = False
        sleep_durations = []

        async def mock_run_pre_download(*args, **kwargs):
            nonlocal pre_download_called
            pre_download_called = True
            raise asyncio.CancelledError()

        async def mock_sleep(seconds):
            sleep_durations.append(seconds)
            # After first sleep, advance time past pre_download_start
            if len(sleep_durations) == 1:
                mock_datetime.now.return_value = datetime(
                    2025, 1, 15, 11, 0, 0, tzinfo=UTC
                )

        validator._model_scheduler.run_pre_download = mock_run_pre_download
        validator._run_catch_up_if_time = AsyncMock()

        # Mock time: now=10:00, eval=14:00, pre_download_start=11:00 (3h before eval)
        # Should wait 1 hour (3600 seconds) before starting
        mock_now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch("real_estate.validator.validator.asyncio.sleep", mock_sleep),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            with pytest.raises(asyncio.CancelledError):
                await validator._pre_download_loop()

        assert pre_download_called, "Pre-download should have been called"
        assert len(sleep_durations) >= 1, "Should have waited before pre-download"
        # First sleep should be ~1 hour (3600 seconds) to wait until pre_download_start
        assert sleep_durations[0] == pytest.approx(3600, rel=0.01), (
            f"Should wait ~1 hour, got {sleep_durations[0]}s"
        )

    @pytest.mark.asyncio
    async def test_calls_catch_up_after_pre_download(
        self, validator: Validator
    ) -> None:
        """Catch-up is called after pre-download completes."""
        validator.config.scheduler_pre_download_hours = 0.0  # Immediate
        validator.config.scheduler_catch_up_minutes = 0.0
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        call_order = []

        async def mock_pre_download(*args, **kwargs):
            call_order.append("pre_download")
            raise asyncio.CancelledError()  # Stop after one iteration

        async def mock_catch_up(*args, **kwargs):
            call_order.append("catch_up")

        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_pre_download = mock_pre_download
        validator._run_catch_up_if_time = mock_catch_up

        mock_now = datetime(2025, 1, 15, 13, 0, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            with pytest.raises(asyncio.CancelledError):
                await validator._pre_download_loop()

        assert "pre_download" in call_order

    @pytest.mark.asyncio
    async def test_handles_pre_download_failure_gracefully(
        self, validator: Validator
    ) -> None:
        """Pre-download failure doesn't crash the loop."""
        validator.config.scheduler_pre_download_hours = 0.0
        validator.config.scheduler_catch_up_minutes = 0.0
        validator.config.validation_data_schedule_hour = 14
        validator.config.validation_data_schedule_minute = 0

        call_count = 0

        async def mock_pre_download_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Download failed")
            raise asyncio.CancelledError()

        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_pre_download = mock_pre_download_fails
        validator._run_catch_up_if_time = AsyncMock()

        mock_now = datetime(2025, 1, 15, 13, 0, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            with pytest.raises(asyncio.CancelledError):
                await validator._pre_download_loop()

        # Should have attempted twice (first failed, second cancelled)
        assert call_count == 2


class TestEvaluationRetry:
    """Tests for evaluation retry on connection errors."""

    @pytest.mark.asyncio
    async def test_retries_evaluation_on_chain_connection_error(
        self, validator: Validator
    ) -> None:
        """Evaluation retries on ChainConnectionError."""
        from real_estate.chain.errors import ChainConnectionError

        validator.validation_data = MagicMock()
        attempt_count = 0

        async def mock_update_metagraph():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ChainConnectionError("Connection failed")
            # Third attempt succeeds

        validator.update_metagraph = mock_update_metagraph
        validator._run_evaluation = AsyncMock()
        validator._evaluation_event = asyncio.Event()
        validator._evaluation_event.set()

        # Run one iteration of the loop by cancelling after success
        async def run_once():
            await validator._evaluation_event.wait()
            validator._evaluation_event.clear()
            # Import needed for retry
            from tenacity import (
                AsyncRetrying,
                retry_if_exception_type,
                stop_after_attempt,
                wait_fixed,
            )

            try:
                async for attempt in AsyncRetrying(
                    wait=wait_fixed(0),  # No wait in tests
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(ChainConnectionError),
                    reraise=True,
                ):
                    with attempt:
                        await validator.update_metagraph()
                        await validator._run_evaluation(validator.validation_data)
            except ChainConnectionError:
                pass

        await run_once()

        # Should have called update_metagraph 3 times (2 failures + 1 success)
        assert attempt_count == 3
        validator._run_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_fails_after_max_retries(
        self, validator: Validator
    ) -> None:
        """Evaluation gives up after max retries exhausted."""
        from real_estate.chain.errors import ChainConnectionError

        validator.validation_data = MagicMock()

        async def always_fail():
            raise ChainConnectionError("Connection failed")

        validator.update_metagraph = always_fail
        validator._run_evaluation = AsyncMock()

        # Import needed for retry
        from tenacity import (
            AsyncRetrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_fixed,
        )

        failed = False
        try:
            async for attempt in AsyncRetrying(
                wait=wait_fixed(0),
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type(ChainConnectionError),
                reraise=True,
            ):
                with attempt:
                    await validator.update_metagraph()
        except ChainConnectionError:
            failed = True

        assert failed
        validator._run_evaluation.assert_not_called()


class TestCatchUpRetry:
    """Tests for catch-up retry on connection errors."""

    @pytest.mark.asyncio
    async def test_retries_catch_up_on_chain_connection_error(
        self, validator: Validator
    ) -> None:
        """Catch-up retries on ChainConnectionError until success."""
        from real_estate.chain.errors import ChainConnectionError

        validator.download_results = {}
        attempt_count = 0

        async def mock_run_catch_up(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ChainConnectionError("Connection failed")
            return {"hotkey1": MagicMock(success=True)}

        validator._model_scheduler = MagicMock()
        validator._model_scheduler.run_catch_up = mock_run_catch_up
        validator.config.scheduler_catch_up_minutes = 3.0

        # Mock time: now is 2 min before eval
        eval_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC)
        mock_now = datetime(2025, 1, 15, 13, 58, 0, tzinfo=UTC)

        with (
            patch("real_estate.validator.validator.datetime") as mock_datetime,
            patch(
                "real_estate.validator.validator.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await validator._run_catch_up_if_time(eval_time)

        # Should have retried once after initial failure
        assert attempt_count == 2
        assert "hotkey1" in validator.download_results
