"""Tests for Validator class."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from real_estate.chain.models import Metagraph, Neuron
from real_estate.validator import Validator


def create_mock_neuron(uid: int, hotkey: str) -> Neuron:
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
        mock_wallet.return_value = MagicMock(hotkey=MagicMock(ss58_address="our_hotkey"))
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
        new_hotkeys = ["new_hotkey_0", "hotkey_1", "new_hotkey_2", "hotkey_3", "new_hotkey_4"]
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
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3"]
        validator.scores = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        validator.metagraph = create_mock_metagraph(validator.hotkeys)

        # Set up mock chain client
        mock_chain = MagicMock()
        mock_chain.set_weights = AsyncMock()
        validator.chain = mock_chain

        await validator.set_weights()

        # [1, 0, 3, 0] / 4 = [0.25, 0, 0.75, 0] -> only non-zero in dict
        mock_chain.set_weights.assert_called_once_with({
            "hotkey_0": 0.25,
            "hotkey_2": 0.75
        })


class TestRunEvaluationScores:
    """Tests for score updates in _run_evaluation."""

    @pytest.mark.asyncio
    async def test_old_scores_are_zeroed_before_update(
        self, validator: Validator
    ) -> None:
        """Miners not in evaluation results get 0, not stale scores."""
        # Setup: 4 hotkeys, hotkey_1 had old score
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3"]
        validator.scores = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)  # hotkey_1 has old score
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
        mock_weights.weights = {"hotkey_0": 0.99, "hotkey_2": 0.01}  # hotkey_1 NOT included
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
        assert validator.scores[1] == 0.0   # hotkey_1 - zeroed (was 0.5)
        assert validator.scores[2] == 0.01  # hotkey_2 - from weights
        assert validator.scores[3] == 0.0   # hotkey_3 - zeroed

    @pytest.mark.asyncio
    async def test_all_scores_come_from_current_evaluation(
        self, validator: Validator
    ) -> None:
        """After evaluation, scores exactly match weights from orchestrator."""
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.3, 0.4, 0.3], dtype=np.float32)  # All have old scores
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
            validator.scores,
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )


class TestGetNextEvalTime:
    """Tests for _get_next_eval_time method."""

    def test_returns_today_if_before_scheduled_time(
        self, validator: Validator
    ) -> None:
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

    def test_respects_minute_configuration(
        self, validator: Validator
    ) -> None:
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
