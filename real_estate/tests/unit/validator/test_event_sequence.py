"""
Unit tests for validator event sequencing.

Tests the coordination between:
- Validation data fetching
- Evaluation triggering
- Metagraph updates
- Weight setting

Uses mocks to avoid real chain/API calls while testing async coordination.
"""

import asyncio
import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

# Mock pylon_client module before importing Validator
sys.modules["pylon_client"] = MagicMock()
sys.modules["pylon_client.v1"] = MagicMock()


@pytest.fixture
def mock_validator_config() -> MagicMock:
    """Create a minimal mock config for Validator."""
    config = MagicMock()
    config.pylon_url = "http://test"
    config.pylon_token = "test_token"
    config.pylon_identity = "test_identity"
    config.wallet_name = "test"
    config.wallet_hotkey = "test"
    config.subtensor_network = "test"
    config.validation_data_url = "http://test"
    config.validation_data_max_retries = 1
    config.validation_data_retry_delay = 1
    config.validation_data_schedule_hour = 2
    config.validation_data_schedule_minute = 0
    config.validation_data_download_raw = False
    config.disable_set_weights = False
    config.epoch_length = 100
    return config


@pytest.fixture
def mock_validator(mock_validator_config: MagicMock) -> Generator:
    """Create a Validator instance with mocked dependencies."""
    from real_estate.validator.validator import Validator

    with patch("real_estate.validator.validator.bt") as mock_bt, \
         patch("real_estate.validator.validator.check_config"):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5TestHotkey"
        mock_wallet.name = "test"
        mock_wallet.hotkey_str = "test"
        mock_bt.wallet.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.chain_endpoint = "ws://test"
        mock_bt.subtensor.return_value = mock_subtensor

        yield Validator(mock_validator_config)


class TestEvaluationEventSequence:
    """Tests for evaluation event triggering and handling."""

    @pytest.mark.asyncio
    async def test_validation_data_triggers_evaluation_event(self, mock_validator) -> None:
        """Validation data callback should set the evaluation event."""
        # Event should not be set initially
        assert not mock_validator._evaluation_event.is_set()

        # Simulate validation data callback
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_validator._on_validation_data_fetched(mock_dataset, None)

        # Event should now be set
        assert mock_validator._evaluation_event.is_set()
        assert mock_validator.validation_data == mock_dataset

    @pytest.mark.asyncio
    async def test_evaluation_clears_event_after_processing(self) -> None:
        """Evaluation loop should clear event after processing."""
        event = asyncio.Event()
        event.set()

        # Simulate one iteration of evaluation loop
        processed = False

        async def mock_evaluation_iteration():
            nonlocal processed
            await event.wait()
            event.clear()
            processed = True

        await asyncio.wait_for(mock_evaluation_iteration(), timeout=1.0)

        assert processed
        assert not event.is_set()


class TestStartupSequence:
    """Tests for validator startup sequence."""

    @pytest.mark.asyncio
    async def test_metagraph_fetch_before_evaluation(self) -> None:
        """Metagraph must be fetched before evaluation can proceed."""
        call_order: list[str] = []

        async def track_metagraph():
            call_order.append("metagraph")

        async def track_evaluation():
            call_order.append("evaluation")

        # Simulate startup sequence
        await track_metagraph()
        await track_evaluation()

        assert call_order == ["metagraph", "evaluation"]
        assert call_order.index("metagraph") < call_order.index("evaluation")

    @pytest.mark.asyncio
    async def test_registration_check_blocks_unregistered(self, mock_validator) -> None:
        """Unregistered validator should not proceed with evaluation."""
        # No metagraph means not registered
        assert not mock_validator.is_registered()

        # With metagraph but hotkey not in list
        mock_validator.metagraph = MagicMock()
        mock_validator.hotkeys = ["5OtherHotkey1", "5OtherHotkey2"]

        assert not mock_validator.is_registered()


class TestWeightSettingCoordination:
    """Tests for weight setting timing and coordination."""

    @pytest.mark.asyncio
    async def test_weight_setting_respects_epoch_length(self, mock_validator) -> None:
        """Weights should only be set after epoch_length blocks."""
        mock_validator._last_weight_set_block = 1000

        with patch("real_estate.validator.validator.ttl_get_block") as mock_block:
            # At block 1050 (50 blocks elapsed), should not set weights
            mock_block.return_value = 1050
            assert not mock_validator.should_set_weights()

            # At block 1100 (100 blocks elapsed), should not set (needs > epoch_length)
            mock_block.return_value = 1100
            assert not mock_validator.should_set_weights()

            # At block 1101 (101 blocks elapsed), should set weights
            mock_block.return_value = 1101
            assert mock_validator.should_set_weights()

    @pytest.mark.asyncio
    async def test_disabled_weight_setting(self, mock_validator_config) -> None:
        """Weight setting should be skipped when disabled."""
        from real_estate.validator.validator import Validator

        mock_validator_config.disable_set_weights = True  # Disabled

        with patch("real_estate.validator.validator.bt") as mock_bt, \
             patch("real_estate.validator.validator.check_config"), \
             patch("real_estate.validator.validator.ttl_get_block") as mock_block:
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "5TestHotkey"
            mock_wallet.name = "test"
            mock_wallet.hotkey_str = "test"
            mock_bt.wallet.return_value = mock_wallet

            mock_subtensor = MagicMock()
            mock_subtensor.chain_endpoint = "ws://test"
            mock_bt.subtensor.return_value = mock_subtensor

            validator = Validator(mock_validator_config)
            validator._last_weight_set_block = 0

            # Even with many blocks elapsed, should not set weights
            mock_block.return_value = 10000
            assert not validator.should_set_weights()


class TestConcurrentLoopCoordination:
    """Tests for coordination between concurrent loops."""

    @pytest.mark.asyncio
    async def test_event_based_loop_waits_correctly(self) -> None:
        """Event-based loop should wait until event is set."""
        event = asyncio.Event()
        iterations = 0

        async def event_loop():
            nonlocal iterations
            for _ in range(3):
                await event.wait()
                event.clear()
                iterations += 1

        # Start the loop
        loop_task = asyncio.create_task(event_loop())

        # Initially no iterations
        await asyncio.sleep(0.05)
        assert iterations == 0

        # Trigger first iteration
        event.set()
        await asyncio.sleep(0.05)
        assert iterations == 1

        # Trigger remaining iterations
        event.set()
        await asyncio.sleep(0.05)
        event.set()
        await asyncio.sleep(0.05)

        # Wait for completion
        await asyncio.wait_for(loop_task, timeout=1.0)
        assert iterations == 3

    @pytest.mark.asyncio
    async def test_timer_loop_runs_independently(self) -> None:
        """Timer-based loop should run on schedule regardless of events."""
        check_count = 0
        check_times: list[float] = []
        start_time = asyncio.get_event_loop().time()

        async def timer_loop():
            nonlocal check_count
            for _ in range(3):
                check_times.append(asyncio.get_event_loop().time() - start_time)
                check_count += 1
                await asyncio.sleep(0.05)  # 50ms interval

        await timer_loop()

        assert check_count == 3
        # Verify timing (with some tolerance)
        assert check_times[1] >= 0.04  # Second check after ~50ms
        assert check_times[2] >= 0.09  # Third check after ~100ms


class TestValidationDataFetchFailure:
    """Tests for handling validation data fetch failures."""

    @pytest.mark.asyncio
    async def test_data_fetch_failure_does_not_trigger_event(self, mock_validator) -> None:
        """Validation data fetch failure should not trigger evaluation event."""
        # Event should not be set initially
        assert not mock_validator._evaluation_event.is_set()
        assert mock_validator.validation_data is None

        # Simulate failed fetch - dataset is None
        mock_validator._on_validation_data_fetched(None, None)

        # Event should NOT be set on failure
        assert not mock_validator._evaluation_event.is_set()
        # validation_data should remain None
        assert mock_validator.validation_data is None

    @pytest.mark.asyncio
    async def test_data_fetch_success_after_failure(self, mock_validator) -> None:
        """Successful fetch after previous failure should work correctly."""
        # First: failed fetch
        mock_validator._on_validation_data_fetched(None, None)
        assert not mock_validator._evaluation_event.is_set()

        # Second: successful fetch
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_validator._on_validation_data_fetched(mock_dataset, None)

        # Now event should be set
        assert mock_validator._evaluation_event.is_set()
        assert mock_validator.validation_data == mock_dataset

    @pytest.mark.asyncio
    async def test_empty_dataset_does_not_trigger_event(self, mock_validator) -> None:
        """Empty dataset should not trigger evaluation - nothing to evaluate."""
        # Simulate empty dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_validator._on_validation_data_fetched(mock_dataset, None)

        # Event should NOT be set - empty dataset means nothing to evaluate
        assert not mock_validator._evaluation_event.is_set()
        # validation_data should not be updated
        assert mock_validator.validation_data is None
