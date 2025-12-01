"""Tests for Validator class."""

from datetime import datetime
from unittest.mock import MagicMock, patch

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
        netuid=1,
        block=block,
        neurons=neurons,
        timestamp=datetime.now(),
    )


class TestOnMetagraphUpdatedFirstSync:
    """Tests for _on_metagraph_updated first sync initialization."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config for Validator."""
        config = MagicMock()
        config.pylon_url = "http://test.pylon"
        config.pylon_token = "test_token"
        config.subtensor_network = "test"
        config.netuid = 1
        config.hotkey = "our_hotkey"
        config.state_path = MagicMock()
        config.disable_set_weights = False
        config.epoch_length = 100
        config.moving_average_alpha = 0.1
        return config

    @pytest.fixture
    def validator(self, mock_config: MagicMock) -> Validator:
        """Create a Validator instance with mocked dependencies."""
        with (
            patch("real_estate.validator.check_config"),
            patch("real_estate.validator.bt.subtensor") as mock_subtensor,
        ):
            mock_subtensor.return_value = MagicMock(chain_endpoint="mock_endpoint")
            return Validator(mock_config)

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
