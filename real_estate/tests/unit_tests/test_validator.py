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

@pytest.fixture
def mock_config() -> MagicMock:
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
def validator(mock_config: MagicMock) -> Validator:
    """Create a Validator instance with mocked dependencies."""
    with (
        patch("real_estate.validator.check_config"),
        patch("real_estate.validator.bt.subtensor") as mock_subtensor,
    ):
        mock_subtensor.return_value = MagicMock(chain_endpoint="mock_endpoint")
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
        
        # Score at UID 1 should be zeroed, UID 0, 2 should remain unchanged, new UIDs should get 0 scores
        expected_scores = np.array([0.5, 0.0, 0.3, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys
        
    def test_validator_deregistered_uid_becomes_none(
        self, validator: Validator
    ) -> None:
        """Test that when validator's hotkey is removed from metagraph, uid becomes None."""
        # Ensure validator's hotkey is set correctly
        assert validator.hotkey == "our_hotkey", f"Expected hotkey to be 'our_hotkey', got '{validator.hotkey}'"
        
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

        with patch.object(validator.pylon, "set_weights") as mock_set_weights:
            await validator.set_weights()

            # [1, 0, 3, 0] / 4 = [0.25, 0, 0.75, 0] → only non-zero in dict
            mock_set_weights.assert_called_once_with({
                "hotkey_0": 0.25,
                "hotkey_2": 0.75
            })