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


class TestOnMetagraphUpdated:
    """Tests for _on_metagraph_updated method."""

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
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3"]
        validator.scores = np.array([0.5, 0.8, 0.3, 0.9], dtype=np.float32)
        
        # Replace hotkeys at UIDs 0 and 2
        new_hotkeys = ["new_hotkey_0", "hotkey_1", "new_hotkey_2", "hotkey_3"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Scores at UIDs 0 and 2 should be zeroed
        expected_scores = np.array([0.0, 0.8, 0.0, 0.9], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_all_hotkeys_replaced_zeros_all_scores(
        self, validator: Validator
    ) -> None:
        """Test that replacing all hotkeys zeros out all scores."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        
        # Replace all hotkeys
        new_hotkeys = ["new_hotkey_0", "new_hotkey_1", "new_hotkey_2"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # All scores should be zeroed
        expected_scores = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_no_hotkey_changes_preserves_scores(
        self, validator: Validator
    ) -> None:
        """Test that when no hotkeys change, scores remain unchanged."""
        # Initialize with hotkeys and scores
        initial_hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.hotkeys = initial_hotkeys.copy()
        validator.scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        
        # Metagraph with same hotkeys
        validator.metagraph = create_mock_metagraph(initial_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Scores should remain unchanged
        expected_scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == initial_hotkeys

    def test_hotkey_replaced_at_last_position(
        self, validator: Validator
    ) -> None:
        """Test that hotkey replacement at the last UID zeros its score."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.5, 0.8, 0.9], dtype=np.float32)
        
        # Replace hotkey at last position
        new_hotkeys = ["hotkey_0", "hotkey_1", "new_hotkey_2"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Score at last UID should be zeroed
        expected_scores = np.array([0.5, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_hotkey_replaced_at_first_position(
        self, validator: Validator
    ) -> None:
        """Test that hotkey replacement at UID 0 zeros its score."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.5, 0.8, 0.9], dtype=np.float32)
        
        # Replace hotkey at first position
        new_hotkeys = ["new_hotkey_0", "hotkey_1", "hotkey_2"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Score at UID 0 should be zeroed
        expected_scores = np.array([0.0, 0.8, 0.9], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_hotkey_replacement_with_metagraph_growth(
        self, validator: Validator
    ) -> None:
        """Test hotkey replacement when metagraph also grows in size."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1"]
        validator.scores = np.array([0.5, 0.8], dtype=np.float32)
        
        # Replace hotkey at UID 1 and add new hotkeys
        new_hotkeys = ["hotkey_0", "new_hotkey_1", "hotkey_2", "hotkey_3"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Score at UID 1 should be zeroed, new UIDs should have 0 scores
        expected_scores = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys

    def test_hotkey_replacement_preserves_unchanged_scores_with_growth(
        self, validator: Validator
    ) -> None:
        """Test that unchanged hotkeys preserve scores even when metagraph grows."""
        # Initialize with original hotkeys and scores
        validator.hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2"]
        validator.scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        
        # Keep existing hotkeys, add new ones
        new_hotkeys = ["hotkey_0", "hotkey_1", "hotkey_2", "hotkey_3", "hotkey_4"]
        validator.metagraph = create_mock_metagraph(new_hotkeys)
        
        validator._on_metagraph_updated()
        
        # Original scores preserved, new UIDs get 0
        expected_scores = np.array([0.5, 0.8, 0.3, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(validator.scores, expected_scores)
        assert validator.hotkeys == new_hotkeys
