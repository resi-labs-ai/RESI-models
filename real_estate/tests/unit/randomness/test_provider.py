"""Tests for DecentralizedSeedProvider and combine_reveals."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from real_estate.randomness import (
    DecentralizedSeedProvider,
    RandomnessConfig,
    SeedResult,
    combine_reveals,
)


class TestCombineReveals:
    def test_deterministic(self) -> None:
        """Same inputs produce same seed."""
        reveals = {"hotkey_a": "abc123", "hotkey_b": "def456"}
        s1 = combine_reveals(reveals, 2**32)
        s2 = combine_reveals(reveals, 2**32)
        assert s1 == s2

    def test_order_independent(self) -> None:
        """Dict insertion order doesn't matter — sorted by key."""
        r1 = {"b": "val2", "a": "val1"}
        r2 = {"a": "val1", "b": "val2"}
        assert combine_reveals(r1, 2**32) == combine_reveals(r2, 2**32)

    def test_different_inputs_differ(self) -> None:
        """Different reveals produce different seeds."""
        s1 = combine_reveals({"a": "val1", "b": "val2"}, 2**32)
        s2 = combine_reveals({"a": "val1", "c": "val3"}, 2**32)
        assert s1 != s2

    def test_within_modulus(self) -> None:
        """Seed is always within [0, modulus)."""
        for modulus in [100, 1000, 2**32]:
            seed = combine_reveals({"a": "test"}, modulus)
            assert 0 <= seed < modulus

    def test_single_reveal(self) -> None:
        """Single reveal produces a valid deterministic seed."""
        seed = combine_reveals({"only_validator": "hexdata"}, 2**32)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    def test_empty_reveals_raises(self) -> None:
        """Empty reveals dict raises ValueError."""
        with pytest.raises(ValueError, match="reveals must not be empty"):
            combine_reveals({}, 2**32)


class TestDecentralizedSeedProviderCommit:
    def test_commit_success(self) -> None:
        """Successful commit returns reveal round."""
        subtensor = MagicMock()
        subtensor.set_reveal_commitment.return_value = (True, 42)
        wallet = MagicMock()

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=wallet, netuid=46,
        )
        result = provider.commit()

        assert result == 42
        subtensor.set_reveal_commitment.assert_called_once()
        call_kwargs = subtensor.set_reveal_commitment.call_args.kwargs
        assert call_kwargs["wallet"] is wallet
        assert call_kwargs["netuid"] == 46
        assert call_kwargs["blocks_until_reveal"] == 360
        assert len(call_kwargs["data"]) == 64  # 32 bytes hex = 64 chars

    def test_commit_failure_returns_none(self) -> None:
        """Failed commit returns None."""
        subtensor = MagicMock()
        subtensor.set_reveal_commitment.return_value = (False, None)

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.commit() is None

    def test_commit_exception_returns_none(self) -> None:
        """Exception during commit returns None (after retries exhausted)."""
        provider = DecentralizedSeedProvider(
            subtensor=MagicMock(), wallet=MagicMock(), netuid=46,
        )
        provider._submit_commitment = MagicMock(
            side_effect=RuntimeError("chain error")
        )
        assert provider.commit() is None

    def test_commit_uses_config_blocks(self) -> None:
        """Commit uses blocks_until_reveal from config."""
        subtensor = MagicMock()
        subtensor.set_reveal_commitment.return_value = (True, 10)
        config = RandomnessConfig(blocks_until_reveal=720)

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46, config=config,
        )
        provider.commit()

        call_kwargs = subtensor.set_reveal_commitment.call_args.kwargs
        assert call_kwargs["blocks_until_reveal"] == 720


class TestDecentralizedSeedProviderHarvest:
    def test_harvest_filters_to_validators(self) -> None:
        """Only validator hotkeys are included in seed computation."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "validator_a": ((1, "hex_a"),),
            "validator_b": ((2, "hex_b"),),
            "miner_c": ((3, "hex_c"),),  # not a validator
        }

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        validators = {"validator_a", "validator_b"}
        result = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)

        assert result is not None
        assert result.num_reveals == 2
        assert result.validator_hotkeys == frozenset({"validator_a", "validator_b"})

    def test_harvest_no_validators_returns_none(self) -> None:
        """No validator reveals returns None."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "miner_a": ((1, "hex"),),
        }

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.harvest({"validator_x"}, min_reveal_block=0, committed_hotkeys={"validator_x"}) is None

    def test_harvest_empty_reveals_returns_none(self) -> None:
        """Empty reveals dict returns None."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {}

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.harvest({"validator_a"}, min_reveal_block=0, committed_hotkeys={"validator_a"}) is None

    def test_harvest_exception_returns_none(self) -> None:
        """Exception during fetch returns None (after retries exhausted)."""
        provider = DecentralizedSeedProvider(
            subtensor=MagicMock(), wallet=MagicMock(), netuid=46,
        )
        provider._fetch_revealed_commitments = MagicMock(
            side_effect=RuntimeError("fail")
        )
        assert provider.harvest({"validator_a"}, min_reveal_block=0, committed_hotkeys={"validator_a"}) is None

    def test_harvest_uses_oldest_valid_reveal(self) -> None:
        """When a validator has multiple reveals, the oldest valid is used (anti re-commit)."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((1, "original"), (5, "re_committed")),
            "val_b": ((3, "bbb222"),),
        }
        validators = {"val_a", "val_b"}

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)

        assert result is not None
        expected = combine_reveals({"val_a": "original", "val_b": "bbb222"}, 2**32)
        assert result.seed == expected

    def test_harvest_sorts_unsorted_entries(self) -> None:
        """Entries from chain may be unsorted — oldest by block is still picked."""
        subtensor = MagicMock()
        # Chain returns entries in storage hash order, NOT block order
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((10, "re_committed"), (3, "original")),
            "val_b": ((2, "bbb222"),),
        }
        validators = {"val_a", "val_b"}

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)

        assert result is not None
        # Should use block=3 ("original"), not block=10 ("re_committed")
        expected = combine_reveals({"val_a": "original", "val_b": "bbb222"}, 2**32)
        assert result.seed == expected

    def test_harvest_deterministic_across_calls(self) -> None:
        """Same reveals produce same seed across multiple harvest calls."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((1, "hex_a"),),
            "val_b": ((2, "hex_b"),),
        }
        validators = {"val_a", "val_b"}

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        r1 = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)
        r2 = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)

        assert r1.seed == r2.seed

    def test_harvest_seed_within_modulus(self) -> None:
        """Harvested seed respects config modulus."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((1, "hex_data"),),
            "val_b": ((2, "hex_b"),),
        }
        validators = {"val_a", "val_b"}
        config = RandomnessConfig(seed_modulus=1000)

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46, config=config,
        )
        result = provider.harvest(validators, min_reveal_block=0, committed_hotkeys=validators)

        assert result is not None
        assert 0 <= result.seed < 1000


class TestGetPendingCommitmentHotkeys:
    """Tests for DecentralizedSeedProvider.get_pending_commitment_hotkeys()."""

    def test_detects_raw_commitment(self) -> None:
        """Validator with Raw-format commitment is detected."""
        subtensor = MagicMock()
        result_mock = MagicMock()
        result_mock.value = {"some": "data"}
        subtensor.query.return_value = result_mock
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.get_pending_commitment_hotkeys({"val_a", "val_b"})
        assert result == {"val_a", "val_b"}

    def test_detects_timelock_encrypted_via_decode_error(self) -> None:
        """TimelockEncrypted commitment causes decode error → treated as committed."""
        subtensor = MagicMock()
        subtensor.query.side_effect = Exception("Error decoding TimelockEncrypted")
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.get_pending_commitment_hotkeys({"val_a", "val_b"})
        assert result == {"val_a", "val_b"}

    def test_no_commitment_returns_empty(self) -> None:
        """Validator with no commitment is excluded."""
        subtensor = MagicMock()
        result_mock = MagicMock()
        result_mock.value = None
        subtensor.query.return_value = result_mock
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.get_pending_commitment_hotkeys({"val_a"}) == set()

    def test_mixed_committed_and_not(self) -> None:
        """Mix of committed, decode-error, and no-commitment validators."""
        subtensor = MagicMock()

        committed_result = MagicMock()
        committed_result.value = {"data": "raw"}
        empty_result = MagicMock()
        empty_result.value = None

        def query_side_effect(*, module, name, params):
            hotkey = params[1]
            if hotkey == "val_committed":
                return committed_result
            elif hotkey == "val_timelock":
                raise Exception("Error decoding TimelockEncrypted")
            else:
                return empty_result

        subtensor.query.side_effect = query_side_effect
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.get_pending_commitment_hotkeys(
            {"val_committed", "val_timelock", "val_none"}
        )
        assert result == {"val_committed", "val_timelock"}

    def test_empty_validator_set(self) -> None:
        """Empty validator set returns empty."""
        provider = DecentralizedSeedProvider(
            subtensor=MagicMock(), wallet=MagicMock(), netuid=46,
        )
        assert provider.get_pending_commitment_hotkeys(set()) == set()


class TestCommittedSnapshot:
    """Tests for _save/_load/_delete_committed_snapshot on Validator."""

    @pytest.fixture
    def mock_validator(self, tmp_path: Path) -> MagicMock:
        """Create a minimal mock validator with snapshot methods."""
        # Mock pylon before importing Validator
        _mock = MagicMock()
        sys.modules.setdefault("pylon_client", _mock)
        sys.modules.setdefault("pylon_client.artanis", _mock.artanis)
        sys.modules.setdefault("pylon_client.artanis.unstable", _mock.artanis.unstable)
        from real_estate.validator.validator import Validator

        stub = MagicMock(spec=Validator)
        stub._snapshot_path = tmp_path / "committed_hotkeys.json"
        stub._randomness_config = RandomnessConfig()
        # Bind real methods
        stub._save_committed_snapshot = (
            Validator._save_committed_snapshot.__get__(stub)
        )
        stub._load_committed_snapshot = (
            Validator._load_committed_snapshot.__get__(stub)
        )
        stub._delete_committed_snapshot = (
            Validator._delete_committed_snapshot.__get__(stub)
        )
        return stub

    def test_save_and_load_roundtrip(self, mock_validator: MagicMock) -> None:
        """Save then load returns the same hotkeys."""
        hotkeys = {"val_a", "val_b", "val_c"}
        mock_validator._save_committed_snapshot(hotkeys)
        loaded = mock_validator._load_committed_snapshot()
        assert loaded == hotkeys

    def test_load_missing_file_returns_none(self, mock_validator: MagicMock) -> None:
        """No snapshot file returns None."""
        assert mock_validator._load_committed_snapshot() is None

    def test_load_stale_snapshot_returns_none(
        self, mock_validator: MagicMock,
    ) -> None:
        """Snapshot older than cycle_window_hours is rejected."""
        hotkeys = {"val_a"}
        mock_validator._save_committed_snapshot(hotkeys)

        # Patch the timestamp to be old
        data = json.loads(mock_validator._snapshot_path.read_text())
        old_time = datetime.now(UTC) - timedelta(hours=5)  # > 4h default
        data["timestamp"] = old_time.isoformat()
        mock_validator._snapshot_path.write_text(json.dumps(data))

        assert mock_validator._load_committed_snapshot() is None

    def test_load_fresh_snapshot_accepted(
        self, mock_validator: MagicMock,
    ) -> None:
        """Snapshot within cycle_window_hours is accepted."""
        hotkeys = {"val_a", "val_b"}
        mock_validator._save_committed_snapshot(hotkeys)

        # Patch timestamp to 1 hour ago (within 4h window)
        data = json.loads(mock_validator._snapshot_path.read_text())
        recent_time = datetime.now(UTC) - timedelta(hours=1)
        data["timestamp"] = recent_time.isoformat()
        mock_validator._snapshot_path.write_text(json.dumps(data))

        assert mock_validator._load_committed_snapshot() == hotkeys

    def test_delete_removes_file(self, mock_validator: MagicMock) -> None:
        """Delete removes the snapshot file."""
        mock_validator._save_committed_snapshot({"val_a"})
        assert mock_validator._snapshot_path.exists()

        mock_validator._delete_committed_snapshot()
        assert not mock_validator._snapshot_path.exists()

    def test_delete_missing_file_no_error(self, mock_validator: MagicMock) -> None:
        """Delete on missing file does not raise."""
        assert not mock_validator._snapshot_path.exists()
        mock_validator._delete_committed_snapshot()  # should not raise

    def test_save_is_atomic(self, mock_validator: MagicMock) -> None:
        """Save uses atomic write — no partial files on success."""
        mock_validator._save_committed_snapshot({"val_a"})
        # File should exist and be valid JSON
        data = json.loads(mock_validator._snapshot_path.read_text())
        assert set(data["hotkeys"]) == {"val_a"}
        assert "timestamp" in data

    def test_load_corrupted_file_returns_none(
        self, mock_validator: MagicMock,
    ) -> None:
        """Corrupted snapshot file returns None (not crash)."""
        mock_validator._snapshot_path.write_text("not valid json{{{")
        assert mock_validator._load_committed_snapshot() is None
