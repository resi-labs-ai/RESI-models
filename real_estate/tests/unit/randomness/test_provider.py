"""Tests for DecentralizedSeedProvider and combine_reveals."""

from unittest.mock import MagicMock, PropertyMock

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
        """Exception during commit returns None."""
        subtensor = MagicMock()
        subtensor.set_reveal_commitment.side_effect = RuntimeError("chain error")

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
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
        result = provider.harvest({"validator_a", "validator_b"})

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
        assert provider.harvest({"validator_x"}) is None

    def test_harvest_empty_reveals_returns_none(self) -> None:
        """Empty reveals dict returns None."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {}

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.harvest({"validator_a"}) is None

    def test_harvest_exception_returns_none(self) -> None:
        """Exception during fetch returns None."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.side_effect = RuntimeError("fail")

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        assert provider.harvest({"validator_a"}) is None

    def test_harvest_uses_latest_reveal(self) -> None:
        """When a validator has multiple reveals, the latest (last) is used."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((1, "old_hex"), (2, "new_hex")),
        }

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46,
        )
        result = provider.harvest({"val_a"})

        assert result is not None
        # Verify determinism with the latest value
        expected_seed = combine_reveals({"val_a": "new_hex"}, 2**32)
        assert result.seed == expected_seed

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
        r1 = provider.harvest(validators)
        r2 = provider.harvest(validators)

        assert r1.seed == r2.seed

    def test_harvest_seed_within_modulus(self) -> None:
        """Harvested seed respects config modulus."""
        subtensor = MagicMock()
        subtensor.get_all_revealed_commitments.return_value = {
            "val_a": ((1, "hex_data"),),
        }
        config = RandomnessConfig(seed_modulus=1000)

        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=MagicMock(), netuid=46, config=config,
        )
        result = provider.harvest({"val_a"})

        assert result is not None
        assert 0 <= result.seed < 1000
