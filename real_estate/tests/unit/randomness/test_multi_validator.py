"""Multi-validator consensus tests for decentralized randomness.

Simulates N validators sharing the same mocked chain state.
Each validator is a separate DecentralizedSeedProvider instance
with its own wallet, but all reading from the same revealed commitments.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from real_estate.randomness import (
    DecentralizedSeedProvider,
    RandomnessConfig,
    combine_reveals,
)


def _make_provider(
    chain_state: dict[str, tuple[tuple[int, str], ...]],
    hotkey: str = "val_default",
) -> DecentralizedSeedProvider:
    """Create a provider with mocked subtensor returning *chain_state*."""
    subtensor = MagicMock()
    subtensor.get_all_revealed_commitments.return_value = chain_state
    wallet = MagicMock()
    wallet.hotkey.ss58_address = hotkey
    return DecentralizedSeedProvider(
        subtensor=subtensor, wallet=wallet, netuid=46,
    )


# Shared chain data used by multiple tests
CHAIN_3V = {
    "val_a": ((1, "aaa111"),),
    "val_b": ((2, "bbb222"),),
    "val_c": ((3, "ccc333"),),
}

VALIDATOR_HOTKEYS_3 = {"val_a", "val_b", "val_c"}


class TestMultiValidatorConsensus:
    """All validators converge on the same seed when seeing the same reveals."""

    def test_three_validators_same_seed(self) -> None:
        """3 validators each harvest -> identical SeedResult.seed."""
        providers = [
            _make_provider(CHAIN_3V, hotkey=hk)
            for hk in ("val_a", "val_b", "val_c")
        ]

        results = [
            p.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=VALIDATOR_HOTKEYS_3)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1, f"Expected 1 unique seed, got {seeds}"
        for r in results:
            assert r.num_reveals == 3
            assert r.validator_hotkeys == frozenset(VALIDATOR_HOTKEYS_3)

    def test_partial_participation_two_of_three(self) -> None:
        """2 of 3 validators reveal -> harvesters still agree on seed."""
        chain_state = {
            "val_a": ((1, "aaa111"),),
            "val_b": ((2, "bbb222"),),
            # val_c did not reveal
        }
        committed = {"val_a", "val_b", "val_c"}

        providers = [
            _make_provider(chain_state, hotkey=hk)
            for hk in ("val_a", "val_b", "val_c")
        ]

        results = [
            p.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=committed)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1
        for r in results:
            assert r.num_reveals == 2
            assert r.validator_hotkeys == frozenset({"val_a", "val_b"})

    def test_single_validator_below_quorum_returns_none(self) -> None:
        """Only 1 validator reveals -> below min_quorum, returns None."""
        chain_state = {"val_a": ((1, "aaa111"),)}

        providers = [
            _make_provider(chain_state, hotkey=hk)
            for hk in ("val_a", "val_b")
        ]

        results = [
            p.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=VALIDATOR_HOTKEYS_3)
            for p in providers
        ]
        assert all(r is None for r in results)

    def test_no_reveals_all_get_none(self) -> None:
        """No validators reveal -> all harvest returns None."""
        providers = [
            _make_provider({}, hotkey=hk) for hk in ("val_a", "val_b")
        ]

        for p in providers:
            assert p.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=VALIDATOR_HOTKEYS_3) is None

    def test_late_joiner_gets_same_seed(self) -> None:
        """Validator that didn't commit can still harvest others' reveals
        (as long as committed_hotkeys includes them)."""
        chain_state = {
            "val_a": ((1, "aaa111"),),
            "val_b": ((2, "bbb222"),),
        }
        committed = {"val_a", "val_b"}

        provider_a = _make_provider(chain_state, hotkey="val_a")
        provider_c = _make_provider(chain_state, hotkey="val_c")  # different validator

        result_a = provider_a.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=committed)
        result_c = provider_c.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=committed)

        assert result_a is not None
        assert result_c is not None
        assert result_a.seed == result_c.seed
        assert result_a.num_reveals == 2

    def test_extra_miner_reveals_ignored(self) -> None:
        """Miner reveals on chain are filtered out — don't affect seed."""
        chain_with_miner = {
            "val_a": ((1, "aaa111"),),
            "val_b": ((2, "bbb222"),),
            "miner_x": ((9, "miner_data"),),
        }
        chain_without_miner = {
            "val_a": ((1, "aaa111"),),
            "val_b": ((2, "bbb222"),),
        }
        validator_set = {"val_a", "val_b"}

        p_with = _make_provider(chain_with_miner)
        p_without = _make_provider(chain_without_miner)

        r_with = p_with.harvest(validator_set, min_reveal_round=0, committed_hotkeys=validator_set)
        r_without = p_without.harvest(validator_set, min_reveal_round=0, committed_hotkeys=validator_set)

        assert r_with.seed == r_without.seed
        assert r_with.num_reveals == 2
        assert "miner_x" not in r_with.validator_hotkeys

    def test_validator_subset_different_seed(self) -> None:
        """Different validator sets produce different seeds."""
        provider_small = _make_provider(CHAIN_3V)
        provider_large = _make_provider(CHAIN_3V)

        small_set = {"val_a", "val_b"}
        large_set = {"val_a", "val_b", "val_c"}
        r_small = provider_small.harvest(small_set, min_reveal_round=0, committed_hotkeys=small_set)
        r_large = provider_large.harvest(large_set, min_reveal_round=0, committed_hotkeys=large_set)

        assert r_small.seed != r_large.seed

    def test_stale_plus_fresh_reveals_uses_latest(self) -> None:
        """Validator with multiple reveals -> latest used, all harvesters agree."""
        chain_state = {
            "val_a": ((1, "old_hex"), (2, "new_hex")),
            "val_b": ((1, "bbb222"),),
        }
        validator_set = {"val_a", "val_b"}

        providers = [
            _make_provider(chain_state, hotkey=hk) for hk in ("val_a", "val_b")
        ]

        results = [
            p.harvest(validator_set, min_reveal_round=0, committed_hotkeys=validator_set)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1

        # Verify it used the latest reveal, not the old one
        expected = combine_reveals({"val_a": "new_hex", "val_b": "bbb222"}, 2**32)
        assert results[0].seed == expected

    def test_custom_modulus_consensus(self) -> None:
        """Validators with same custom modulus agree on seed."""
        config = RandomnessConfig(seed_modulus=1000)

        def make_with_config(hotkey: str) -> DecentralizedSeedProvider:
            subtensor = MagicMock()
            subtensor.get_all_revealed_commitments.return_value = CHAIN_3V
            wallet = MagicMock()
            return DecentralizedSeedProvider(
                subtensor=subtensor, wallet=wallet, netuid=46, config=config,
            )

        providers = [make_with_config(hk) for hk in ("val_a", "val_b", "val_c")]
        results = [
            p.harvest(VALIDATOR_HOTKEYS_3, min_reveal_round=0, committed_hotkeys=VALIDATOR_HOTKEYS_3)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1
        assert all(0 <= r.seed < 1000 for r in results)


class TestEpochAwareCommitTiming:
    """Tests for DecentralizedSeedProvider.get_target_commit_block()."""

    def _make_provider_with_tempo(
        self,
        tempo: int | None,
        blocks_since_last_step: int = 0,
    ) -> DecentralizedSeedProvider:
        subtensor = MagicMock()
        subtensor.tempo.return_value = tempo
        subtensor.blocks_since_last_step.return_value = blocks_since_last_step
        wallet = MagicMock()
        return DecentralizedSeedProvider(
            subtensor=subtensor, wallet=wallet, netuid=46,
        )

    def test_target_commit_block_basic(self) -> None:
        """Finds epoch boundary before latest_commit deadline."""
        # tempo=360, current_block=500, blocks_since_last_step=140
        # last_epoch_start = 500 - 140 = 360
        # latest_commit = 2000 - 360 - 90 = 1550
        # epochs_ahead = (1550 - 360) // 360 = 3
        # target = 360 + 3*360 = 1440, > 500 → valid
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=140,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=500
        )
        assert result == 1440

    def test_target_commit_block_with_netuid_offset(self) -> None:
        """Uses netuid-specific epoch offset, not block-0-aligned."""
        # tempo=360, current_block=500, blocks_since_last_step=100
        # last_epoch_start = 500 - 100 = 400 (offset epochs: 400, 760, 1120, 1480...)
        # latest_commit = 2000 - 360 - 90 = 1550
        # epochs_ahead = (1550 - 400) // 360 = 3
        # target = 400 + 3*360 = 1480, > 500 → valid
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=100,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=500
        )
        assert result == 1480

    def test_target_commit_block_already_past(self) -> None:
        """Returns None when the computed target is already past."""
        # current_block=1500, blocks_since_last_step=60
        # last_epoch_start = 1440, latest_commit = 1550
        # epochs_ahead = (1550 - 1440) // 360 = 0
        # target = 1440, <= 1500 → None
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=60,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=1500
        )
        assert result is None

    def test_target_commit_block_chain_query_fails(self) -> None:
        """Returns None when tempo() returns None."""
        provider = self._make_provider_with_tempo(tempo=None)
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=500
        )
        assert result is None

    def test_target_commit_block_zero_tempo(self) -> None:
        """Returns None when tempo is zero."""
        provider = self._make_provider_with_tempo(tempo=0)
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=500
        )
        assert result is None

    def test_target_commit_block_tight_timing(self) -> None:
        """Returns None when eval is too close for any epoch boundary."""
        # current_block=1900, blocks_since_last_step=100
        # last_epoch_start = 1800, latest_commit = 2100-360-90 = 1650
        # epochs_ahead = (1650 - 1800) // 360 = -1 → negative
        # target = 1800 + (-1)*360 = 1440, <= 1900 → None
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=100,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2100, current_block=1900
        )
        assert result is None

    def test_target_commit_block_exactly_at_current(self) -> None:
        """Returns None when target == current_block (must be strictly future)."""
        # tempo=360, current_block=1440, blocks_since_last_step=0
        # last_epoch_start = 1440 - 0 = 1440
        # latest_commit = 2000 - 360 - 90 = 1550
        # epochs_ahead = (1550 - 1440) // 360 = 0
        # target = 1440 + 0 = 1440, == 1440 → None
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=0,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=1440
        )
        assert result is None

    def test_target_commit_block_one_past_current(self) -> None:
        """Returns target when it's exactly one block ahead of current."""
        # tempo=360, current_block=1439, blocks_since_last_step=359
        # last_epoch_start = 1439 - 359 = 1080
        # latest_commit = 2000 - 360 - 90 = 1550
        # epochs_ahead = (1550 - 1080) // 360 = 1
        # target = 1080 + 1*360 = 1440, > 1439 → valid
        provider = self._make_provider_with_tempo(
            tempo=360, blocks_since_last_step=359,
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=1439
        )
        assert result == 1440

    def test_target_commit_block_exception_on_tempo(self) -> None:
        """Returns None when epoch query raises (after retries exhausted)."""
        provider = DecentralizedSeedProvider(
            subtensor=MagicMock(), wallet=MagicMock(), netuid=46,
        )
        provider._query_epoch_info = MagicMock(
            side_effect=Exception("chain down")
        )
        result = provider.get_target_commit_block(
            eval_block_estimate=2000, current_block=500
        )
        assert result is None

    def test_multiple_validators_same_target(self) -> None:
        """All validators with same chain state get same target block."""
        targets = []
        for _ in range(3):
            provider = self._make_provider_with_tempo(
                tempo=360, blocks_since_last_step=140,
            )
            targets.append(
                provider.get_target_commit_block(
                    eval_block_estimate=2000, current_block=500
                )
            )
        assert len(set(targets)) == 1
        assert targets[0] == 1440


class TestReHarvestRecovery:
    """Tests for harvest recovery after crash/restart."""

    def test_harvest_after_restart_gets_existing_reveals(self) -> None:
        """Fresh provider can harvest reveals already on chain."""
        chain_state = {
            "val_a": ((1, "aaa111"),),
            "val_b": ((2, "bbb222"),),
        }
        validators = {"val_a", "val_b"}
        # Simulate a freshly created provider (as after restart)
        provider = _make_provider(chain_state, hotkey="val_a")
        result = provider.harvest(validators, min_reveal_round=0, committed_hotkeys=validators)

        assert result is not None
        assert result.num_reveals == 2
        assert result.seed == combine_reveals(
            {"val_a": "aaa111", "val_b": "bbb222"},
            2**32,
        )


class TestGetEpochWaitSeconds:
    """Tests for Validator._get_epoch_wait_seconds (validator-side wiring)."""

    def _make_validator_stub(
        self,
        current_block: int,
        target_block: int | None,
    ) -> MagicMock:
        """Create a minimal mock with _seed_provider and _get_epoch_wait_seconds."""
        from real_estate.validator.validator import Validator

        seed_provider = MagicMock()
        seed_provider.get_target_commit_block.return_value = target_block

        stub = MagicMock(spec=Validator)
        stub._seed_provider = seed_provider
        stub._randomness_config = RandomnessConfig()
        # Bind the real method
        stub._get_epoch_wait_seconds = (
            Validator._get_epoch_wait_seconds.__get__(stub)
        )
        # Store for patching ttl_get_block
        stub._current_block = current_block
        return stub

    @pytest.mark.asyncio
    @patch("real_estate.validator.validator.ttl_get_block")
    async def test_returns_wait_seconds(self, mock_block: MagicMock) -> None:
        """Returns block-based wait when target is in the future."""
        stub = self._make_validator_stub(current_block=500, target_block=600)
        mock_block.return_value = 500
        next_eval = datetime.now(UTC) + timedelta(hours=4)

        result = await stub._get_epoch_wait_seconds(next_eval)

        assert result == (600 - 500) * 12  # 1200 seconds

    @pytest.mark.asyncio
    @patch("real_estate.validator.validator.ttl_get_block")
    async def test_returns_none_when_target_is_none(self, mock_block: MagicMock) -> None:
        """Returns None when provider can't find a valid target."""
        stub = self._make_validator_stub(current_block=500, target_block=None)
        mock_block.return_value = 500
        next_eval = datetime.now(UTC) + timedelta(hours=4)

        assert await stub._get_epoch_wait_seconds(next_eval) is None

    @pytest.mark.asyncio
    @patch("real_estate.validator.validator.ttl_get_block")
    async def test_returns_none_when_target_already_past(self, mock_block: MagicMock) -> None:
        """Returns None when target block <= current block."""
        stub = self._make_validator_stub(current_block=500, target_block=400)
        mock_block.return_value = 500
        next_eval = datetime.now(UTC) + timedelta(hours=4)

        assert await stub._get_epoch_wait_seconds(next_eval) is None

    @pytest.mark.asyncio
    @patch("real_estate.validator.validator.ttl_get_block")
    async def test_returns_none_on_exception(self, mock_block: MagicMock) -> None:
        """Returns None when ttl_get_block raises."""
        stub = self._make_validator_stub(current_block=500, target_block=600)
        mock_block.side_effect = Exception("fail")
        next_eval = datetime.now(UTC) + timedelta(hours=4)

        assert await stub._get_epoch_wait_seconds(next_eval) is None


class TestRevealFreshness:
    """Tests for min_reveal_round freshness filtering in harvest()."""

    def test_stale_reveals_filtered_out(self) -> None:
        """Reveals with round below threshold are skipped."""
        chain_state = {
            "val_a": ((100, "aaa111"),),  # stale
            "val_b": ((500, "bbb222"),),  # fresh
            "val_c": ((600, "ccc333"),),  # fresh
        }
        validators = {"val_a", "val_b", "val_c"}
        provider = _make_provider(chain_state)

        result = provider.harvest(validators, min_reveal_round=200, committed_hotkeys=validators)

        assert result is not None
        assert result.num_reveals == 2
        assert "val_b" in result.validator_hotkeys
        assert "val_c" in result.validator_hotkeys
        assert "val_a" not in result.validator_hotkeys

    def test_all_stale_returns_none(self) -> None:
        """All reveals below threshold -> None."""
        chain_state = {
            "val_a": ((10, "aaa111"),),
            "val_b": ((20, "bbb222"),),
        }
        validators = {"val_a", "val_b"}
        provider = _make_provider(chain_state)

        assert provider.harvest(validators, min_reveal_round=100, committed_hotkeys=validators) is None

    def test_exact_threshold_accepted(self) -> None:
        """Reveal round == min_reveal_round is NOT stale (only < is stale)."""
        chain_state = {
            "val_a": ((200, "aaa111"),),
            "val_b": ((300, "bbb222"),),
        }
        provider = _make_provider(chain_state)
        validators = {"val_a", "val_b"}

        result = provider.harvest(validators, min_reveal_round=200, committed_hotkeys=validators)

        assert result is not None
        assert result.num_reveals == 2

    def test_freshness_consensus(self) -> None:
        """Multiple validators with same threshold get same seed."""
        chain_state = {
            "val_a": ((100, "old_a"),),   # stale
            "val_b": ((500, "bbb222"),),  # fresh
            "val_c": ((600, "ccc333"),),  # fresh
        }
        validator_set = {"val_a", "val_b", "val_c"}

        providers = [
            _make_provider(chain_state, hotkey=hk)
            for hk in ("val_a", "val_b", "val_c")
        ]

        results = [
            p.harvest(validator_set, min_reveal_round=200, committed_hotkeys=validator_set)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1
        for r in results:
            assert r.num_reveals == 2
            assert r.validator_hotkeys == frozenset({"val_b", "val_c"})

    def test_latest_entry_checked_not_all(self) -> None:
        """Only the latest (last) entry's round is checked for freshness."""
        chain_state = {
            # Old entry is stale, but latest entry is fresh
            "val_a": ((50, "old_hex"), (500, "new_hex")),
            "val_b": ((600, "bbb222"),),
        }
        validators = {"val_a", "val_b"}
        provider = _make_provider(chain_state)

        result = provider.harvest(validators, min_reveal_round=200, committed_hotkeys=validators)

        assert result is not None
        assert "val_a" in result.validator_hotkeys


class TestGetMinRevealRound:
    """Tests for get_min_reveal_round() drand round threshold computation."""

    def test_basic_computation(self) -> None:
        """Computes threshold from current drand round and max age."""
        subtensor = MagicMock()
        subtensor.last_drand_round.return_value = 10000
        wallet = MagicMock()
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=wallet, netuid=46,
        )

        # 4320 seconds / 3s per round = 1440 rounds back
        result = provider.get_min_reveal_round(max_age_seconds=4320)

        assert result == 10000 - 1440

    def test_returns_none_on_chain_failure(self) -> None:
        """Returns None when last_drand_round raises (after retries exhausted)."""
        provider = DecentralizedSeedProvider(
            subtensor=MagicMock(), wallet=MagicMock(), netuid=46,
        )
        provider.last_drand_round = MagicMock(
            side_effect=Exception("chain down")
        )

        assert provider.get_min_reveal_round(max_age_seconds=4320) is None

    def test_clamps_to_zero(self) -> None:
        """Returns 0 when max_age exceeds current drand round."""
        subtensor = MagicMock()
        subtensor.last_drand_round.return_value = 100
        wallet = MagicMock()
        provider = DecentralizedSeedProvider(
            subtensor=subtensor, wallet=wallet, netuid=46,
        )

        # 100000 seconds / 3s = 33333 rounds back, but current is only 100
        result = provider.get_min_reveal_round(max_age_seconds=100000)

        assert result == 0


class TestCommittedHotkeysFilter:
    """Tests for committed_hotkeys (pre-reveal snapshot) anti-gaming filter."""

    def test_late_commit_rejected(self) -> None:
        """Reveal from validator not in pre-reveal snapshot is rejected."""
        chain_state = {
            "val_a": ((500, "aaa111"),),
            "val_b": ((500, "bbb222"),),
            "attacker": ((504, "evil_value"),),  # committed after snapshot
        }
        provider = _make_provider(chain_state)

        result = provider.harvest(
            {"val_a", "val_b", "attacker"},
            min_reveal_round=0,
            committed_hotkeys={"val_a", "val_b"},  # attacker not in snapshot
        )

        assert result is not None
        assert result.num_reveals == 2
        assert "attacker" not in result.validator_hotkeys
        # Seed should match as if attacker didn't exist
        expected = combine_reveals(
            {"val_a": "aaa111", "val_b": "bbb222"}, 2**32,
        )
        assert result.seed == expected

    def test_all_committed_accepted(self) -> None:
        """All validators in snapshot have their reveals accepted."""
        chain_state = {
            "val_a": ((500, "aaa111"),),
            "val_b": ((500, "bbb222"),),
        }
        provider = _make_provider(chain_state)

        result = provider.harvest(
            {"val_a", "val_b"},
            min_reveal_round=0,
            committed_hotkeys={"val_a", "val_b"},
        )

        assert result is not None
        assert result.num_reveals == 2

    def test_all_late_returns_none(self) -> None:
        """If no reveals match the snapshot, returns None."""
        chain_state = {
            "attacker_a": ((500, "aaa"),),
            "attacker_b": ((500, "bbb"),),
        }
        provider = _make_provider(chain_state)

        result = provider.harvest(
            {"attacker_a", "attacker_b"},
            min_reveal_round=0,
            committed_hotkeys=set(),  # nobody committed on time
        )

        assert result is None

    def test_combined_filters_stale_and_late(self) -> None:
        """Both freshness and committed_hotkeys filters work together."""
        chain_state = {
            "val_a": ((500, "aaa111"),),   # fresh + committed
            "val_b": ((10, "old_bbb"),),   # stale (old round)
            "late_c": ((500, "ccc333"),),  # fresh but late commit
            "val_d": ((600, "ddd444"),),   # fresh + committed
        }
        provider = _make_provider(chain_state)

        result = provider.harvest(
            {"val_a", "val_b", "late_c", "val_d"},
            min_reveal_round=200,
            committed_hotkeys={"val_a", "val_b", "val_d"},
        )

        assert result is not None
        assert result.num_reveals == 2
        assert result.validator_hotkeys == frozenset({"val_a", "val_d"})

    def test_consensus_with_committed_filter(self) -> None:
        """All validators with same snapshot get same seed."""
        chain_state = {
            "val_a": ((500, "aaa111"),),
            "val_b": ((500, "bbb222"),),
            "attacker": ((504, "evil"),),
        }
        committed = {"val_a", "val_b"}
        validator_set = {"val_a", "val_b", "attacker"}

        providers = [
            _make_provider(chain_state, hotkey=hk)
            for hk in ("val_a", "val_b", "attacker")
        ]

        results = [
            p.harvest(validator_set, min_reveal_round=0, committed_hotkeys=committed)
            for p in providers
        ]

        seeds = {r.seed for r in results}
        assert len(seeds) == 1
        assert all(r.num_reveals == 2 for r in results)


