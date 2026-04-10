"""Localnet observational tests for decentralized randomness.

These tests observe an already-running localnet. They require
``SUBTENSOR_NETWORK`` env var pointing at a subtensor websocket endpoint.

Run:
    SUBTENSOR_NETWORK=ws://... pytest -m localnet \
        real_estate/tests/integration/localnet/test_randomness.py -v
"""

from __future__ import annotations

import os
import time

import bittensor as bt
import pytest

from real_estate.randomness import DecentralizedSeedProvider, RandomnessConfig

pytestmark = pytest.mark.localnet


@pytest.fixture(scope="session")
def subtensor_endpoint():
    """Subtensor websocket endpoint, skip if not set."""
    endpoint = os.environ.get("SUBTENSOR_NETWORK", "")
    if not endpoint:
        pytest.skip("SUBTENSOR_NETWORK not set")
    return endpoint


@pytest.fixture(scope="session")
def subtensor(subtensor_endpoint):
    """Live subtensor connection."""
    return bt.subtensor(network=subtensor_endpoint)


@pytest.fixture(scope="session")
def netuid():
    return int(os.environ.get("NETUID", "46"))


@pytest.fixture(scope="session")
def validator_hotkeys(subtensor, netuid):
    """Hotkeys with validator_permit from metagraph."""
    metagraph = subtensor.metagraph(netuid)
    return {
        n.hotkey for n in metagraph.neurons if n.validator_permit
    }


class TestRandomnessObservation:
    """Observe randomness state on a running localnet."""

    def test_read_revealed_commitments(
        self, subtensor, netuid,
    ) -> None:
        """Can read revealed commitments from chain (may be empty)."""
        result = subtensor.get_all_revealed_commitments(netuid)
        assert isinstance(result, dict)

    def test_harvest_produces_valid_seed(
        self, subtensor, netuid, validator_hotkeys,
    ) -> None:
        """If reveals exist on chain, harvest produces a valid SeedResult."""
        provider = DecentralizedSeedProvider(
            subtensor=subtensor,
            wallet=bt.wallet(),  # dummy — harvest doesn't sign
            netuid=netuid,
        )
        result = provider.harvest(validator_hotkeys, min_reveal_block=0, committed_hotkeys=validator_hotkeys)

        if result is None:
            pytest.skip("No validator reveals on chain — nothing to harvest")

        assert result.seed >= 0
        assert result.num_reveals > 0
        assert len(result.validator_hotkeys) == result.num_reveals

    def test_harvest_is_deterministic(
        self, subtensor, netuid, validator_hotkeys,
    ) -> None:
        """Harvesting twice from same chain state -> same seed."""
        provider = DecentralizedSeedProvider(
            subtensor=subtensor,
            wallet=bt.wallet(),
            netuid=netuid,
        )
        r1 = provider.harvest(validator_hotkeys, min_reveal_block=0, committed_hotkeys=validator_hotkeys)
        r2 = provider.harvest(validator_hotkeys, min_reveal_block=0, committed_hotkeys=validator_hotkeys)

        if r1 is None:
            pytest.skip("No validator reveals on chain")

        assert r1.seed == r2.seed
        assert r1.num_reveals == r2.num_reveals

    @pytest.mark.slow
    def test_commitment_round_trip(
        self, subtensor, netuid, validator_hotkeys,
    ) -> None:
        """Submit commitment and verify it appears in reveals after timelock.

        This test is SLOW — timelock duration depends on config.
        Use a short ``blocks_until_reveal`` (e.g. 5 blocks ~1 min) for practical runs.
        """
        blocks = int(os.environ.get("BLOCKS_UNTIL_REVEAL", "5"))
        config = RandomnessConfig(blocks_until_reveal=blocks)

        wallet_name = os.environ.get("WALLET_NAME", "default")
        hotkey_name = os.environ.get("WALLET_HOTKEY", "default")
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

        provider = DecentralizedSeedProvider(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            config=config,
        )

        reveal_round = provider.commit()
        assert reveal_round is not None, "Commitment failed"

        # Wait for timelock to expire (blocks * 12s + buffer)
        wait_seconds = blocks * 12 + 60
        time.sleep(wait_seconds)

        # Our hotkey should now appear in reveals
        our_hotkey = wallet.hotkey.ss58_address
        all_hotkeys = validator_hotkeys | {our_hotkey}
        result = provider.harvest(all_hotkeys, min_reveal_block=0, committed_hotkeys=all_hotkeys)

        assert result is not None, "Harvest returned None after commitment"
        assert our_hotkey in result.validator_hotkeys
