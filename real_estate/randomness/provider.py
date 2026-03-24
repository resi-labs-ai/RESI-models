"""Decentralized seed provider using timelocked chain commitments."""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import TYPE_CHECKING

from .models import RandomnessConfig, SeedResult

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)


def combine_reveals(reveals: dict[str, str], modulus: int) -> int:
    """
    Deterministic combination of all validator reveals into a seed.

    Uses SHA256 for uniform distribution and manipulation resistance
    (changing one input completely changes the output).

    Args:
        reveals: Mapping of hotkey -> revealed hex value.
        modulus: Seed range [0, modulus].

    Returns:
        Deterministic seed integer.
    """
    sorted_values = [v for _, v in sorted(reveals.items())]
    combined = "|".join(sorted_values)
    hash_hex = hashlib.sha256(combined.encode()).hexdigest()
    return int(hash_hex, 16) % modulus


class DecentralizedSeedProvider:
    """
    Manages commit and harvest lifecycle for decentralized randomness.

    Each validator submits a timelocked commitment (random value) to the chain
    via bittensor SDK's Drand timelock encryption. After the timelock period,
    all validators harvest the revealed values and combine them into a shared seed.

    Uses bt.subtensor directly (not Pylon) for timelocked commitment operations.
    """

    def __init__(
        self,
        subtensor: bt.subtensor,
        wallet: bt.wallet,
        netuid: int,
        config: RandomnessConfig | None = None,
    ):
        self._subtensor = subtensor
        self._wallet = wallet
        self._netuid = netuid
        self._config = config or RandomnessConfig()

    def commit(self) -> int | None:
        """
        Generate random value and submit timelocked commitment to chain.

        Returns:
            Reveal round number if successful, None on failure.
        """
        value = secrets.token_hex(32)

        try:
            success, reveal_round = self._subtensor.set_reveal_commitment(
                wallet=self._wallet,
                netuid=self._netuid,
                data=value,
                blocks_until_reveal=self._config.blocks_until_reveal,
            )
        except Exception as e:
            logger.error(f"Failed to submit randomness commitment: {e}")
            return None

        if not success:
            logger.error("set_reveal_commitment returned False")
            return None

        logger.info(
            f"Randomness commitment submitted (reveal_round={reveal_round})"
        )
        return reveal_round

    def harvest(self, validator_hotkeys: set[str]) -> SeedResult | None:
        """
        Read all revealed commitments, filter to validators, combine into seed.

        Args:
            validator_hotkeys: Set of hotkeys with validator_permit.

        Returns:
            SeedResult with combined seed, or None if no reveals available.
        """
        try:
            all_revealed: dict[str, tuple[tuple[int, str], ...]] = (
                self._subtensor.get_all_revealed_commitments(self._netuid)
            )
        except Exception as e:
            logger.error(f"Failed to fetch revealed commitments: {e}")
            return None

        # Filter to validator hotkeys and extract latest reveal per validator
        reveals: dict[str, str] = {}
        for hotkey, entries in all_revealed.items():
            if hotkey not in validator_hotkeys:
                continue
            if not entries:
                continue
            # Each entry is (round, hex_data). Use the latest (last) entry.
            _, hex_data = entries[-1]
            reveals[hotkey] = hex_data

        if not reveals:
            logger.warning(
                f"No validator reveals found "
                f"(total reveals: {len(all_revealed)}, "
                f"validator hotkeys: {len(validator_hotkeys)})"
            )
            return None

        seed = combine_reveals(reveals, self._config.seed_modulus)

        result = SeedResult(
            seed=seed,
            num_reveals=len(reveals),
            validator_hotkeys=frozenset(reveals.keys()),
        )

        logger.info(
            f"Harvested seed={seed} from {result.num_reveals} validators"
        )
        return result
