"""Decentralized seed provider using timelocked chain commitments."""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import TYPE_CHECKING

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_fixed,
)

from .models import RandomnessConfig, SeedResult

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)

# Retry policy for read-only subtensor queries.
# 3 attempts × 5s wait = max ~15s overhead per call.
_subtensor_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def combine_reveals(reveals: dict[str, str], modulus: int) -> int:
    """
    Deterministic combination of all validator reveals into a seed.

    Uses SHA256 for uniform distribution and manipulation resistance
    (changing one input completely changes the output).

    Args:
        reveals: Mapping of hotkey -> revealed hex value.
        modulus: Seed range [0, modulus).

    Returns:
        Deterministic seed integer.

    Raises:
        ValueError: If reveals is empty.
    """
    if not reveals:
        raise ValueError("reveals must not be empty")
    sorted_values = [v for _, v in sorted(reveals.items())]
    combined = "|".join(sorted_values)
    hash_hex = hashlib.sha256(combined.encode()).hexdigest()
    return int(hash_hex, 16) % modulus


def _oldest_valid_reveal(
    entries: tuple[tuple[int, str], ...],
    min_reveal_block: int,
) -> str | None:
    """Pick the oldest valid reveal from a list of (block, data) entries.

    Uses oldest (not latest) to prevent re-commit attacks: a validator
    who re-commits after seeing Drand reveals can't overwrite their
    original timelocked value.

    Chain entries are NOT sorted by block number — they come in storage
    hash order — so we sort explicitly.
    """
    valid = [
        (blk, data)
        for blk, data in entries
        if blk >= min_reveal_block and isinstance(data, str) and data
    ]
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    return valid[0][1]


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

    @_subtensor_retry
    def last_drand_round(self) -> int:
        """Proxy to subtensor.last_drand_round()."""
        return self._subtensor.last_drand_round()

    def get_pending_commitment_hotkeys(self, validator_hotkeys: set[str]) -> set[str]:
        """Return which validators have pending (unrevealed) commitments.

        Checks each validator individually rather than bulk query_map,
        because bittensor's SCALE type decoder silently drops
        TimelockEncrypted entries during bulk iteration — meaning
        query_map never yields our Drand-timelocked commitments.

        A single-key query that fails to decode raises an exception,
        which we catch and treat as "commitment exists" (the data is
        there, just in TimelockEncrypted format the SDK can't parse).

        Args:
            validator_hotkeys: Set of hotkeys to check.

        Returns:
            Subset of validator_hotkeys that have pending commitments.
        """
        committed: set[str] = set()
        for hk in validator_hotkeys:
            if self._has_commitment(hk):
                committed.add(hk)
        return committed

    def _has_commitment(self, hotkey: str) -> bool:
        """Check if a hotkey has any commitment on chain.

        Returns True if the storage key has data (regardless of format).
        TimelockEncrypted entries cause decode errors — we treat those
        as "commitment exists" since the error proves data is present.
        """
        try:
            result = self._subtensor.query(
                module="Commitments",
                name="CommitmentOf",
                params=[self._netuid, hotkey],
            )
            # None / empty = no commitment
            return result is not None and result.value is not None
        except Exception:
            # Decode error → data exists but can't be parsed
            # (TimelockEncrypted format). This IS a pending commitment.
            logger.debug(
                f"Commitment decode error for {hotkey[:16]}... (treating as committed)"
            )
            return True

    def commit(self) -> int | None:
        """
        Generate random value and submit timelocked commitment to chain.

        Returns:
            Reveal round number if successful, None on failure.
        """
        value = secrets.token_hex(32)

        try:
            success, reveal_round = self._submit_commitment(value)
        except Exception as e:
            logger.error(f"Failed to submit randomness commitment: {e}", exc_info=True)
            return None

        if not success:
            logger.error("set_reveal_commitment returned False")
            return None

        logger.info(f"Randomness commitment submitted (reveal_round={reveal_round})")
        return reveal_round

    @_subtensor_retry
    def _submit_commitment(self, value: str) -> tuple[bool, int]:
        """Submit commitment to chain with retries.

        Separated from commit() so retries reuse the same random value.
        """
        return self._subtensor.set_reveal_commitment(
            wallet=self._wallet,
            netuid=self._netuid,
            data=value,
            blocks_until_reveal=self._config.blocks_until_reveal,
        )

    @_subtensor_retry
    def _query_epoch_info(self) -> tuple[int | None, int]:
        """Query tempo and blocks_since_last_step with retries."""
        tempo = self._subtensor.tempo(self._netuid)
        blocks_since_step = self._subtensor.blocks_since_last_step(self._netuid)
        return tempo, blocks_since_step

    @_subtensor_retry
    def _fetch_revealed_commitments(
        self,
    ) -> dict[str, tuple[tuple[int, str], ...]]:
        """Fetch all revealed commitments with retries."""
        return self._subtensor.get_all_revealed_commitments(self._netuid)

    def get_target_commit_block(
        self, eval_block_estimate: int, current_block: int
    ) -> int | None:
        """Find the epoch boundary block to commit at.

        Picks the latest epoch start block such that:
          commit_block + blocks_until_reveal + buffer < eval_block

        All validators see the same epoch boundaries (deterministic from
        chain state + netuid), so they converge on the same commit block.

        Args:
            eval_block_estimate: Estimated block number at evaluation time.
            current_block: Current chain block (passed in to avoid
                redundant RPC calls).

        Returns None if chain queries fail or no valid epoch found.
        """
        try:
            tempo, blocks_since_step = self._query_epoch_info()
        except Exception as e:
            logger.error(f"Chain query failed for epoch timing: {e}", exc_info=True)
            return None

        if tempo is None or tempo <= 0:
            logger.warning("Could not query tempo from chain")
            return None

        # Quarter-epoch safety margin between expected reveal time and
        # evaluation. Absorbs block-time variance (chain may produce blocks
        # faster than the nominal 12s), clock drift across validators, and
        # chain propagation delays for the reveal extrinsic.
        safety_margin_blocks = tempo // 4
        latest_commit = (
            eval_block_estimate
            - self._config.blocks_until_reveal
            - safety_margin_blocks
        )

        if latest_commit <= 0:
            return None

        # Epoch boundaries are offset per-netuid. Compute using the actual
        # last epoch start rather than assuming alignment to block 0.
        last_epoch_start = current_block - blocks_since_step
        epochs_ahead = (latest_commit - last_epoch_start) // tempo
        target = last_epoch_start + epochs_ahead * tempo

        if target <= current_block:
            return None

        return target

    def harvest(
        self,
        validator_hotkeys: set[str],
        min_reveal_block: int,
        committed_hotkeys: set[str],
    ) -> SeedResult | None:
        """
        Read all revealed commitments, filter to validators, combine into seed.

        Args:
            validator_hotkeys: Set of hotkeys with validator_permit.
            min_reveal_block: Discard reveals with block number below this
                value (stale reveals from previous cycles).
            committed_hotkeys: Only accept reveals from validators who had
                pending commitments before reveals landed — prevents
                late-commit gaming.

        Returns:
            SeedResult with combined seed, or None if no reveals available.
        """
        try:
            all_revealed = self._fetch_revealed_commitments()
        except Exception as e:
            logger.error(f"Failed to fetch revealed commitments: {e}", exc_info=True)
            return None

        # Filter to validator hotkeys and extract oldest valid reveal
        reveals: dict[str, str] = {}
        stale_count = 0
        late_count = 0
        for hotkey, entries in all_revealed.items():
            if hotkey not in validator_hotkeys:
                continue
            if not entries:
                continue
            hex_data = _oldest_valid_reveal(entries, min_reveal_block)
            if hex_data is None:
                stale_count += 1
                continue
            if hotkey not in committed_hotkeys:
                late_count += 1
                continue
            reveals[hotkey] = hex_data

        if not reveals:
            logger.warning(
                f"No validator reveals found "
                f"(total reveals: {len(all_revealed)}, "
                f"validator hotkeys: {len(validator_hotkeys)}, "
                f"stale: {stale_count}, late: {late_count})"
            )
            return None

        if len(reveals) < self._config.min_quorum:
            logger.warning(
                f"Quorum not met: {len(reveals)} reveals < "
                f"min_quorum {self._config.min_quorum}, "
                f"falling back to non-deterministic seed"
            )
            return None

        seed = combine_reveals(reveals, self._config.seed_modulus)

        result = SeedResult(
            seed=seed,
            num_reveals=len(reveals),
            validator_hotkeys=frozenset(reveals.keys()),
        )

        skipped = []
        if stale_count:
            skipped.append(f"{stale_count} stale")
        if late_count:
            skipped.append(f"{late_count} late")
        suffix = f" (skipped: {', '.join(skipped)})" if skipped else ""
        logger.info(
            f"Harvested seed={seed} from {result.num_reveals} validators" + suffix
        )
        return result

    def get_min_reveal_block(self, max_age_seconds: float) -> int | None:
        """Compute minimum acceptable reveal block for freshness filtering.

        The chain stores revealed commitments keyed by block number (not drand
        round), so the threshold must also be a block number.

        Args:
            max_age_seconds: Maximum age of a reveal in seconds.
                Reveals from blocks older than this are considered stale.

        Returns:
            Minimum block number, or None if chain query fails.
        """
        try:
            current_block = self._subtensor.block
        except Exception as e:
            logger.warning(f"Failed to query current block: {e}")
            return None

        blocks_back = int(max_age_seconds / self._config.block_time_seconds)
        return max(0, current_block - blocks_back)
