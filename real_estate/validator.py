"""
Real Estate Subnet Validator.

Uses Pylon for chain interactions (metagraph, weights, commitments).
Uses subtensor websocket for block number (TTL cached).
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import bittensor as bt
import numpy as np

from real_estate.chain import PylonClient, PylonConfig
from real_estate.chain.models import Metagraph
from real_estate.config import check_config, config_to_dict, get_config
from real_estate.utils.misc import ttl_get_block

logger = logging.getLogger(__name__)


class Validator:
    """
    Real Estate Subnet Validator.

    Responsibilities:
    - Sync metagraph periodically
    - Set weights periodically
    - Manage scores for miners

    The actual validation/evaluation logic will be added via
    the validation_loop() method.
    """

    def __init__(self, config: argparse.Namespace):
        """
        Initialize validator.

        Args:
            config: Validator configuration.
        """
        self.config = config

        # Validate config
        check_config(config)

        logger.info(f"Config: {config_to_dict(config)}")

        # Pylon client config
        self.pylon_config = PylonConfig(
            url=self.config.pylon_url,
            token=self.config.pylon_token,
        )

        # Subtensor for block fetching (persistent websocket connection)
        self.subtensor = bt.subtensor(network=self.config.subtensor_network)
        logger.info(f"Connected to subtensor: {self.subtensor.chain_endpoint}")

        # State
        self.metagraph: Metagraph | None = None
        self.scores: np.ndarray = np.array([], dtype=np.float32)
        self.hotkeys: list[str] = []
        self.uid: int | None = None
        self.step: int = 0

        # Block tracker for weight setting
        self._last_weight_set_block: int = 0

    @property
    def hotkey(self) -> str:
        """Our hotkey address."""
        value: str = self.config.hotkey
        return value

    @property
    def is_registered(self) -> bool:
        """Check if we're registered on the subnet."""
        if self.metagraph is None:
            return False
        return self.hotkey in self.metagraph.hotkeys

    @property
    def block(self) -> int:
        """Current block number with TTL caching (12 seconds)."""
        result: int = ttl_get_block(self)
        return result

    async def get_metagraph(self) -> Metagraph:
        """
        Get fresh metagraph from Pylon.

        This is the primary way to get metagraph data. Call this
        whenever you need current chain state (e.g., before evaluating miners).
        Handles hotkey/score updates automatically.

        Returns:
            Fresh metagraph from chain.
        """
        logger.debug("Fetching fresh metagraph...")

        async with PylonClient(self.pylon_config) as client:
            self.metagraph = await client.get_metagraph(self.config.netuid)

        logger.info(
            f"Metagraph fetched: {len(self.metagraph.neurons)} neurons "
            f"at block {self.metagraph.block}"
        )

        # Update local state (hotkeys, scores, uid)
        self._on_metagraph_updated()

        return self.metagraph

    def _on_metagraph_updated(self) -> None:
        """Handle metagraph changes - update hotkeys and scores."""
        if self.metagraph is None:
            return

        new_hotkeys = self.metagraph.hotkeys

        # First sync - initialize everything
        if not self.hotkeys:
            self.hotkeys = new_hotkeys.copy()
            self.scores = np.zeros(len(new_hotkeys), dtype=np.float32)
            self.uid = self.metagraph.get_uid(self.hotkey)
            logger.info(f"Initialized with {len(self.hotkeys)} hotkeys, our UID: {self.uid}")
            return

        # Check for changes
        if self.hotkeys == new_hotkeys:
            return

        logger.info("Hotkeys changed, updating scores...")

        # Zero out scores for replaced hotkeys
        for uid, old_hotkey in enumerate(self.hotkeys):
            if (
                uid < len(new_hotkeys)
                and old_hotkey != new_hotkeys[uid]
                and uid < len(self.scores)
            ):
                logger.debug(f"UID {uid} hotkey changed, zeroing score")
                self.scores[uid] = 0

        # Resize scores if metagraph grew
        if len(new_hotkeys) > len(self.scores):
            new_scores = np.zeros(len(new_hotkeys), dtype=np.float32)
            new_scores[: len(self.scores)] = self.scores
            self.scores = new_scores
            logger.info(f"Expanded scores array to {len(new_scores)}")

        # Update state
        self.hotkeys = new_hotkeys.copy()
        self.uid = self.metagraph.get_uid(self.hotkey)

    def check_registered(self) -> None:
        """Verify we're registered. Exits if not."""
        if not self.is_registered:
            logger.error(
                f"Hotkey {self.hotkey} is not registered on subnet {self.config.netuid}. "
                f"Please register with `btcli subnets register`"
            )
            raise SystemExit(1)

    async def set_weights(self) -> None:
        """
        Set weights on chain based on current scores.

        Logs success/failure internally. Exceptions bubble up for
        critical errors (auth, connection).
        """
        if self.metagraph is None:
            logger.error("Cannot set weights - metagraph not synced")
            return

        # Handle NaN values
        if np.isnan(self.scores).any():
            logger.warning("Scores contain NaN, replacing with 0")
            self.scores = np.nan_to_num(self.scores, nan=0)

        # Normalize scores to weights
        total = np.sum(self.scores)
        if total == 0:
            logger.warning("All scores are zero, skipping weight setting")
            return

        weights = self.scores / total

        # Build UID -> weight mapping for non-zero weights
        uids: list[int] = []
        weight_values: list[float] = []

        for uid, weight in enumerate(weights):
            if weight > 0:
                uids.append(uid)
                weight_values.append(float(weight))

        if not uids:
            logger.warning("No non-zero weights to set")
            return

        logger.info(f"Setting weights for {len(uids)} UIDs")

        try:
            async with PylonClient(self.pylon_config) as client:
                await client.set_weights(
                    netuid=self.config.netuid,
                    uids=uids,
                    weights=weight_values,
                )
            logger.info("set_weights on chain successfully!")
            self._last_weight_set_block = self.block
        except Exception as e:
            logger.error(f"set_weights failed: {e}")

    def update_scores(self, uids: list[int], rewards: np.ndarray) -> None:
        """
        Update scores using exponential moving average.

        Args:
            uids: List of miner UIDs
            rewards: Corresponding reward values
        """
        rewards = np.asarray(rewards, dtype=np.float32)

        if len(uids) != len(rewards):
            raise ValueError(f"UIDs ({len(uids)}) and rewards ({len(rewards)}) length mismatch")

        if len(uids) == 0:
            return

        # Handle NaN
        if np.isnan(rewards).any():
            logger.warning("Rewards contain NaN, replacing with 0")
            rewards = np.nan_to_num(rewards, nan=0)

        # Scatter rewards to full array
        scattered = np.zeros_like(self.scores)
        for uid, reward in zip(uids, rewards):
            if uid < len(scattered):
                scattered[uid] = reward

        # Exponential moving average
        alpha = self.config.moving_average_alpha
        self.scores = alpha * scattered + (1 - alpha) * self.scores

        logger.debug(f"Updated scores for {len(uids)} UIDs")

    def set_score(self, uid: int, score: float) -> None:
        """Set score for a single UID directly."""
        if uid < len(self.scores):
            self.scores[uid] = score

    def save_state(self) -> None:
        """Save validator state to disk."""
        state_file = self.config.state_path / "state.npz"

        np.savez(
            state_file,
            scores=self.scores,
            hotkeys=np.array(self.hotkeys, dtype=object),
        )
        logger.debug(f"State saved to {state_file}")

    def load_state(self) -> bool:
        """
        Load validator state from disk.

        Returns:
            True if state was loaded, False if no state file exists.
        """
        state_file = self.config.state_path / "state.npz"

        if not state_file.exists():
            logger.info("No saved state found")
            return False

        try:
            state = np.load(state_file, allow_pickle=True)
            self.scores = state["scores"]
            self.hotkeys = list(state["hotkeys"])
            logger.info(f"State loaded: {len(self.hotkeys)} hotkeys")
            return True
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False

    def should_set_weights(self) -> bool:
        """
        Check if enough blocks have elapsed to set weights.
        """
        if self.step == 0:
            return False

        disable_set_weights: bool = self.config.disable_set_weights
        if disable_set_weights:
            return False

        elapsed = self.block - self._last_weight_set_block
        epoch_length: int = self.config.epoch_length
        return elapsed > epoch_length

    async def run(self) -> None:
        """
        Main entry point - single loop that handles everything.

        Metagraph is fetched on-demand via get_metagraph() when needed.
        """
        logger.info(f"Starting validator for subnet {self.config.netuid}")

        # Initial metagraph fetch
        await self.get_metagraph()
        self.check_registered()
        self.load_state()

        # Initialize weight tracking block
        self._last_weight_set_block = self.block

        logger.info(f"Validator ready - UID {self.uid}, {len(self.hotkeys)} miners")

        # Main loop
        try:
            while True:
                logger.info(f"Step {self.step}")

                # Set weights if needed
                if self.should_set_weights():
                    await self.get_metagraph()
                    self.check_registered()
                    await self.set_weights()

                self.save_state()
                self.step += 1

                # Sleep between iterations
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("Validator stopped by keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def main() -> None:
    """CLI entry point."""
    config = get_config()
    setup_logging(config.log_level)

    validator = Validator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
