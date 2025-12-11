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
from real_estate.config import check_config, config_to_dict, get_config, setup_logging
from real_estate.data import ScraperClient, ScraperConfig, ValidationDataset
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

        # Pylon client for chain interactions
        self.pylon = PylonClient(
            PylonConfig(
                url=self.config.pylon_url,
                token=self.config.pylon_token,
                identity=self.config.pylon_identity,
                netuid=self.config.netuid,
            )
        )

        # Subtensor for block fetching (persistent websocket connection)
        self.subtensor = bt.subtensor(network=self.config.subtensor_network)
        logger.info(f"Connected to subtensor: {self.subtensor.chain_endpoint}")

        # Wallet for signing
        self.wallet = bt.wallet(
            name=self.config.wallet_name,
            hotkey=self.config.wallet_hotkey,
        )
        self.hotkey: str = self.wallet.hotkey.ss58_address
        logger.info(f"Loaded wallet: {self.wallet.name}/{self.wallet.hotkey_str}")

        # Scraper client for fetching validation data
        self.scraper = ScraperClient(
            ScraperConfig(
                url=self.config.scraper_url,
                realm=self.config.realm,
                schedule_hour=self.config.scraper_schedule_hour,
                schedule_minute=self.config.scraper_schedule_minute,
                max_retries=self.config.scraper_max_retries,
                retry_delay_seconds=self.config.scraper_retry_delay,
            ),
            self.wallet.hotkey,
        )

        # State
        self.metagraph: Metagraph | None = None
        self.validation_data: ValidationDataset | None = None
        self.scores: np.ndarray = np.array([], dtype=np.float32)
        self.hotkeys: list[str] = []
        self.uid: int | None = None

        # Block tracker for weight setting
        self._last_weight_set_block: int = 0

    @property
    def block(self) -> int:
        """Current block number with TTL caching (12 seconds)."""
        result: int = ttl_get_block(self)
        return result

    async def update_metagraph(self) -> None:
        """
        Fetch fresh metagraph from Pylon and update local state.

        Updates self.metagraph, hotkeys, scores, and uid.

        Raises:
            Exception: If metagraph fetch fails.
        """
        logger.debug("Fetching fresh metagraph...")

        self.metagraph = await self.pylon.get_metagraph()

        logger.info(
            f"Metagraph updated: {len(self.metagraph.neurons)} neurons "
            f"at block {self.metagraph.block}"
        )

        # Update local state (hotkeys, scores, uid)
        self._on_metagraph_updated()

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
            logger.info(
                f"Initialized with {len(self.hotkeys)} hotkeys, this validator's UID: {self.uid}"
            )
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

    def is_registered(self) -> bool:
        """Check if our hotkey is registered on the subnet."""
        if self.metagraph is None:
            logger.error("Cannot check registration - no metagraph")
            return False
        return self.hotkey in self.hotkeys

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

        normalized_weights = self.scores / total

        # Build hotkey -> weight mapping for non-zero weights
        weights: dict[str, float] = {}
        for uid, weight in enumerate(normalized_weights):
            if weight > 0 and uid < len(self.hotkeys):
                hotkey = self.hotkeys[uid]
                weights[hotkey] = float(weight)

        if not weights:
            logger.warning("No non-zero weights to set")
            return

        logger.info(f"Setting weights for {len(weights)} hotkeys")

        try:
            await self.pylon.set_weights(weights)
            logger.info("set_weights on chain successfully!")
            self._last_weight_set_block = self.block
        except Exception as e:
            logger.error(f"set_weights failed: {e}")

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
        disable_set_weights: bool = self.config.disable_set_weights
        if disable_set_weights:
            return False

        elapsed = self.block - self._last_weight_set_block
        epoch_length: int = self.config.epoch_length
        return elapsed > epoch_length

    def _on_validation_data_fetched(self, data: ValidationDataset) -> None:
        """Callback when new validation data is fetched."""
        self.validation_data = data
        logger.info(f"Validation data updated: {len(data)} properties")

    async def run(self) -> None:
        """
        Main entry point - single loop that handles everything.

        Metagraph is updated on-demand via update_metagraph() when needed.
        """
        logger.info(f"Starting validator for subnet {self.config.netuid}")

        # Initial metagraph fetch - required for startup
        try:
            await self.update_metagraph()
        except Exception as e:
            logger.error(f"Failed to fetch initial metagraph: {e}")
            raise SystemExit(1) from e

        if not self.is_registered():
            raise SystemExit(
                f"Hotkey {self.hotkey} is not registered on subnet {self.config.netuid}. "
                f"Please register with `btcli subnets register`"
            )

        self.load_state()

        # Initialize weight tracking block
        self._last_weight_set_block = self.block

        logger.info(f"Validator ready - UID {self.uid}, {len(self.hotkeys)} miners")

        # Initial validation data fetch
        logger.info("Performing initial validation data fetch...")
        try:
            data = await self.scraper.fetch_with_retry()
            self._on_validation_data_fetched(data)
        except Exception as e:
            logger.warning(f"Initial validation data fetch failed: {e}")

        # Start scheduled daily data fetcher (cron job)
        scheduler = self.scraper.start_scheduled(
            on_fetch=self._on_validation_data_fetched,
        )

        # Main loop
        try:
            while True:
                # TODO: REMOVE - temporary hardcoded scores for testing
                self.scores = np.array([0.3, 0.7], dtype=np.float32)

                # Set weights if needed
                if self.should_set_weights():
                    try:
                        await self.update_metagraph()
                        if not self.is_registered():
                            raise ValueError(
                                f"Hotkey {self.hotkey} is not registered on subnet {self.config.netuid}."
                                f"Please register with `btcli subnets register`"
                            )
                        await self.set_weights()
                    except Exception as e:
                        logger.warning(f"Weight setting failed: {e}")

                self.save_state()
                await asyncio.sleep(5)  # TODO: change back to 60 after testing

        except KeyboardInterrupt:
            logger.info("Validator stopped by keyboard interrupt")
        finally:
            scheduler.shutdown()


async def main() -> None:
    """CLI entry point."""
    config = get_config()
    setup_logging(config.log_level)

    validator = Validator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
