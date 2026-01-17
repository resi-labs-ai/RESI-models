"""
Real Estate Subnet Validator.

Uses Pylon for chain interactions (metagraph, weights, commitments).
Uses subtensor websocket for block number (TTL cached).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import bittensor as bt
import numpy as np

from real_estate.chain.models import Metagraph

if TYPE_CHECKING:
    from real_estate.chain import ChainClient

from real_estate.data import (
    ValidationDataset,
    ValidationDatasetClient,
    ValidationDatasetClientConfig,
)
from real_estate.incentives import NoValidModelsError
from real_estate.models import (
    DownloadConfig,
    DownloadResult,
    SchedulerConfig,
    create_model_scheduler,
)
from real_estate.orchestration import ValidationOrchestrator
from real_estate.utils.misc import ttl_get_block

from .config import check_config, config_to_dict, get_config, setup_logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Real Estate Subnet Validator.

    Runs two concurrent loops:
    - _evaluation_loop: Waits for validation data, runs evaluation via orchestrator
    - _weight_setting_loop: Periodically sets weights on chain

    Evaluation is triggered when new validation data arrives (daily).
    The ValidationOrchestrator handles the actual evaluation logic.
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

        # Pylon client config (client created as context manager in run())
        from real_estate.chain import PylonConfig

        self._pylon_config = PylonConfig(
            url=self.config.pylon_url,
            token=self.config.pylon_token,
            identity=self.config.pylon_identity,
        )
        self.chain: ChainClient | None = None

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

        # Validation data client for fetching validation data from dashboard API
        self.validation_client = ValidationDatasetClient(
            ValidationDatasetClientConfig(
                url=self.config.validation_data_url,
                max_retries=self.config.validation_data_max_retries,
                retry_delay_seconds=self.config.validation_data_retry_delay,
                schedule_hour=self.config.validation_data_schedule_hour,
                schedule_minute=self.config.validation_data_schedule_minute,
                download_raw=self.config.validation_data_download_raw,
            ),
            self.wallet.hotkey,
        )

        # Model scheduler (initialized in run() when chain is available)
        self._model_scheduler = None

        # Validation orchestrator
        self._orchestrator = ValidationOrchestrator.create(
            score_threshold=self.config.score_threshold,
        )

        # State
        self.metagraph: Metagraph | None = None
        self.validation_data: ValidationDataset | None = None
        self.scores: np.ndarray = np.array([], dtype=np.float32)
        self.hotkeys: list[str] = []
        self.uid: int | None = None
        self.download_results: dict[str, DownloadResult] = {}  # hotkey -> result

        # Block tracker for weight setting
        self._last_weight_set_block: int = 0

        # Event to signal new validation data needs evaluation
        self._evaluation_event: asyncio.Event = asyncio.Event()

        # Lock to prevent concurrent metagraph updates
        self._metagraph_lock: asyncio.Lock = asyncio.Lock()

    @property
    def block(self) -> int:
        """Current block number with TTL caching (12 seconds)."""
        result: int = ttl_get_block(self)
        return result

    def _ensure_chain(self) -> ChainClient:
        """Ensure chain client is initialized."""
        if self.chain is None:
            raise RuntimeError("ChainClient not initialized - call run() first")
        return self.chain

    async def update_metagraph(self) -> None:
        """
        Fetch fresh metagraph from Pylon and update local state.

        Updates self.metagraph, hotkeys, scores, and uid.
        Uses lock to prevent concurrent updates from multiple loops.

        Raises:
            Exception: If metagraph fetch fails.
        """
        async with self._metagraph_lock:
            logger.debug("Fetching fresh metagraph...")

            self.metagraph = await self._ensure_chain().get_metagraph()

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
            await self._ensure_chain().set_weights(weights)
            logger.info("set_weights on chain successfully!")
            self._last_weight_set_block = self.block
        except Exception as e:
            logger.error(f"set_weights failed: {e}")

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

    def _get_next_eval_time(self) -> datetime:
        """Calculate next scheduled evaluation time based on config."""
        now = datetime.now(UTC)
        next_eval = now.replace(
            hour=self.config.validation_data_schedule_hour,
            minute=self.config.validation_data_schedule_minute,
            second=0,
            microsecond=0,
        )
        if next_eval <= now:
            next_eval += timedelta(days=1)
        return next_eval

    def _on_validation_data_fetched(
        self,
        validation_data: ValidationDataset | None,
        raw_data: dict[str, dict] | None,  # noqa: ARG002
    ) -> None:
        """Callback when new validation data is fetched."""
        if validation_data is None:
            logger.warning("Validation data fetch returned None")
            return

        if len(validation_data) == 0:
            logger.warning("Validation data is empty, skipping evaluation")
            return

        self.validation_data = validation_data
        logger.info(f"Validation data updated: {len(validation_data)} properties")
        self._evaluation_event.set()

    async def _run_evaluation(self, dataset: ValidationDataset) -> None:
        """
        Run evaluation pipeline on the given dataset.

        Updates self.scores based on orchestrator results.
        """
        # Get current metagraph hotkeys
        registered_hotkeys = set(self.hotkeys)

        # Get successful model paths, filtered to registered hotkeys only
        model_paths = {
            hotkey: result.model_path
            for hotkey, result in self.download_results.items()
            if result.success
            and result.model_path is not None
            and hotkey in registered_hotkeys
        }

        if not model_paths:
            logger.warning("No models available for evaluation")
            return

        # Get cached metadata from scheduler, filtered to models we're evaluating
        chain_metadata = {
            hotkey: meta
            for hotkey, meta in self._model_scheduler.known_commitments.items()
            if hotkey in model_paths
        }

        logger.info(f"Running evaluation with {len(model_paths)} models")

        try:
            result = await self._orchestrator.run(dataset, model_paths, chain_metadata)

            # Reset all scores - miners not evaluated get 0
            self.scores.fill(0.0)

            # Update scores from weights
            for hotkey, weight in result.weights.weights.items():
                if hotkey in self.hotkeys:
                    uid = self.hotkeys.index(hotkey)
                    self.scores[uid] = weight

            logger.info(
                f"Evaluation complete: winner={result.winner.winner_hotkey}, "
                f"score={result.winner.winner_score:.4f}"
            )

        except NoValidModelsError as e:
            logger.warning(f"Evaluation skipped: {e}")

    async def _evaluation_loop(self) -> None:
        """Loop that waits for evaluation events and runs evaluation."""
        while True:
            await self._evaluation_event.wait()
            self._evaluation_event.clear()

            if self.validation_data is None:
                logger.warning("Evaluation triggered but validation_data is None")
                continue

            try:
                await self.update_metagraph()
                await self._run_evaluation(self.validation_data)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")

    async def _weight_setting_loop(self) -> None:
        """Loop that periodically checks and sets weights."""
        while True:
            if self.should_set_weights():
                try:
                    await self.update_metagraph()
                    if not self.is_registered():
                        logger.error(
                            f"Hotkey {self.hotkey} is not registered on subnet "
                            f"{self.config.netuid}"
                        )
                    else:
                        await self.set_weights()
                except Exception as e:
                    logger.warning(f"Weight setting failed: {e}")

            await asyncio.sleep(60)

    async def run(self) -> None:
        """
        Main entry point.

        Runs concurrent loops for evaluation and weight setting.
        """
        logger.info(f"Starting validator for subnet {self.config.netuid}")

        from real_estate.chain import ChainClient

        async with ChainClient(self._pylon_config) as chain:
            self.chain = chain

            # Initialize model scheduler (requires chain client)
            self._model_scheduler = create_model_scheduler(
                chain_client=chain,
                cache_dir=self.config.model_cache_path,
                download_config=DownloadConfig(
                    max_model_size_bytes=self.config.model_max_size_mb * 1024 * 1024,
                ),
                scheduler_config=SchedulerConfig(
                    min_commitment_age_blocks=0,  # TODO For testing - accept all commitments
                ),
                required_license=self.config.model_required_license,
            )

            await self._startup()

            # Start scheduled daily data fetcher (cron job)
            validation_scheduler = self.validation_client.start_scheduled(
                on_fetch=self._on_validation_data_fetched,
            )

            try:
                await asyncio.gather(
                    self._evaluation_loop(),
                    self._weight_setting_loop(),
                )
            except KeyboardInterrupt:
                logger.info("Validator stopped by keyboard interrupt")
            finally:
                validation_scheduler.shutdown()

    async def _startup(self) -> None:
        """Run startup tasks: metagraph, models, initial data fetch."""
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

        self._last_weight_set_block = self.block

        logger.info(f"Validator ready - UID {self.uid}, {len(self.hotkeys)} miners")

        # Initial model download - spread until next scheduled eval time
        logger.info("Downloading miner models...")
        try:
            next_eval = self._get_next_eval_time()
            logger.info(f"Next evaluation scheduled at {next_eval}")
            self.download_results = await self._model_scheduler.run_pre_download(
                eval_time=next_eval
            )
        except Exception as e:
            logger.warning(f"Initial model download failed: {e}")

        # Initial validation data fetch
        logger.info("Performing initial validation data fetch...")
        try:
            validation_data, raw_data = await self.validation_client.fetch_with_retry(
                download_validation=True,
                download_raw=self.config.validation_data_download_raw,
            )
            self._on_validation_data_fetched(validation_data, raw_data)
        except Exception as e:
            logger.warning(f"Initial validation data fetch failed: {e}")


async def main() -> None:
    """CLI entry point."""
    config = get_config()
    setup_logging(config.log_level)

    validator = Validator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
