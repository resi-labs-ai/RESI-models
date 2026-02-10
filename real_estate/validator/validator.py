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
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)

from real_estate.chain.errors import ChainConnectionError
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
from real_estate.observability import WandbLogger, create_wandb_logger
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
            path=self.config.wallet_path,
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
                test_data_path=self.config.test_data_path,
            ),
            self.wallet.hotkey,
        )

        # Model scheduler (initialized in run() when chain is available)
        self._model_scheduler = None

        # Validation orchestrator
        self._orchestrator = ValidationOrchestrator.create(
            score_threshold=self.config.score_threshold,
            docker_timeout=self.config.docker_timeout,
            docker_memory=self.config.docker_memory,
            docker_cpu=self.config.docker_cpu,
            docker_max_concurrent=self.config.docker_max_concurrent,
        )

        # WandB logger for evaluation metrics
        self._wandb_logger: WandbLogger = create_wandb_logger(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity or None,
            api_key=self.config.wandb_api_key or None,
            validator_hotkey=self.hotkey,
            netuid=self.config.netuid,
            enabled=not self.config.wandb_off,
            offline=self.config.wandb_offline,
            log_predictions_table=self.config.wandb_log_predictions,
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

        # Check validator permit before attempting to set weights
        if not self.metagraph.has_validator_permit(self.hotkey):
            logger.error(
                f"Cannot set weights - validator {self.hotkey} does not have "
                f"validator_permit. Ensure sufficient stake on subnet {self.config.netuid}"
            )
            return

        # Handle NaN values
        if np.isnan(self.scores).any():
            logger.warning("Scores contain NaN, replacing with 0")
            self.scores = np.nan_to_num(self.scores, nan=0)

        # Normalize scores to weights
        total = np.sum(self.scores)

        # Build hotkey -> weight mapping for non-zero weights
        weights: dict[str, float] = {}
        if total > 0:
            normalized_weights = self.scores / total
            for uid, weight in enumerate(normalized_weights):
                if weight > 0 and uid < len(self.hotkeys):
                    hotkey = self.hotkeys[uid]
                    weights[hotkey] = float(weight)

        # Apply burn if configured (works even with empty weights)
        weights = self._apply_burn(weights)

        if not weights:
            logger.warning("No weights to set (no scores and no burn configured)")
            return

        logger.info(f"Setting weights for {len(weights)} hotkeys")

        try:
            await self._ensure_chain().set_weights(weights)
            logger.info("Weights submitted to Pylon")
            self._last_weight_set_block = self.block
        except Exception as e:
            logger.error(f"Failed to submit weights to Pylon: {e}", exc_info=True)

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

    def _apply_burn(self, weights: dict[str, float]) -> dict[str, float]:
        """
        Apply burn allocation to weights.

        Burn mechanism allocates a fraction of emissions to the subnet owner UID,
        which the protocol then burns. Remaining emissions are distributed
        proportionally to other miners.

        Example with 50% burn:
          Before: {A: 0.6, B: 0.3, C: 0.1}
          After:  {A: 0.3, B: 0.15, C: 0.05, burn_hotkey: 0.5}

        Args:
            weights: Original weight distribution (must sum to 1.0)

        Returns:
            Adjusted weights with burn allocation (sums to 1.0)
        """
        burn_amount: float = 0.5  # Hardcoded to ensure autoupdater picks up change
        burn_uid: int = self.config.burn_uid

        # No burn configured
        if burn_amount <= 0.0 or burn_uid < 0:
            return weights

        # Get burn hotkey from UID
        if burn_uid >= len(self.hotkeys):
            logger.error(
                f"burn_uid {burn_uid} out of range (max {len(self.hotkeys) - 1}), skipping burn"
            )
            return weights

        burn_hotkey = self.hotkeys[burn_uid]

        # Scale down all existing weights
        remaining_share = 1.0 - burn_amount
        adjusted_weights = {
            hotkey: weight * remaining_share for hotkey, weight in weights.items()
        }

        # Add burn allocation (overwrite if burn_hotkey already has weight)
        existing_burn_weight = adjusted_weights.get(burn_hotkey, 0.0)
        adjusted_weights[burn_hotkey] = existing_burn_weight + burn_amount

        logger.info(
            f"Applied burn: {burn_amount:.1%} to UID {burn_uid} ({burn_hotkey[:8]}...), "
            f"remaining {remaining_share:.1%} distributed to {len(weights)} miners"
        )

        return adjusted_weights

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

    async def _run_catch_up_if_time(self, eval_time: datetime) -> None:
        """
        Run catch-up phase if there's time before evaluation.

        Catch-up retries downloads that failed during pre-download phase.
        This handles cases where HuggingFace was temporarily unavailable.
        It runs in the window between deadline and eval_time.

        Timing rules:
        - If now < deadline: wait until deadline, then run catch-up
        - If deadline <= now < eval_time: run catch-up immediately
        - If now >= eval_time: skip (no time)
        """
        now = datetime.now(UTC)
        catch_up_minutes = self.config.scheduler_catch_up_minutes
        deadline = eval_time - timedelta(minutes=catch_up_minutes)

        # No time for catch-up
        if now >= eval_time:
            logger.info("Catch-up skipped: evaluation time already reached")
            return

        # Wait until deadline if pre-download finished early
        if now < deadline:
            wait_seconds = (deadline - now).total_seconds()
            logger.info(
                f"Waiting {wait_seconds:.0f}s until catch-up phase (deadline: {deadline})"
            )
            await asyncio.sleep(wait_seconds)

        # Re-check after waiting
        now = datetime.now(UTC)
        if now >= eval_time:
            logger.info("Catch-up skipped: evaluation time reached during wait")
            return

        # Run catch-up with retry on connection errors
        time_remaining = (eval_time - now).total_seconds()
        logger.info(f"Starting catch-up phase ({time_remaining:.0f}s until evaluation)")

        # Extract failed hotkeys from pre-download results
        failed_hotkeys = {
            hotkey
            for hotkey, result in self.download_results.items()
            if not result.success
        }
        if failed_hotkeys:
            logger.info(f"Catch-up will retry {len(failed_hotkeys)} failed downloads")

        try:
            # Retry catch-up on connection errors until eval_time
            async for attempt in AsyncRetrying(
                wait=wait_fixed(30),  # 30s between retries
                stop=stop_after_delay(
                    max(0, time_remaining - 10)
                ),  # Stop 10s before eval
                retry=retry_if_exception_type(ChainConnectionError),
                reraise=True,
            ):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        logger.info(
                            f"Retrying catch-up (attempt "
                            f"{attempt.retry_state.attempt_number})"
                        )
                    catch_up_results = await self._model_scheduler.run_catch_up(
                        failed_hotkeys=failed_hotkeys if failed_hotkeys else None
                    )

                    if catch_up_results:
                        # Merge with existing results
                        if self.download_results:
                            self.download_results.update(catch_up_results)
                        else:
                            self.download_results = catch_up_results
        except ChainConnectionError as e:
            logger.warning(f"Catch-up failed after retries: {e}")
        except Exception as e:
            logger.warning(f"Catch-up phase failed: {e}", exc_info=True)

    def _on_validation_data_fetched(
        self,
        validation_data: ValidationDataset | None,
        raw_data: dict[str, dict] | None,  # noqa: ARG002
    ) -> None:
        """Callback when new validation data is fetched."""
        if validation_data is None:
            logger.warning("Validation data fetch returned None")
            # TODO: More sophisticated burn logic will be implemented.
            # For now, zero scores so the burn mechanism kicks in on next weight setting.
            self.scores.fill(0.0)
            return

        if len(validation_data) == 0:
            logger.warning("Validation data is empty, skipping evaluation")
            # TODO: More sophisticated burn logic will be implemented.
            # For now, zero scores so the burn mechanism kicks in on next weight setting.
            self.scores.fill(0.0)
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

        # Get all available models from cache (handles pre-download failures gracefully)
        model_paths = self._model_scheduler.get_available_models(
            registered_hotkeys, self.block
        )

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

        # Start WandB run to measure evaluation time
        self._wandb_logger.start_run()

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

            # Collect download failures for WandB logging
            download_failures: dict[str, str] = {}
            for hotkey, dl_result in self.download_results.items():
                if not dl_result.success and hotkey not in model_paths:
                    download_failures[hotkey] = (
                        dl_result.error_message or "Download failed"
                    )

            # Log evaluation results to WandB
            self._wandb_logger.log_evaluation(
                result, dataset, download_failures=download_failures
            )

        except NoValidModelsError as e:
            logger.warning(f"Evaluation skipped: {e}")
        finally:
            # Always finish WandB run
            self._wandb_logger.finish()

    async def _evaluation_loop(self) -> None:
        """Loop that waits for evaluation events and runs evaluation."""
        while True:
            await self._evaluation_event.wait()
            self._evaluation_event.clear()

            if self.validation_data is None:
                logger.warning("Evaluation triggered but validation_data is None")
                continue

            try:
                async for attempt in AsyncRetrying(
                    wait=wait_fixed(60),
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(ChainConnectionError),
                    reraise=True,
                ):
                    with attempt:
                        if attempt.retry_state.attempt_number > 1:
                            logger.info(
                                f"Retrying evaluation (attempt "
                                f"{attempt.retry_state.attempt_number}/3)"
                            )
                        await self.update_metagraph()
                        await self._run_evaluation(self.validation_data)
            except ChainConnectionError as e:
                logger.error(f"Evaluation failed after 3 attempts: {e}")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}", exc_info=True)

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

    async def _pre_download_loop(self) -> None:
        """
        Loop that runs pre-download before each scheduled evaluation.

        Timeline (for 22:30 UTC eval with 3h pre-download, 30min catch-up):
        - 19:30: Pre-download starts (downloads spread over 2.5h)
        - 22:00: Catch-up phase (retry failed downloads)
        - 22:30: Evaluation runs (models already downloaded)

        This loop handles ongoing pre-download scheduling after startup.
        Startup handles the first round, this loop handles subsequent rounds.
        """
        while True:
            # Calculate next evaluation time and when to start pre-download
            next_eval = self._get_next_eval_time()
            pre_download_start = next_eval - timedelta(
                hours=self.config.scheduler_pre_download_hours
            )

            # Wait until pre-download should start
            now = datetime.now(UTC)
            if now < pre_download_start:
                wait_seconds = (pre_download_start - now).total_seconds()
                if wait_seconds >= 3600:
                    wait_str = f"{wait_seconds / 3600:.1f}h"
                else:
                    wait_str = f"{wait_seconds / 60:.0f}m"
                logger.info(
                    f"Next pre-download at {pre_download_start} (waiting {wait_str})"
                )
                await asyncio.sleep(wait_seconds)

            # Run pre-download phase
            logger.info(f"Starting pre-download for evaluation at {next_eval}")
            try:
                self.download_results = await self._model_scheduler.run_pre_download(
                    eval_time=next_eval
                )
            except Exception as e:
                logger.warning(f"Pre-download failed: {e}")

            # Run catch-up phase
            await self._run_catch_up_if_time(next_eval)

            # Wait until after evaluation time before calculating next round
            now = datetime.now(UTC)
            if now < next_eval:
                wait_seconds = (next_eval - now).total_seconds() + 60  # +1min buffer
                await asyncio.sleep(wait_seconds)

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
                    min_commitment_age_blocks=self.config.model_min_commitment_age_blocks,
                    pre_download_hours=self.config.scheduler_pre_download_hours,
                    catch_up_minutes=self.config.scheduler_catch_up_minutes,
                ),
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
                    self._pre_download_loop(),
                )
            except asyncio.CancelledError:
                logger.info("Validator stopped")
            finally:
                validation_scheduler.shutdown()

    async def _startup(self) -> None:
        """Run startup tasks: metagraph, models, initial data fetch."""
        # Initial metagraph fetch - required for startup
        try:
            await self.update_metagraph()
        except Exception as e:
            logger.error(f"Failed to fetch initial metagraph: {e}", exc_info=True)
            raise SystemExit(1) from e

        if not self.is_registered():
            raise SystemExit(
                f"Hotkey {self.hotkey} is not registered on subnet {self.config.netuid}. "
                f"Please register with `btcli subnets register`"
            )

        # Check validator permit early
        if self.metagraph and not self.metagraph.has_validator_permit(self.hotkey):
            logger.warning(
                f"Validator {self.hotkey} does NOT have validator_permit. "
                f"Weight setting will fail until sufficient stake is added on subnet {self.config.netuid}."
            )

        self._last_weight_set_block = self.block

        logger.info(f"Validator ready - UID {self.uid}, {len(self.hotkeys)} miners")

        logger.info(
            f"Next evaluation at {self._get_next_eval_time()}, "
            f"pre-download loop will handle model downloads"
        )


async def main() -> None:
    """CLI entry point."""
    config = get_config()
    setup_logging(config.log_level)

    validator = Validator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
