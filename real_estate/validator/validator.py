"""
Real Estate Subnet Validator.

Uses Pylon for chain interactions (metagraph, weights, commitments).
Uses subtensor websocket for block number (TTL cached).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from real_estate.chain.subtensor import patch_subtensor_reconnect

if TYPE_CHECKING:
    from real_estate.chain import ChainClient
    from real_estate.orchestration.models import ValidationResult

from real_estate.data import (
    ATHRecord,
    ValidationClient,
    ValidationClientConfig,
    ValidationDataset,
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
from real_estate.randomness import DecentralizedSeedProvider, RandomnessConfig
from real_estate.utils.misc import ttl_get_block

from .config import check_config, config_to_dict, get_config, setup_logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Real Estate Subnet Validator.

    Runs concurrent loops:
    - _evaluation_loop: Waits for validation data, runs evaluation via orchestrator
    - _weight_setting_loop: Periodically sets weights on chain
    - _randomness_loop: Commits randomness and harvests shared seed before evaluation

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
        patch_subtensor_reconnect(self.subtensor)
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
        self.validation_client = ValidationClient(
            ValidationClientConfig(
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

        # Decentralized randomness seed provider
        self._randomness_config = RandomnessConfig(
            cycle_window_hours=self.config.randomness_cycle_window_hours,
            blocks_until_reveal=self.config.randomness_blocks_until_reveal,
            reveal_buffer_seconds=self.config.randomness_reveal_buffer_seconds,
        )
        self._seed_provider: DecentralizedSeedProvider | None = None
        if self.config.randomness_enabled:
            self._seed_provider = DecentralizedSeedProvider(
                subtensor=self.subtensor,
                wallet=self.wallet,
                netuid=self.config.netuid,
                config=self._randomness_config,
            )
        self._current_seed: int | None = None
        self._snapshot_path = (
            Path.home()
            / ".bittensor"
            / "randomness"
            / f"netuid_{self.config.netuid}"
            / "committed_hotkeys.json"
        )

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
            predictions_top_n_miners=self.config.wandb_predictions_top_n,
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
            if not self.config.test_mode:
                logger.error(
                    f"Cannot set weights - validator {self.hotkey} does not have "
                    f"validator_permit. Ensure sufficient stake on subnet {self.config.netuid}"
                )
                return
            logger.warning("No validator permit but test_mode=True, setting weights anyway")

        # In test mode, force all weight on UID 1 to bootstrap vpermit
        if self.config.test_mode and self.uid is not None and len(self.hotkeys) > 1:
            target_uid = 1
            target_hotkey = self.hotkeys[target_uid]
            weights: dict[str, float] = {target_hotkey: 1.0}
            logger.info(
                f"[TEST MODE] Forcing weight=1.0 on UID {target_uid} ({target_hotkey})"
            )
        else:
            # Handle NaN values
            if np.isnan(self.scores).any():
                logger.warning("Scores contain NaN, replacing with 0")
                self.scores = np.nan_to_num(self.scores, nan=0)

            # Normalize scores to weights
            total = np.sum(self.scores)

            # Build hotkey -> weight mapping for non-zero weights
            weights = {}
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
        for hotkey, weight in sorted(weights.items(), key=lambda x: -x[1]):
            logger.debug(f"  Weight {hotkey}: {weight:.6f}")

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
        burn_amount: float = (
            0.0  # Hardcoded: 0% burn. Autoupdater picks this up.
        )
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
        # Crash recovery: re-harvest if seed is missing but reveals may exist
        if self._current_seed is None:
            await self._try_reharvest_seed()

        # Update seed for this evaluation round
        self._orchestrator.set_seed(self._current_seed)
        logger.info(f"Evaluation seed: {self._current_seed}")

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

            if self.config.ath_enabled:
                pre_eval_ath = await self.validation_client.fetch_ath()
                await self._resolve_weights_with_ath(result, pre_eval_ath)
            else:
                self._apply_evaluation_weights(result)

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

    async def _randomness_loop(self) -> None:
        """
        Commit random value before each evaluation, harvest seed after reveal.

        Timeline (360 blocks reveal ≈ 72 min, epoch-aligned commit):
        1. Wait for target epoch boundary (or wall-clock fallback)
        2. Submit timelocked commitment
        3. ~36 min: Mid-epoch snapshot of who committed (anti-gaming)
        4. ~72 min + buffer: Chain auto-reveals, harvest & combine seed
        5. Evaluation uses the shared deterministic seed

        Falls back to None seed (random, non-deterministic) on any failure.
        """
        if self._seed_provider is None:
            logger.info("Randomness disabled, skipping randomness loop")
            return

        while True:
            try:
                next_eval = self._get_next_eval_time()

                await self._wait_for_commit_time(next_eval)

                # Submit commitment (sync RPC — run in thread to avoid
                # blocking the event loop during chain transaction).
                logger.info("Submitting randomness commitment...")
                reveal_round = await asyncio.to_thread(
                    self._seed_provider.commit
                )
                if reveal_round is None:
                    logger.warning(
                        "Randomness commitment failed, "
                        "falling back to non-deterministic seed"
                    )
                    self._current_seed = None
                    # Wait until after eval before next cycle
                    await self._sleep_until_after_eval(next_eval)
                    continue

                # Wait until mid-epoch, then snapshot pending commitments.
                # All honest validators should have committed by then, but
                # reveals haven't landed yet (those happen at epoch end).
                block_time = self._randomness_config.block_time_seconds
                reveal_total_seconds = (
                    self._randomness_config.blocks_until_reveal * block_time
                )
                snapshot_wait = reveal_total_seconds // 2
                remaining_wait = (
                    reveal_total_seconds
                    - snapshot_wait
                    + self._randomness_config.reveal_buffer_seconds
                )

                logger.info(
                    f"Waiting {snapshot_wait / 60:.0f}m until mid-epoch "
                    f"snapshot (reveal_round={reveal_round})"
                )
                await asyncio.sleep(snapshot_wait)

                # Snapshot which validators have pending commitments BEFORE
                # reveals land. Only these committed "on time" (before seeing
                # others' reveals). Late commits are rejected at harvest.
                validator_hotkeys = self._get_validator_hotkeys()
                try:
                    all_pending = await asyncio.to_thread(
                        self._seed_provider.get_pending_commitment_hotkeys
                    )
                    committed = all_pending & validator_hotkeys
                    self._save_committed_snapshot(committed)
                    logger.info(
                        f"Pre-reveal snapshot: {len(committed)} validator "
                        f"commitments on chain "
                        f"({len(all_pending)} total pending, "
                        f"{len(validator_hotkeys)} vpermit)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to snapshot pending commitments: {e}"
                    )
                    committed = None

                logger.info(
                    f"Waiting {remaining_wait / 60:.0f}m for reveals"
                )
                await asyncio.sleep(remaining_wait)

                # Harvest seed from all validator reveals — reject stale
                # reveals from previous cycles and late commits.
                max_age = self._randomness_config.cycle_window_hours * 3600
                min_block = await asyncio.to_thread(
                    self._seed_provider.get_min_reveal_block, max_age
                )

                if min_block is None or committed is None:
                    logger.warning(
                        "Cannot harvest: missing integrity checks "
                        f"(min_block={min_block is not None}, "
                        f"committed={committed is not None}), "
                        "falling back to non-deterministic seed"
                    )
                    self._current_seed = None
                else:
                    seed_result = await asyncio.to_thread(
                        self._seed_provider.harvest,
                        validator_hotkeys,
                        min_block,
                        committed,
                    )
                    if seed_result is not None:
                        self._current_seed = seed_result.seed
                        self._delete_committed_snapshot()
                    else:
                        self._current_seed = None
                        logger.warning(
                            "No reveals harvested, "
                            "falling back to non-deterministic seed"
                        )

                # Wait until after eval before next cycle
                await self._sleep_until_after_eval(next_eval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Randomness loop error: {e}", exc_info=True)
                self._current_seed = None
                await asyncio.sleep(300)  # Back off 5 min on unexpected error

    def _get_validator_hotkeys(self) -> set[str]:
        """Extract validator hotkeys from metagraph, or empty set if unavailable."""
        if self.metagraph is None:
            return set()
        return {n.hotkey for n in self.metagraph.neurons if n.validator_permit}

    def _save_committed_snapshot(self, hotkeys: set[str]) -> None:
        """Persist committed hotkeys snapshot to disk for crash recovery.

        Uses atomic write (tmp file + os.replace) to avoid corruption
        if the process crashes mid-write.
        """
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "hotkeys": sorted(hotkeys),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=self._snapshot_path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp_path, self._snapshot_path)
            except BaseException:
                os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning(f"Failed to save committed snapshot: {e}")

    def _load_committed_snapshot(self) -> set[str] | None:
        """Load committed hotkeys snapshot from disk, or None if stale/unavailable.

        Rejects snapshots older than cycle_window_hours to avoid
        using yesterday's snapshot after a failed cycle.
        """
        try:
            if not self._snapshot_path.exists():
                return None
            data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["timestamp"])
            max_age = timedelta(
                hours=self._randomness_config.cycle_window_hours
            )
            if datetime.now(UTC) - ts > max_age:
                logger.info("Committed snapshot too old, ignoring")
                return None
            return set(data["hotkeys"])
        except Exception as e:
            logger.warning(f"Failed to load committed snapshot: {e}")
            return None

    def _delete_committed_snapshot(self) -> None:
        """Remove snapshot file after successful harvest."""
        try:
            self._snapshot_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete committed snapshot: {e}")

    async def _try_reharvest_seed(self) -> None:
        """Attempt to harvest seed from existing on-chain reveals.

        Covers crash, restart, autoupdate — any scenario where the
        randomness loop missed the harvest window but reveals exist.
        Uses freshness check to avoid using stale reveals from previous cycles.
        Loads committed_hotkeys snapshot from disk to maintain late-commit
        filtering even after restart.
        """
        if self._seed_provider is None:
            return

        try:
            validator_hotkeys = self._get_validator_hotkeys()
            if not validator_hotkeys:
                return

            max_age = self._randomness_config.cycle_window_hours * 3600
            min_block = await asyncio.to_thread(
                self._seed_provider.get_min_reveal_block, max_age
            )
            committed = self._load_committed_snapshot()

            if min_block is None or committed is None:
                logger.info(
                    "Re-harvest skipped: missing integrity checks "
                    f"(min_block={min_block is not None}, "
                    f"committed={committed is not None})"
                )
                return

            seed_result = await asyncio.to_thread(
                self._seed_provider.harvest,
                validator_hotkeys,
                min_block,
                committed,
            )
            if seed_result is None:
                return

            self._current_seed = seed_result.seed
            self._delete_committed_snapshot()
            logger.info(
                f"Re-harvested seed at eval start: {seed_result.seed} "
                f"({seed_result.num_reveals} validators)"
            )
        except Exception as e:
            logger.warning(f"Re-harvest failed (will use seed=None): {e}")

    async def _wait_for_commit_time(self, next_eval: datetime) -> None:
        """Wait until commit time, preferring epoch-aligned block timing.

        Tries to align the commit to an epoch boundary so all validators
        converge on the same block. Falls back to wall-clock timing if
        chain queries fail.
        """
        wait_seconds = await self._get_epoch_wait_seconds(next_eval)
        if wait_seconds is not None:
            await asyncio.sleep(wait_seconds)
            return

        # Fallback: wall-clock timing
        commit_time = next_eval - timedelta(
            hours=self._randomness_config.cycle_window_hours
        )
        now = datetime.now(UTC)
        if now < commit_time:
            wait_seconds = (commit_time - now).total_seconds()
            logger.info(
                f"Randomness: wall-clock commit at {commit_time} "
                f"(waiting {wait_seconds / 60:.0f}m, epoch fallback)"
            )
            await asyncio.sleep(wait_seconds)
        else:
            logger.warning(
                f"Randomness: already past commit time {commit_time}, "
                f"committing immediately"
            )

    async def _get_epoch_wait_seconds(self, next_eval: datetime) -> float | None:
        """Compute seconds to wait for epoch-aligned commit, or None on failure."""
        try:
            current_block = ttl_get_block(self)
            seconds_until_eval = (next_eval - datetime.now(UTC)).total_seconds()
            block_time = self._randomness_config.block_time_seconds
            eval_block_est = current_block + int(seconds_until_eval / block_time)
            target_block = await asyncio.to_thread(
                self._seed_provider.get_target_commit_block,
                eval_block_est,
                current_block,
            )
        except Exception as e:
            logger.warning(f"Epoch timing query failed: {e}")
            return None

        if target_block is None:
            return None

        blocks_to_wait = target_block - current_block
        if blocks_to_wait <= 0:
            return None

        wait_seconds = blocks_to_wait * self._randomness_config.block_time_seconds
        logger.info(
            f"Randomness: epoch-aligned commit at block {target_block} "
            f"(waiting {wait_seconds / 60:.0f}m, {blocks_to_wait} blocks)"
        )
        return wait_seconds

    async def _sleep_until_after_eval(self, eval_time: datetime) -> None:
        """Sleep until after the given evaluation time + 1 min buffer."""
        now = datetime.now(UTC)
        if now < eval_time:
            await asyncio.sleep((eval_time - now).total_seconds() + 60)

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
            try:
                if self.should_set_weights():
                    await self.update_metagraph()
                    if not self.is_registered():
                        logger.error(
                            f"Hotkey {self.hotkey} is not registered on subnet "
                            f"{self.config.netuid}"
                        )
                    else:
                        await self.set_weights()
            except Exception as e:
                logger.warning(f"Weight setting failed: {e}", exc_info=True)

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

        # Use static scheduler when test-models-dir is set
        if self.config.test_models_dir:
            from .testing import build_static_scheduler

            self._model_scheduler = build_static_scheduler(
                Path(self.config.test_models_dir)
            )

        from real_estate.chain import ChainClient

        async with ChainClient(self._pylon_config) as chain:
            self.chain = chain

            # Initialize model scheduler (requires chain client) unless static
            if not self.config.test_models_dir:
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
                    hf_token=self.config.hf_token or None,
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
                    self._randomness_loop(),
                )
            except asyncio.CancelledError:
                logger.info("Validator stopped")
            finally:
                validation_scheduler.shutdown()

    async def _set_burn_weights(self) -> None:
        """
        Set 100% weight to the burn UID.

        Used as a fallback when no other weight data is available,
        so the validator keeps setting weights on chain rather than going dark.
        """
        burn_uid: int = self.config.burn_uid
        if burn_uid < 0 or burn_uid >= len(self.hotkeys):
            logger.warning(
                "No valid burn_uid configured, "
                "cannot set weights until first evaluation"
            )
            return

        burn_hotkey = self.hotkeys[burn_uid]
        logger.info(
            f"Setting 100% weight to burn UID {burn_uid} ({burn_hotkey[:8]}...)"
        )
        try:
            await self._ensure_chain().set_weights({burn_hotkey: 1.0})
            self._last_weight_set_block = self.block
        except Exception as e:
            logger.warning(f"Burn weight setting failed: {e}")

    async def _resolve_weights_with_ath(
        self,
        result: ValidationResult,
        pre_eval_ath: ATHRecord | None,
    ) -> None:
        """
        Decide weights based on ATH comparison after evaluation.

        Fetches fresh ATH (with pre-eval fallback), compares eval winner
        against ATH, and applies either ATH-based or standard weights.
        """
        post_eval_ath = await self.validation_client.fetch_ath()
        if post_eval_ath:
            logger.info(
                f"Post-eval ATH fetched: {post_eval_ath.hotkey} "
                f"(score={post_eval_ath.score:.4f})"
            )
        elif pre_eval_ath:
            logger.warning("Post-eval ATH fetch failed, using pre-eval ATH as fallback")
        else:
            logger.warning(
                "Both ATH fetches failed, falling back to standard evaluation weights"
            )
        ath = post_eval_ath or pre_eval_ath

        if (
            ath
            and ath.hotkey in self.hotkeys
            and result.winner.winner_score <= ath.score
        ):
            logger.info(
                f"ATH not beaten: eval winner {result.winner.winner_score:.4f} "
                f"<= ATH {ath.score:.4f} ({ath.hotkey})"
            )
            self._apply_ath_weights(ath, result)
        else:
            if ath and ath.hotkey not in self.hotkeys:
                logger.warning(
                    f"ATH hotkey {ath.hotkey} not found in metagraph, "
                    f"rewarding local evaluation winner instead"
                )
            elif ath and result.winner.winner_score > ath.score:
                logger.info(
                    f"New ATH! {result.winner.winner_hotkey} "
                    f"score={result.winner.winner_score:.4f} > ATH {ath.score:.4f}"
                )
            elif not ath:
                logger.info(
                    "No ATH available, rewarding local evaluation winner instead"
                )
            self._apply_evaluation_weights(result)

    def _apply_evaluation_weights(self, result: ValidationResult) -> None:
        """Apply standard evaluation weights (existing behavior)."""
        self.scores.fill(0.0)
        for hotkey, weight in result.weights.weights.items():
            if hotkey in self.hotkeys:
                uid = self.hotkeys.index(hotkey)
                self.scores[uid] = weight

    def _apply_ath_weights(self, ath: ATHRecord, result: ValidationResult) -> None:
        """
        Apply ATH-based weights: 99% to ATH winner, 1% distributed to rest.

        Uses evaluation results for the 1% proportional distribution among
        non-ATH, non-copier successful miners.
        """
        self.scores.fill(0.0)

        copiers = result.duplicate_result.copier_hotkeys

        # Collect non-ATH, non-copier successful results for 1% distribution
        others = [
            r
            for r in result.eval_batch.successful_results
            if r.hotkey != ath.hotkey and r.hotkey not in copiers
        ]

        # ATH winner gets 99%
        ath_uid = self.hotkeys.index(ath.hotkey)
        self.scores[ath_uid] = 0.99

        # Distribute 1% proportionally by score
        if others:
            total_score = sum(r.score for r in others)
            if total_score > 0:
                for r in others:
                    if r.hotkey in self.hotkeys:
                        uid = self.hotkeys.index(r.hotkey)
                        self.scores[uid] = (r.score / total_score) * 0.01
            else:
                equal_share = 0.01 / len(others)
                for r in others:
                    if r.hotkey in self.hotkeys:
                        uid = self.hotkeys.index(r.hotkey)
                        self.scores[uid] = equal_share

        logger.info(f"ATH weights: {ath.hotkey} = 99%, {len(others)} miners share 1%")

    async def _bootstrap_weights(self) -> None:
        """
        Bootstrap weights on startup.

        Priority (when ATH enabled):
        1. ATH winner from dashboard -> 100% to ATH
        2. Chain consensus incentive values
        3. 100% to burn UID (fallback)

        Priority (when ATH disabled):
        1. Chain consensus incentive values
        2. 100% to burn UID (fallback)
        """
        if self.metagraph is None or not self.hotkeys:
            logger.warning("Cannot bootstrap weights: metagraph not available")
            return

        # ATH bootstrap (only when enabled)
        if self.config.ath_enabled:
            ath = await self.validation_client.fetch_ath()
            if not ath:
                logger.warning(
                    "ATH fetch failed or no ATH available, "
                    "falling back to consensus bootstrap"
                )
            if ath and ath.hotkey in self.hotkeys:
                uid = self.hotkeys.index(ath.hotkey)
                self.scores.fill(0.0)
                self.scores[uid] = 1.0
                logger.info(
                    f"Bootstrapping from ATH winner: {ath.hotkey} "
                    f"(score={ath.score:.4f}, achieved={ath.achieved_at})"
                )
                try:
                    await self.set_weights()
                except Exception as e:
                    logger.warning(f"ATH bootstrap weight setting failed: {e}")
                return

            if ath:
                logger.warning(
                    f"ATH hotkey {ath.hotkey} not found in current metagraph, "
                    f"falling back to consensus bootstrap"
                )

        # Consensus bootstrap
        bootstrap_count = 0
        for neuron in self.metagraph.neurons:
            if neuron.uid < len(self.scores) and neuron.incentive > 0:
                self.scores[neuron.uid] = float(neuron.incentive)
                bootstrap_count += 1

        if bootstrap_count > 0:
            logger.info(
                f"Bootstrapping from chain consensus: "
                f"{bootstrap_count} neurons with incentive > 0"
            )
            try:
                await self.set_weights()
            except Exception as e:
                logger.warning(f"Bootstrap weight setting failed: {e}")
            return

        # Burn fallback
        logger.info("No incentive data, falling back to burn")
        await self._set_burn_weights()

    async def _startup(self) -> None:
        """Run startup tasks: metagraph, models, initial data fetch."""
        # In local test mode with static models, skip chain-dependent startup
        if self.config.test_models_dir and self.config.test_mode:
            logger.info(
                "Static model mode: skipping registration check and bootstrap weights"
            )
            return

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

        # Bootstrap weights from consensus to avoid burn period after restart
        if not self.config.disable_set_weights:
            await self._bootstrap_weights()

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
