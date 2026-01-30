"""Pre-download scheduling for model downloads."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ..chain.models import ChainModelMetadata
from .downloader import ModelDownloader
from .errors import ModelError
from .models import DownloadResult

if TYPE_CHECKING:
    from ..chain.client import ChainClient

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for download scheduler."""

    pre_download_hours: float = 3.0  # Start downloading 3h before eval
    catch_up_minutes: float = 30.0  # Last 30min for catch-up
    min_delay_between_downloads_seconds: float = 5.0  # Minimum gap between downloads
    min_commitment_age_blocks: int = (
        7200  # ~24h at 12s/block, only eval older commitments
    )


class ModelDownloadScheduler:
    """
    Schedule model downloads before evaluation.

    Spreads downloads linearly across pre-validation window
    to avoid HuggingFace rate limits.

    Timeline (for 16:00 UTC eval):
    - 13:00: Pre-download phase starts
    - 13:00-15:30: Main downloads (spread linearly)
    - 15:30-16:00: Catch-up phase for new commitments
    - 16:00: Evaluation starts
    """

    def __init__(
        self,
        config: SchedulerConfig,
        downloader: ModelDownloader,
        chain_client: ChainClient,
    ):
        """
        Initialize scheduler.

        Args:
            config: Scheduler configuration
            downloader: Model downloader instance
            chain_client: Chain client for fetching commitments
        """
        self._config = config
        self._downloader = downloader
        self._chain = chain_client
        self._known_commitments: dict[str, ChainModelMetadata] = {}
        self._pre_download_ran = False

    @property
    def known_commitments(self) -> dict[str, ChainModelMetadata]:
        """Cached commitments from last download run."""
        return self._known_commitments

    def get_available_models(self, registered_hotkeys: set[str]) -> dict[str, Path]:
        """
        Get all models ready for evaluation.

        Returns cached models that:
        1. Are registered on chain (in registered_hotkeys)
        2. Have a known commitment
        3. Are in cache with matching hash

        Args:
            registered_hotkeys: Set of hotkeys currently registered on metagraph

        Returns:
            Dict mapping hotkey to model path
        """
        result: dict[str, Path] = {}
        for hotkey in registered_hotkeys:
            if hotkey not in self._known_commitments:
                continue
            commitment = self._known_commitments[hotkey]
            cached = self._downloader._cache.get(hotkey)
            if cached and cached.metadata.hash == commitment.model_hash:
                result[hotkey] = cached.path
        return result

    def _update_commitment_block(
        self,
        commitment: ChainModelMetadata,
        commit_block: int,
        target: dict[str, ChainModelMetadata] | None = None,
    ) -> None:
        """Update commitment metadata with correct block from Pylon."""
        target = target if target is not None else self._known_commitments
        target[commitment.hotkey] = ChainModelMetadata(
            hotkey=commitment.hotkey,
            hf_repo_id=commitment.hf_repo_id,
            model_hash=commitment.model_hash,
            block_number=commit_block,
        )

    async def run_pre_download(
        self,
        eval_time: datetime,
    ) -> dict[str, DownloadResult]:
        """
        Run pre-download phase.

        1. Fetch all commitments from chain
        2. Sort by priority (new/changed hash first)
        3. Spread downloads to finish before eval_time - catch_up_minutes
        4. Return results for each hotkey

        Timing logic:
        - Downloads must complete by: eval_time - catch_up_minutes
        - If enough time available: spread over pre_download_hours window
        - If less time: compress to fit available time
        - If past deadline: download immediately with minimal delays

        Args:
            eval_time: Scheduled evaluation time (UTC)

        Returns:
            Dict mapping hotkey to DownloadResult
        """
        # Reset flag at start of new cycle
        self._pre_download_ran = False

        now = datetime.now(UTC)
        deadline = eval_time - timedelta(minutes=self._config.catch_up_minutes)
        time_available = (deadline - now).total_seconds()

        if time_available > 0:
            logger.info(
                f"Pre-download phase: eval at {eval_time}, "
                f"deadline {deadline}, {time_available:.0f}s available"
            )
        else:
            logger.info(
                f"Pre-download phase: eval at {eval_time}, "
                f"deadline {deadline} (already passed by {-time_available:.0f}s)"
            )

        # Fetch all commitments and current block
        commitments = await self._chain.get_all_commitments()
        metagraph = await self._chain.get_metagraph()
        current_block = metagraph.block
        logger.info(f"Found {len(commitments)} commitments at block {current_block}")

        # Update known commitments
        self._known_commitments = {c.hotkey: c for c in commitments}
        self._pre_download_ran = True

        # Cleanup cache for hotkeys no longer on chain
        active_hotkeys = {c.hotkey for c in commitments}
        self._downloader.cleanup_stale_cache(active_hotkeys)

        # Filter to models that need downloading and are old enough
        cutoff_block = current_block - self._config.min_commitment_age_blocks
        to_download = self._filter_needs_download(commitments, cutoff_block)

        # Calculate download window based on available time
        max_window = self._config.pre_download_hours * 3600
        if time_available <= 0:
            # Past deadline - download ASAP
            window_seconds = 0.0
            logger.warning("Past deadline, downloading immediately")
        elif time_available < max_window:
            # Less time than ideal - compress window
            window_seconds = time_available
            logger.info(f"Compressed window: {window_seconds:.0f}s")
        else:
            # Plenty of time - use configured window
            window_seconds = max_window
            logger.info(f"Full window: {window_seconds:.0f}s")

        # Schedule downloads
        schedule = self._calculate_download_schedule(to_download, window_seconds)

        # Execute downloads
        results: dict[str, DownloadResult] = {}
        start_time = datetime.now(UTC)

        for delay_seconds, commitment in schedule:
            # Wait until scheduled time
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            wait_time = delay_seconds - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Download
            result, commit_block = await self._download_single(commitment)
            results[commitment.hotkey] = result

            # Update metadata with correct commit block from Pylon
            if result.success and commit_block is not None:
                self._update_commitment_block(commitment, commit_block)

        # Include already-cached models
        for commitment in commitments:
            if commitment.hotkey not in results:
                cached = self._downloader._cache.get(commitment.hotkey)
                if cached and cached.metadata.hash == commitment.model_hash:
                    result, commit_block = await self._download_single(commitment)
                    results[commitment.hotkey] = result
                    if result.success and commit_block is not None:
                        self._update_commitment_block(commitment, commit_block)

        # Filter out models that are too new based on real commit_block from Pylon
        # (We couldn't do this earlier because get_commitments doesn't return commit block)
        too_new_hotkeys = []
        for hotkey, result in results.items():
            if result.success:
                real_block = self._known_commitments[hotkey].block_number
                if real_block > cutoff_block:
                    too_new_hotkeys.append(hotkey)
                    logger.warning(
                        f"Excluding {hotkey}: committed at block {real_block}, "
                        f"cutoff is {cutoff_block}"
                    )

        for hotkey in too_new_hotkeys:
            results[hotkey] = DownloadResult(
                hotkey=hotkey,
                success=False,
                error=ModelError(
                    f"Commitment too recent (block {self._known_commitments[hotkey].block_number} > {cutoff_block})"
                ),
            )

        logger.info(
            f"Pre-download complete: {sum(1 for r in results.values() if r.success)} "
            f"success, {sum(1 for r in results.values() if not r.success)} failed"
            f"{f', {len(too_new_hotkeys)} excluded (too recent)' if too_new_hotkeys else ''}"
        )

        return results

    async def run_catch_up(
        self, failed_hotkeys: set[str] | None = None
    ) -> dict[str, DownloadResult]:
        """
        Run catch-up phase to retry failed downloads.

        Retries downloads that failed during pre-download phase.
        This handles cases where HuggingFace was temporarily unavailable.
        Called in the last N minutes before evaluation.

        Args:
            failed_hotkeys: Set of hotkeys that failed during pre-download.
                           If None, determines failed from known_commitments vs cache.

        Returns:
            Dict mapping hotkey to DownloadResult for retried downloads
        """
        logger.info("Starting catch-up phase")

        # Fetch current commitments and block
        current_commitments = await self._chain.get_all_commitments()
        metagraph = await self._chain.get_metagraph()
        current_block = metagraph.block
        cutoff_block = current_block - self._config.min_commitment_age_blocks

        current_map = {c.hotkey: c for c in current_commitments}

        # Find commitments to retry:
        # 1. Failed during pre-download (if provided)
        # 2. Known but not cached (fallback detection)
        # 3. All uncached if pre-download never ran (crashed before fetching commitments)
        pre_download_never_ran = not self._pre_download_ran and not failed_hotkeys
        if pre_download_never_ran:
            logger.info(
                "Pre-download never completed, will download all uncached models"
            )

        to_retry = []
        for hotkey, commitment in current_map.items():
            # Skip if already cached successfully
            if self._downloader.is_cached(hotkey, commitment.model_hash):
                continue

            # Retry if explicitly marked as failed
            if failed_hotkeys and hotkey in failed_hotkeys:
                to_retry.append(commitment)
                logger.info(f"Retrying failed download for {hotkey}")
            # Or if it's a known commitment that's not cached
            elif hotkey in self._known_commitments:
                to_retry.append(commitment)
                logger.info(f"Retrying uncached commitment for {hotkey}")
            # Or if pre-download never ran, download everything uncached
            elif pre_download_never_ran:
                to_retry.append(commitment)
                logger.info(f"Downloading uncached commitment for {hotkey}")

        if not to_retry:
            logger.info("No failed downloads to retry in catch-up phase")
            return {}

        logger.info(f"Retrying {len(to_retry)} failed downloads")

        # Download with minimal delays
        results: dict[str, DownloadResult] = {}
        for commitment in to_retry:
            result, commit_block = await self._download_single(commitment)
            results[commitment.hotkey] = result

            # Update metadata with correct commit block from Pylon
            if result.success and commit_block is not None:
                self._update_commitment_block(commitment, commit_block, current_map)

            await asyncio.sleep(self._config.min_delay_between_downloads_seconds)

        # Update known commitments (with corrected block numbers)
        self._known_commitments = current_map

        # Filter out models that are too new based on real commit_block from Pylon
        too_new_hotkeys = []
        for hotkey, result in results.items():
            if result.success:
                real_block = self._known_commitments[hotkey].block_number
                if real_block > cutoff_block:
                    too_new_hotkeys.append(hotkey)
                    logger.warning(
                        f"Excluding {hotkey}: committed at block {real_block}, "
                        f"cutoff is {cutoff_block}"
                    )

        for hotkey in too_new_hotkeys:
            results[hotkey] = DownloadResult(
                hotkey=hotkey,
                success=False,
                error=ModelError(
                    f"Commitment too recent (block {self._known_commitments[hotkey].block_number} > {cutoff_block})"
                ),
            )

        logger.info(
            f"Catch-up complete: {sum(1 for r in results.values() if r.success)} "
            f"success, {sum(1 for r in results.values() if not r.success)} failed"
            f"{f', {len(too_new_hotkeys)} excluded (too recent)' if too_new_hotkeys else ''}"
        )

        return results

    async def _download_single(
        self,
        commitment: ChainModelMetadata,
    ) -> tuple[DownloadResult, int | None]:
        """
        Download a single model and return result with commit block.

        Returns:
            Tuple of (DownloadResult, commit_block from Pylon or None if failed)
        """
        try:
            result = await self._downloader.download_model(commitment)
            return (
                DownloadResult(
                    hotkey=commitment.hotkey,
                    success=True,
                    path=result.path,
                ),
                result.commit_block,
            )
        except ModelError as e:
            logger.warning(f"Download failed for {commitment.hotkey}: {e}")
            return (
                DownloadResult(
                    hotkey=commitment.hotkey,
                    success=False,
                    error=e,
                ),
                None,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error downloading {commitment.hotkey}: {e}", exc_info=True
            )
            return (
                DownloadResult(
                    hotkey=commitment.hotkey,
                    success=False,
                    error=e,
                ),
                None,
            )

    def _filter_needs_download(
        self,
        commitments: list[ChainModelMetadata],
        cutoff_block: int,
    ) -> list[ChainModelMetadata]:
        """
        Filter to commitments that need downloading.

        Skips:
        - Models already cached with matching hash
        - Models committed after cutoff_block (too recent)
        """
        eligible = [c for c in commitments if c.block_number <= cutoff_block]
        too_recent = len(commitments) - len(eligible)

        needs_download = [
            c
            for c in eligible
            if not self._downloader.is_cached(c.hotkey, c.model_hash)
        ]

        logger.info(
            f"{len(needs_download)} need download, "
            f"{len(eligible) - len(needs_download)} already cached, "
            f"{too_recent} too recent (after block {cutoff_block})"
        )

        return needs_download

    def _calculate_download_schedule(
        self,
        commitments: list[ChainModelMetadata],
        window_seconds: float,
    ) -> list[tuple[float, ChainModelMetadata]]:
        """
        Calculate download times spread across window.

        Args:
            commitments: Prioritized list of commitments
            window_seconds: Total window duration in seconds

        Returns:
            List of (delay_seconds, commitment) tuples
        """
        if not commitments:
            return []

        # Calculate interval between downloads
        if len(commitments) == 1:
            interval = 0
        else:
            interval = window_seconds / (len(commitments) - 1)

        # Ensure minimum delay
        interval = max(interval, self._config.min_delay_between_downloads_seconds)

        schedule = []
        for i, commitment in enumerate(commitments):
            delay = i * interval
            schedule.append((delay, commitment))

        logger.info(
            f"Scheduled {len(commitments)} downloads over {window_seconds:.0f}s "
            f"(interval: {interval:.1f}s)"
        )

        return schedule

    def get_download_results(
        self,
        hotkeys: list[str],
    ) -> dict[str, DownloadResult]:
        """
        Get cached paths for hotkeys (for evaluation).

        Args:
            hotkeys: List of hotkeys to get results for

        Returns:
            Dict mapping hotkey to DownloadResult
        """
        results = {}
        for hotkey in hotkeys:
            path = self._downloader.get_cached_path(hotkey)
            if path:
                results[hotkey] = DownloadResult(
                    hotkey=hotkey,
                    success=True,
                    path=path,
                )
            else:
                results[hotkey] = DownloadResult(
                    hotkey=hotkey,
                    success=False,
                    error=ModelError(f"Model not cached for {hotkey}"),
                )
        return results
