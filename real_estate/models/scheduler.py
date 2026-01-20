"""Pre-download scheduling for model downloads."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from .downloader import ModelDownloader
from .errors import ModelError
from .models import DownloadResult

if TYPE_CHECKING:
    from ..chain.client import ChainClient
    from ..chain.models import ChainModelMetadata

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

    @property
    def known_commitments(self) -> dict[str, ChainModelMetadata]:
        """Cached commitments from last download run."""
        return self._known_commitments

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
        now = datetime.now(UTC)
        deadline = eval_time - timedelta(minutes=self._config.catch_up_minutes)
        time_available = (deadline - now).total_seconds()

        logger.info(
            f"Pre-download phase: eval at {eval_time}, "
            f"deadline {deadline}, {time_available:.0f}s available"
        )

        # Fetch all commitments and current block
        commitments = await self._chain.get_all_commitments()
        metagraph = await self._chain.get_metagraph()
        current_block = metagraph.block
        logger.info(f"Found {len(commitments)} commitments at block {current_block}")

        # Update known commitments
        self._known_commitments = {c.hotkey: c for c in commitments}

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
            result = await self._download_single(commitment)
            results[commitment.hotkey] = result

        logger.info(
            f"Pre-download complete: {sum(1 for r in results.values() if r.success)} "
            f"success, {sum(1 for r in results.values() if not r.success)} failed"
        )

        return results

    async def run_catch_up(self) -> dict[str, DownloadResult]:
        """
        Run catch-up phase for new commitments.

        Downloads any commitments that appeared since pre-download started,
        but only if they meet the age requirement.
        Called in the last 30 minutes before evaluation.

        Returns:
            Dict mapping hotkey to DownloadResult for new downloads
        """
        logger.info("Starting catch-up phase")

        # Fetch current commitments and block
        current_commitments = await self._chain.get_all_commitments()
        metagraph = await self._chain.get_metagraph()
        current_block = metagraph.block
        cutoff_block = current_block - self._config.min_commitment_age_blocks

        current_map = {c.hotkey: c for c in current_commitments}

        # Find new or changed commitments that are old enough
        new_commitments = []
        for hotkey, commitment in current_map.items():
            if commitment.block_number > cutoff_block:
                continue  # Too recent
            known = self._known_commitments.get(hotkey)
            if known is None or known.model_hash != commitment.model_hash:
                new_commitments.append(commitment)
                logger.info(f"New/changed commitment for {hotkey}")

        if not new_commitments:
            logger.info("No eligible new commitments in catch-up phase")
            return {}

        # Download with minimal delays
        results: dict[str, DownloadResult] = {}
        for commitment in new_commitments:
            result = await self._download_single(commitment)
            results[commitment.hotkey] = result
            await asyncio.sleep(self._config.min_delay_between_downloads_seconds)

        # Update known commitments
        self._known_commitments = current_map

        logger.info(
            f"Catch-up complete: {sum(1 for r in results.values() if r.success)} "
            f"success, {sum(1 for r in results.values() if not r.success)} failed"
        )

        return results

    async def _download_single(
        self,
        commitment: ChainModelMetadata,
    ) -> DownloadResult:
        """Download a single model and return result."""
        try:
            path = await self._downloader.download_model(commitment)
            return DownloadResult(
                hotkey=commitment.hotkey,
                success=True,
                path=path,
            )
        except ModelError as e:
            logger.warning(f"Download failed for {commitment.hotkey}: {e}")
            return DownloadResult(
                hotkey=commitment.hotkey,
                success=False,
                error=e,
            )
        except Exception as e:
            logger.error(f"Unexpected error downloading {commitment.hotkey}: {e}")
            return DownloadResult(
                hotkey=commitment.hotkey,
                success=False,
                error=e,
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
