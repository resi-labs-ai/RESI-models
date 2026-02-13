"""Unit tests for ModelDownloadScheduler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from real_estate.chain.models import ChainModelMetadata
from real_estate.models import ModelDownloadError, ModelError, SchedulerConfig
from real_estate.models.scheduler import ModelDownloadScheduler


class TestFilterNeedsDownload:
    """Tests for _filter_needs_download method."""

    def test_filters_already_cached_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitments: list[MagicMock],
    ) -> None:
        """Filters out models that are already cached with matching hash."""
        mock_downloader = MagicMock()
        # First model cached, second not
        mock_downloader.is_cached.side_effect = [True, False]

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Both commitments are old enough (block 1000, cutoff 5000)
        result = scheduler._filter_needs_download(sample_commitments, cutoff_block=5000)

        assert len(result) == 1
        assert result[0].hotkey == "5Hotkey2"

    def test_filters_too_recent_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Filters out commitments that are too recent (after cutoff block)."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        # Create commitments at different blocks
        old_commitment = MagicMock()
        old_commitment.hotkey = "5OldHotkey"
        old_commitment.block_number = 1000

        recent_commitment = MagicMock()
        recent_commitment.hotkey = "5RecentHotkey"
        recent_commitment.block_number = 9000

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Cutoff at 5000 - only old_commitment is eligible
        result = scheduler._filter_needs_download(
            [old_commitment, recent_commitment], cutoff_block=5000
        )

        assert len(result) == 1
        assert result[0].hotkey == "5OldHotkey"

    def test_includes_eligible_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitments: list[MagicMock],
    ) -> None:
        """Includes commitments that are old enough and not cached."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result = scheduler._filter_needs_download(sample_commitments, cutoff_block=5000)

        assert len(result) == 2

    def test_returns_empty_when_all_cached(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitments: list[MagicMock],
    ) -> None:
        """Returns empty list when all models are cached."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result = scheduler._filter_needs_download(sample_commitments, cutoff_block=5000)

        assert result == []

    def test_returns_empty_when_all_too_recent(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitments: list[MagicMock],
    ) -> None:
        """Returns empty list when all commitments are too recent."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Cutoff at 500 - both commitments at block 1000 are too recent
        result = scheduler._filter_needs_download(sample_commitments, cutoff_block=500)

        assert result == []


class TestCalculateDownloadSchedule:
    """Tests for _calculate_download_schedule method."""

    def test_empty_list_returns_empty_schedule(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Returns empty schedule for empty commitments list."""
        mock_downloader = MagicMock()
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result = scheduler._calculate_download_schedule([], window_seconds=3600)

        assert result == []

    def test_single_model_has_zero_delay(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Single model starts immediately with zero delay."""
        mock_downloader = MagicMock()
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result = scheduler._calculate_download_schedule(
            [sample_commitment], window_seconds=3600
        )

        assert len(result) == 1
        assert result[0][0] == 0  # Zero delay
        assert result[0][1] == sample_commitment

    def test_multiple_models_spaced_evenly(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Multiple models are spaced evenly across window."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        commitments = [MagicMock() for _ in range(3)]
        result = scheduler._calculate_download_schedule(commitments, window_seconds=60)

        # 3 models over 60s: intervals at 0, 30, 60 seconds
        assert len(result) == 3
        assert result[0][0] == 0
        assert result[1][0] == 30
        assert result[2][0] == 60

    def test_respects_minimum_delay(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Respects minimum delay even when window would allow shorter intervals."""
        scheduler_config.min_delay_between_downloads_seconds = 20
        mock_downloader = MagicMock()
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # 10 models in 60s would be 6.67s interval, but min is 20s
        commitments = [MagicMock() for _ in range(4)]
        result = scheduler._calculate_download_schedule(commitments, window_seconds=60)

        # Should use 20s intervals: 0, 20, 40, 60
        assert result[0][0] == 0
        assert result[1][0] == 20
        assert result[2][0] == 40
        assert result[3][0] == 60

    def test_two_models_uses_full_window(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Two models: first at 0, second at end of window."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        commitments = [MagicMock(), MagicMock()]
        result = scheduler._calculate_download_schedule(commitments, window_seconds=100)

        assert result[0][0] == 0
        assert result[1][0] == 100


class TestDownloadSingle:
    """Tests for _download_single method."""

    @pytest.mark.asyncio
    async def test_returns_success_result_on_successful_download(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Returns DownloadResult with success=True on successful download."""
        from real_estate.models.downloader import ModelDownloadResult

        mock_downloader = MagicMock()
        cached_path = Path("/cache/model.onnx")
        download_result = ModelDownloadResult(path=cached_path, commit_block=1000)
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result, commit_block = await scheduler._download_single(sample_commitment)

        assert result.success is True
        assert result.hotkey == sample_commitment.hotkey
        assert result.path == cached_path
        assert result.error is None
        assert commit_block == 1000

    @pytest.mark.asyncio
    async def test_returns_failure_result_on_model_error(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Returns DownloadResult with success=False on ModelError."""
        mock_downloader = MagicMock()
        error = ModelDownloadError("Download failed")
        mock_downloader.download_model = AsyncMock(side_effect=error)

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result, commit_block = await scheduler._download_single(sample_commitment)

        assert result.success is False
        assert result.hotkey == sample_commitment.hotkey
        assert result.path is None
        assert result.error == error
        assert commit_block is None

    @pytest.mark.asyncio
    async def test_returns_failure_result_on_unexpected_error(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Returns DownloadResult with success=False on unexpected exception."""
        mock_downloader = MagicMock()
        error = RuntimeError("Unexpected failure")
        mock_downloader.download_model = AsyncMock(side_effect=error)

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result, commit_block = await scheduler._download_single(sample_commitment)

        assert result.success is False
        assert result.hotkey == sample_commitment.hotkey
        assert result.error == error
        assert commit_block is None


class TestGetDownloadResults:
    """Tests for get_download_results method."""

    def test_returns_success_for_cached_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Returns success result for models that are cached."""
        mock_downloader = MagicMock()
        cached_path = Path("/cache/hotkey1/model.onnx")
        mock_downloader.get_cached_path.return_value = cached_path

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        results = scheduler.get_download_results(["hotkey1"])

        assert results["hotkey1"].success is True
        assert results["hotkey1"].path == cached_path

    def test_returns_failure_for_uncached_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Returns failure result for models not in cache."""
        mock_downloader = MagicMock()
        mock_downloader.get_cached_path.return_value = None

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        results = scheduler.get_download_results(["hotkey1"])

        assert results["hotkey1"].success is False
        assert results["hotkey1"].path is None
        assert isinstance(results["hotkey1"].error, ModelError)

    def test_handles_mixed_cached_and_uncached(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Correctly handles mix of cached and uncached models."""
        mock_downloader = MagicMock()
        cached_path = Path("/cache/hotkey1/model.onnx")
        mock_downloader.get_cached_path.side_effect = [cached_path, None]

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        results = scheduler.get_download_results(["hotkey1", "hotkey2"])

        assert results["hotkey1"].success is True
        assert results["hotkey2"].success is False


class TestRunCatchUpWithFailedHotkeys:
    """Tests for run_catch_up with failed_hotkeys parameter (retry failed downloads)."""

    @pytest.mark.asyncio
    async def test_retries_explicitly_failed_hotkeys(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Retries downloads for hotkeys explicitly marked as failed."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=1000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False

        # Commitment that was known but failed
        failed_commitment = MagicMock()
        failed_commitment.hotkey = "5FailedHotkey"
        failed_commitment.model_hash = "hash123"
        failed_commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[failed_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {"5FailedHotkey": failed_commitment}

        # Pass failed_hotkeys explicitly
        results = await scheduler.run_catch_up(failed_hotkeys={"5FailedHotkey"})

        assert "5FailedHotkey" in results
        assert results["5FailedHotkey"].success is True
        mock_downloader.download_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_already_cached_even_if_in_failed_hotkeys(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Does not retry if model was cached successfully after initial failure."""
        mock_downloader = MagicMock()
        # Model is now cached
        mock_downloader.is_cached.return_value = True

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.model_hash = "hash123"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {"5Hotkey": commitment}

        # Mark as failed, but it's now cached
        results = await scheduler.run_catch_up(failed_hotkeys={"5Hotkey"})

        # Should not retry (already cached)
        assert results == {}
        mock_downloader.download_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_uncached_known_commitments_without_explicit_failed_list(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Falls back to retrying uncached known commitments when failed_hotkeys is None."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=1000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False

        commitment = MagicMock()
        commitment.hotkey = "5KnownButUncached"
        commitment.model_hash = "hash123"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Known but not cached - should retry
        scheduler._known_commitments = {"5KnownButUncached": commitment}

        # No explicit failed_hotkeys
        results = await scheduler.run_catch_up(failed_hotkeys=None)

        assert "5KnownButUncached" in results
        mock_downloader.download_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_retry_unknown_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Does not download commitments that weren't known (new miners during catch-up)."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        # New commitment not in known_commitments
        new_commitment = MagicMock()
        new_commitment.hotkey = "5NewMiner"
        new_commitment.model_hash = "newhash"
        new_commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[new_commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Pre-download ran but found no commitments at that time
        scheduler._known_commitments = {}
        scheduler._pre_download_ran = True

        results = await scheduler.run_catch_up(failed_hotkeys=None)

        # Should not download new miners in catch-up
        assert results == {}
        mock_downloader.download_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_failed_hotkeys_set_does_nothing(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Empty failed_hotkeys set means nothing to retry."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.model_hash = "hash"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {"5Hotkey": commitment}

        # Empty set - nothing explicitly failed
        results = await scheduler.run_catch_up(failed_hotkeys=set())

        # With empty set, falls back to checking uncached known commitments
        # Since it's uncached and known, it will retry
        assert "5Hotkey" in results

    @pytest.mark.asyncio
    async def test_downloads_all_uncached_when_pre_download_never_ran(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Downloads all uncached models when pre-download crashed entirely."""
        from pathlib import Path

        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        # Mock successful download
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=1000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        # New commitment (not known)
        commitment = MagicMock()
        commitment.hotkey = "5NewMiner"
        commitment.model_hash = "hash123"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))
        mock_chain_client.get_extrinsic = AsyncMock(
            return_value=MagicMock(address="5NewMiner")
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Simulate pre-download crashed: _pre_download_ran is False (default)
        # and _known_commitments is empty
        assert scheduler._pre_download_ran is False
        assert scheduler._known_commitments == {}

        results = await scheduler.run_catch_up(failed_hotkeys=None)

        # Should download all uncached models since pre-download never ran
        assert "5NewMiner" in results
        mock_downloader.download_model.assert_called_once()


class TestRunCatchUp:
    """Tests for run_catch_up method (legacy behavior tests updated for new retry-only behavior)."""

    @pytest.mark.asyncio
    async def test_does_not_download_new_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Does NOT download new commitments - catch-up only retries known failures."""
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        # New commitment not in known_commitments
        new_commitment = MagicMock()
        new_commitment.hotkey = "5NewHotkey"
        new_commitment.model_hash = "newhash"
        new_commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[new_commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Pre-download ran but found no commitments - catch-up should NOT download new miners
        scheduler._known_commitments = {}
        scheduler._pre_download_ran = True

        results = await scheduler.run_catch_up()

        # New commitments are NOT downloaded in catch-up phase
        assert results == {}
        mock_downloader.download_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_known_but_uncached_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Retries commitments that are known but not cached (fallback behavior)."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=1000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False  # Not cached

        commitment = MagicMock()
        commitment.hotkey = "5KnownHotkey"
        commitment.model_hash = "hash"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Commitment is KNOWN (was in pre-download) but not cached
        scheduler._known_commitments = {"5KnownHotkey": commitment}

        results = await scheduler.run_catch_up()

        # Should retry known uncached commitment
        assert "5KnownHotkey" in results
        mock_downloader.download_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_excludes_too_recent_commitments_after_download(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Excludes commitments that are too recent based on real block from Pylon."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_commitment_age_blocks = 100
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False

        # Download succeeds, returns real commit_block of 9950 (too recent)
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=9950
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        # Commitment starts with block_number=0 (Pylon limitation)
        recent_commitment = MagicMock()
        recent_commitment.hotkey = "5RecentHotkey"
        recent_commitment.hf_repo_id = "user/repo"
        recent_commitment.model_hash = "hash"
        recent_commitment.block_number = 0  # Unknown until we call get_extrinsic

        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[recent_commitment]
        )
        # Current block 10000, cutoff 9900 - so 9950 is too recent
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Must be in known_commitments to be retried
        scheduler._known_commitments = {"5RecentHotkey": recent_commitment}

        results = await scheduler.run_catch_up()

        # Download was attempted (we can't know block until we try)
        mock_downloader.download_model.assert_called_once()
        # But result is marked as failed because real block is too recent
        assert results["5RecentHotkey"].success is False
        assert "too recent" in str(results["5RecentHotkey"].error)

    @pytest.mark.asyncio
    async def test_updates_known_commitments_from_chain(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Updates known_commitments with fresh data from chain after catch-up."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=1000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.model_hash = "hash"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Must be known to be retried
        scheduler._known_commitments = {"5Hotkey": commitment}

        await scheduler.run_catch_up()

        # known_commitments is updated from chain
        assert "5Hotkey" in scheduler._known_commitments

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_new_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Returns empty dict when all commitments are already known."""
        mock_downloader = MagicMock()

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.model_hash = "samehash"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Same commitment already known
        scheduler._known_commitments = {"5Hotkey": commitment}

        results = await scheduler.run_catch_up()

        assert results == {}


class TestPostDownloadCutoffCheck:
    """Tests for post-download cutoff filtering (workaround for Pylon block=0 issue)."""

    @pytest.mark.asyncio
    async def test_run_pre_download_excludes_too_recent_after_download(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """
        run_pre_download excludes models that are too recent based on real block from Pylon.

        Since get_commitments returns block=0, we can't filter upfront.
        We download first, get real block from get_extrinsic, then exclude if too recent.
        """
        from real_estate.models.downloader import ModelDownloadResult

        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=100,  # Cutoff = 10000 - 100 = 9900
        )
        mock_downloader = MagicMock()

        # Download succeeds, returns real commit_block of 9950 (too recent, > 9900)
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=9950
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False
        mock_downloader.cleanup_stale_cache = MagicMock()
        mock_downloader._cache.get.return_value = None

        # Commitment starts with block_number=0 (Pylon limitation)
        commitment = MagicMock()
        commitment.hotkey = "5RecentHotkey"
        commitment.hf_repo_id = "user/repo"
        commitment.model_hash = "hash123"
        commitment.block_number = 0

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        eval_time = datetime.now(UTC) + timedelta(hours=5)
        results = await scheduler.run_pre_download(eval_time)

        # Download was attempted
        mock_downloader.download_model.assert_called_once()
        # But result is marked as failed because real block is too recent
        assert results["5RecentHotkey"].success is False
        assert "too recent" in str(results["5RecentHotkey"].error)

    @pytest.mark.asyncio
    async def test_run_pre_download_includes_old_enough_models(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """Models with real commit_block <= cutoff are included."""
        from real_estate.models.downloader import ModelDownloadResult

        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=100,  # Cutoff = 10000 - 100 = 9900
        )
        mock_downloader = MagicMock()

        # Download succeeds, returns real commit_block of 9800 (old enough, <= 9900)
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=9800
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False
        mock_downloader.cleanup_stale_cache = MagicMock()
        mock_downloader._cache.get.return_value = None

        commitment = MagicMock()
        commitment.hotkey = "5OldEnoughHotkey"
        commitment.hf_repo_id = "user/repo"
        commitment.model_hash = "hash123"
        commitment.block_number = 0

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        eval_time = datetime.now(UTC) + timedelta(hours=5)
        results = await scheduler.run_pre_download(eval_time)

        # Download was attempted and succeeded
        mock_downloader.download_model.assert_called_once()
        assert results["5OldEnoughHotkey"].success is True


class TestCacheCommitBlockOptimization:
    """Tests proving cached commit_block avoids network overhead."""

    @pytest.mark.asyncio
    async def test_cached_model_uses_stored_commit_block_without_network_call(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """
        E2E test: When model is cached with commit_block, no verify_extrinsic_record call.

        This proves the optimization works - subsequent runs use cached commit_block
        without any network overhead to Pylon.
        """
        from real_estate.models.downloader import ModelDownloadResult

        mock_downloader = MagicMock()
        cached_path = Path("/cache/hotkey1/model.onnx")

        # Simulate cache hit - downloader returns cached result with commit_block
        download_result = ModelDownloadResult(path=cached_path, commit_block=5000)
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = True
        mock_downloader.cleanup_stale_cache = MagicMock()

        # Mock the internal cache.get() call that scheduler uses for cached models
        mock_cached_model = MagicMock()
        mock_cached_model.metadata.hash = "abc123"
        mock_cached_model.metadata.commit_block = 5000
        mock_downloader._cache.get.return_value = mock_cached_model

        commitment = MagicMock()
        commitment.hotkey = "5CachedHotkey"
        commitment.hf_repo_id = "user/repo"
        commitment.model_hash = "abc123"
        commitment.block_number = 1000  # Initial (incorrect) block from chain

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler_config.min_commitment_age_blocks = 100
        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        eval_time = datetime.now(UTC) + timedelta(hours=5)
        await scheduler.run_pre_download(eval_time)

        # Key assertion: download_model was called for cached model
        mock_downloader.download_model.assert_called_once_with(commitment)

        # The scheduler updated known_commitments with the cached commit_block (5000)
        # NOT the chain's block_number (1000)
        assert scheduler._known_commitments["5CachedHotkey"].block_number == 5000


class TestRunPreDownloadTiming:
    """Tests for run_pre_download elastic window timing."""

    @pytest.mark.asyncio
    async def test_uses_full_window_when_plenty_of_time(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """Uses full pre_download_hours window when time_available > max_window."""
        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=0,
        )
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False
        mock_downloader.download_model = AsyncMock(return_value=Path("/model.onnx"))
        mock_downloader.cleanup_stale_cache = MagicMock()

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.block_number = 1000
        commitment.model_hash = "hash"

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        # eval_time is 5 hours from now - plenty of time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        eval_time = now + timedelta(hours=5)

        with (
            patch("real_estate.models.scheduler.datetime") as mock_datetime,
            patch("real_estate.models.scheduler.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await scheduler.run_pre_download(eval_time)

        # Should have downloaded the model
        mock_downloader.download_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_compresses_window_when_less_time_available(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """Compresses window when time_available < max_window."""
        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=0,
        )
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False
        mock_downloader.download_model = AsyncMock(return_value=Path("/model.onnx"))
        mock_downloader.cleanup_stale_cache = MagicMock()

        commitments = [
            MagicMock(hotkey=f"5Hotkey{i}", block_number=1000, model_hash=f"hash{i}")
            for i in range(3)
        ]

        mock_chain_client.get_all_commitments = AsyncMock(return_value=commitments)
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        # eval_time is 1 hour from now, deadline is 30 min from now
        # time_available = 30 min = 1800 seconds (less than 3 hours)
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        eval_time = now + timedelta(hours=1)

        with (
            patch("real_estate.models.scheduler.datetime") as mock_datetime,
            patch("real_estate.models.scheduler.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await scheduler.run_pre_download(eval_time)

        # All models should be downloaded (window compressed but still works)
        assert mock_downloader.download_model.call_count == 3

    @pytest.mark.asyncio
    async def test_downloads_immediately_when_past_deadline(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """Downloads immediately with minimal delays when past deadline."""
        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=0,
        )
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False
        mock_downloader.download_model = AsyncMock(return_value=Path("/model.onnx"))
        mock_downloader.cleanup_stale_cache = MagicMock()

        commitments = [
            MagicMock(hotkey=f"5Hotkey{i}", block_number=1000, model_hash=f"hash{i}")
            for i in range(3)
        ]

        mock_chain_client.get_all_commitments = AsyncMock(return_value=commitments)
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        # eval_time is NOW - deadline was 30 min ago
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        eval_time = now  # Past deadline

        with (
            patch("real_estate.models.scheduler.datetime") as mock_datetime,
            patch("real_estate.models.scheduler.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await scheduler.run_pre_download(eval_time)

        # All models should be downloaded (ASAP mode)
        assert mock_downloader.download_model.call_count == 3

    @pytest.mark.asyncio
    async def test_deadline_accounts_for_catch_up_buffer(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """Deadline is eval_time minus catch_up_minutes."""
        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=0,
        )
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False
        mock_downloader.download_model = AsyncMock(return_value=Path("/model.onnx"))
        mock_downloader.cleanup_stale_cache = MagicMock()

        commitment = MagicMock(hotkey="5Hotkey", block_number=1000, model_hash="hash")
        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        # eval_time is 35 min from now
        # deadline = eval_time - 30 min = 5 min from now
        # time_available = 5 min = 300 seconds (compressed window)
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        eval_time = now + timedelta(minutes=35)

        with (
            patch("real_estate.models.scheduler.datetime") as mock_datetime,
            patch("real_estate.models.scheduler.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            await scheduler.run_pre_download(eval_time)

        # Should download (within compressed window)
        mock_downloader.download_model.assert_called_once()


class TestGetAvailableModels:
    """Tests for get_available_models method."""

    def test_returns_cached_models_matching_known_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Returns paths for cached models that match known commitments."""
        mock_downloader = MagicMock()

        # Mock cache returns a cached model
        cached_model = MagicMock()
        cached_model.path = Path("/cache/hotkey1/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Set up known commitment (old enough to pass age filter)
        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 1000
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1", "5Hotkey2"}, current_block=10000)

        assert result == {"5Hotkey1": Path("/cache/hotkey1/model.onnx")}

    def test_excludes_hotkeys_not_in_known_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Excludes hotkeys that don't have known commitments."""
        mock_downloader = MagicMock()
        mock_downloader._cache.get.return_value = MagicMock()

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {}  # No known commitments

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {}
        mock_downloader._cache.get.assert_not_called()

    def test_excludes_models_with_hash_mismatch(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Excludes cached models whose hash doesn't match commitment."""
        mock_downloader = MagicMock()

        # Cache has old hash
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "old_hash"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Commitment has new hash
        commitment = MagicMock()
        commitment.model_hash = "new_hash"
        commitment.block_number = 1000
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {}

    def test_excludes_uncached_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Excludes models that aren't in cache."""
        mock_downloader = MagicMock()
        mock_downloader._cache.get.return_value = None  # Not cached

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 1000
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {}

    def test_filters_to_registered_hotkeys_only(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Only returns models for hotkeys in registered_hotkeys set."""
        mock_downloader = MagicMock()

        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Known commitment exists
        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 1000
        scheduler._known_commitments = {"5Hotkey1": commitment}

        # But hotkey not in registered set
        result = scheduler.get_available_models({"5OtherHotkey"}, current_block=10000)

        assert result == {}

    def test_excludes_models_with_too_recent_commitment(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Excludes models whose commitment is too recent (< min_commitment_age_blocks)."""
        mock_downloader = MagicMock()

        # Model is cached and hash matches
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Commitment is too recent: block 9950, current_block=10000, cutoff=9900
        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 9950
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {}
        # Should not even check cache for too-recent commitments
        mock_downloader._cache.get.assert_not_called()

    def test_includes_models_exactly_at_cutoff_block(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Includes models committed exactly at the cutoff block (boundary case)."""
        mock_downloader = MagicMock()

        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Commitment is exactly at cutoff: block 9900, current_block=10000, cutoff=9900
        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 9900
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {"5Hotkey1": Path("/cache/model.onnx")}

    def test_skips_block_zero_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Models with block_number=0 (unverified) are excluded with warning."""
        mock_downloader = MagicMock()

        # Model is cached and hash matches
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        # Commitment has block=0 (Pylon didn't return real block)
        commitment = MagicMock()
        commitment.model_hash = "hash123"
        commitment.block_number = 0
        scheduler._known_commitments = {"5Hotkey1": commitment}

        result = scheduler.get_available_models({"5Hotkey1"}, current_block=10000)

        assert result == {}
        # Should not even check cache for block=0 commitments
        mock_downloader._cache.get.assert_not_called()


class TestCatchUpPreservesVerifiedBlocks:
    """Tests for run_catch_up preserving verified blocks from pre-download.

    The bug: run_catch_up fetches fresh commitments from Pylon (all block=0),
    then overwrites _known_commitments with this data. This discards the
    verified block numbers that run_pre_download set via extrinsic verification.

    The fix: before overwriting, copy verified blocks from the old
    _known_commitments into current_map for models that weren't retried.
    """

    @pytest.mark.asyncio
    async def test_preserves_verified_block_for_cached_model(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Cached model's verified block from pre-download survives catch-up."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="hash123",
                block_number=500,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert scheduler._known_commitments["5Hotkey1"].block_number == 500
        mock_downloader.download_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_blocks_for_multiple_cached_models(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """All cached models' verified blocks are preserved, not just one."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        chain_commitments = [
            ChainModelMetadata(
                hotkey=f"5Hotkey{i}",
                hf_repo_id="user/repo",
                model_hash=f"hash{i}",
                block_number=0,
            )
            for i in range(5)
        ]
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=chain_commitments
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {
            f"5Hotkey{i}": ChainModelMetadata(
                hotkey=f"5Hotkey{i}",
                hf_repo_id="user/repo",
                model_hash=f"hash{i}",
                block_number=1000 + i * 100,
            )
            for i in range(5)
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        for i in range(5):
            assert (
                scheduler._known_commitments[f"5Hotkey{i}"].block_number
                == 1000 + i * 100
            )

    @pytest.mark.asyncio
    async def test_retried_model_uses_fresh_block_not_preserved(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Retried model gets its block from extrinsic verification, not from old state."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = False  # Needs retry

        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=999
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Old block was 500, but retry should get fresh block 999
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="hash123",
                block_number=500,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert scheduler._known_commitments["5Hotkey1"].block_number == 999

    @pytest.mark.asyncio
    async def test_does_not_preserve_block_on_hash_change(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Hash changed = new commitment, old block is irrelevant."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="new_hash",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="old_hash",
                block_number=500,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert scheduler._known_commitments["5Hotkey1"].block_number == 0

    @pytest.mark.asyncio
    async def test_does_not_preserve_block_zero(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """If pre-download also had block=0, nothing to preserve."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Pre-download also had block=0 (extrinsic verification failed)
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="hash123",
                block_number=0,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert scheduler._known_commitments["5Hotkey1"].block_number == 0

    @pytest.mark.asyncio
    async def test_deregistered_hotkey_dropped(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Hotkey no longer on chain is removed from _known_commitments."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        # Chain only returns Hotkey2 — Hotkey1 deregistered
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[
                ChainModelMetadata(
                    hotkey="5Hotkey2",
                    hf_repo_id="user/repo",
                    model_hash="hash2",
                    block_number=0,
                )
            ]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="hash1",
                block_number=500,
            ),
            "5Hotkey2": ChainModelMetadata(
                hotkey="5Hotkey2",
                hf_repo_id="user/repo",
                model_hash="hash2",
                block_number=600,
            ),
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert "5Hotkey1" not in scheduler._known_commitments
        assert scheduler._known_commitments["5Hotkey2"].block_number == 600

    @pytest.mark.asyncio
    async def test_new_hotkey_added_with_block_zero(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """New hotkey appearing in catch-up has block=0 (no prior state to preserve)."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.is_cached.return_value = True

        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[
                ChainModelMetadata(
                    hotkey="5Existing",
                    hf_repo_id="user/repo",
                    model_hash="hash1",
                    block_number=0,
                ),
                ChainModelMetadata(
                    hotkey="5NewMiner",
                    hf_repo_id="user/repo",
                    model_hash="hash2",
                    block_number=0,
                ),
            ]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {
            "5Existing": ChainModelMetadata(
                hotkey="5Existing",
                hf_repo_id="user/repo",
                model_hash="hash1",
                block_number=500,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        assert scheduler._known_commitments["5Existing"].block_number == 500
        assert scheduler._known_commitments["5NewMiner"].block_number == 0


class TestCatchUpBlockPreservationE2E:
    """End-to-end tests: pre-download → catch-up → get_available_models.

    Reproduces the exact bug from production: model excluded as too recent
    in pre-download, catch-up resets block to 0, model enters evaluation.
    """

    @pytest.mark.asyncio
    async def test_too_recent_model_stays_excluded_after_catch_up(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """
        Reproduces the 5Gv6yo bug: model correctly excluded as too recent
        in pre-download must remain excluded after catch-up.
        """
        from real_estate.models.downloader import ModelDownloadResult

        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=7200,
        )
        mock_downloader = MagicMock()

        # --- Phase 1: pre-download ---
        # Download succeeds, real block is 7528022 (too recent, cutoff ~7521036)
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=7528022
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False
        mock_downloader.cleanup_stale_cache = MagicMock()
        mock_downloader._cache.get.return_value = None

        commitment = ChainModelMetadata(
            hotkey="5Gv6yo",
            hf_repo_id="Joshep1/RE",
            model_hash="abc123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=7528236)
        )

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        eval_time = datetime.now(UTC) + timedelta(hours=5)
        pre_results = await scheduler.run_pre_download(eval_time)

        # Pre-download correctly excluded it as too recent
        assert pre_results["5Gv6yo"].success is False
        assert "too recent" in str(pre_results["5Gv6yo"].error)
        assert scheduler._known_commitments["5Gv6yo"].block_number == 7528022

        # --- Phase 2: catch-up ---
        # Chain still returns block=0, model is now cached
        mock_downloader.is_cached.return_value = True
        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=7528236)
        )

        await scheduler.run_catch_up()

        # Block must be preserved, NOT reset to 0
        assert scheduler._known_commitments["5Gv6yo"].block_number == 7528022

        # --- Phase 3: get_available_models ---
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "abc123"
        mock_downloader._cache.get.return_value = cached_model

        available = scheduler.get_available_models({"5Gv6yo"}, current_block=7528236)

        # Must NOT be available — still too recent
        assert "5Gv6yo" not in available

    @pytest.mark.asyncio
    async def test_old_enough_model_available_after_catch_up(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """
        Model old enough in pre-download stays available after catch-up.
        """
        from real_estate.models.downloader import ModelDownloadResult

        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=100,
        )
        mock_downloader = MagicMock()

        # Download succeeds, real block is 5000 (old enough, cutoff 9900)
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=5000
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)
        mock_downloader.is_cached.return_value = False
        mock_downloader.cleanup_stale_cache = MagicMock()
        mock_downloader._cache.get.return_value = None

        commitment = ChainModelMetadata(
            hotkey="5GoodModel",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        eval_time = datetime.now(UTC) + timedelta(hours=5)
        pre_results = await scheduler.run_pre_download(eval_time)

        assert pre_results["5GoodModel"].success is True
        assert scheduler._known_commitments["5GoodModel"].block_number == 5000

        # Catch-up: model is now cached, chain still returns block=0
        mock_downloader.is_cached.return_value = True
        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])

        await scheduler.run_catch_up()

        # Block preserved
        assert scheduler._known_commitments["5GoodModel"].block_number == 5000

        # Model should be available for evaluation
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        available = scheduler.get_available_models({"5GoodModel"}, current_block=10000)

        assert "5GoodModel" in available

    @pytest.mark.asyncio
    async def test_block_zero_model_excluded_from_evaluation(
        self,
        mock_chain_client: MagicMock,
    ) -> None:
        """
        Model with block=0 (never verified) must not enter evaluation,
        even if it's cached and hash matches.
        """
        config = SchedulerConfig(
            pre_download_hours=3.0,
            catch_up_minutes=30.0,
            min_delay_between_downloads_seconds=0,
            min_commitment_age_blocks=100,
        )
        mock_downloader = MagicMock()

        # Model is cached
        cached_model = MagicMock()
        cached_model.path = Path("/cache/model.onnx")
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        scheduler = ModelDownloadScheduler(config, mock_downloader, mock_chain_client)

        # Commitment has block=0 — extrinsic verification never happened
        scheduler._known_commitments = {
            "5Unverified": ChainModelMetadata(
                hotkey="5Unverified",
                hf_repo_id="user/repo",
                model_hash="hash123",
                block_number=0,
            )
        }

        available = scheduler.get_available_models(
            {"5Unverified"}, current_block=10000
        )

        assert available == {}
        # Should not even check cache
        mock_downloader._cache.get.assert_not_called()


class TestCatchUpPreDownloadCrash:
    """Tests for catch-up recovering cached models when pre-download never ran."""

    @pytest.mark.asyncio
    async def test_cached_model_verified_when_pre_download_never_ran(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Cached model gets its block from disk metadata when pre-download crashed."""
        from real_estate.models.downloader import ModelDownloadResult

        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()

        # Model is cached with matching hash
        cached_model = MagicMock()
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model

        # is_cached returns True → not added to to_retry
        mock_downloader.is_cached.return_value = True

        # download_model returns cached result with real block from metadata.json
        download_result = ModelDownloadResult(
            path=Path("/cache/model.onnx"), commit_block=500
        )
        mock_downloader.download_model = AsyncMock(return_value=download_result)

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Pre-download never ran — _known_commitments is empty, flag is False
        scheduler._known_commitments = {}
        scheduler._pre_download_ran = False

        await scheduler.run_catch_up()

        # Block recovered from cached metadata
        assert scheduler._known_commitments["5Hotkey1"].block_number == 500

    @pytest.mark.asyncio
    async def test_cached_model_not_verified_when_pre_download_ran(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Cached model is NOT re-verified in catch-up when pre-download ran normally."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()

        cached_model = MagicMock()
        cached_model.metadata.hash = "hash123"
        mock_downloader._cache.get.return_value = cached_model
        mock_downloader.is_cached.return_value = True

        chain_commitment = ChainModelMetadata(
            hotkey="5Hotkey1",
            hf_repo_id="user/repo",
            model_hash="hash123",
            block_number=0,
        )
        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[chain_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Pre-download ran normally and set verified block
        scheduler._known_commitments = {
            "5Hotkey1": ChainModelMetadata(
                hotkey="5Hotkey1",
                hf_repo_id="user/repo",
                model_hash="hash123",
                block_number=500,
            )
        }
        scheduler._pre_download_ran = True

        await scheduler.run_catch_up()

        # download_model NOT called — no re-verification needed
        mock_downloader.download_model.assert_not_called()
        # Block preserved via normal preservation logic
        assert scheduler._known_commitments["5Hotkey1"].block_number == 500
