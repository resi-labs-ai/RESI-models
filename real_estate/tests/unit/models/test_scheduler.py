"""Unit tests for ModelDownloadScheduler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        mock_downloader = MagicMock()
        cached_path = Path("/cache/model.onnx")
        mock_downloader.download_model = AsyncMock(return_value=cached_path)

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )

        result = await scheduler._download_single(sample_commitment)

        assert result.success is True
        assert result.hotkey == sample_commitment.hotkey
        assert result.path == cached_path
        assert result.error is None

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

        result = await scheduler._download_single(sample_commitment)

        assert result.success is False
        assert result.hotkey == sample_commitment.hotkey
        assert result.path is None
        assert result.error == error

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

        result = await scheduler._download_single(sample_commitment)

        assert result.success is False
        assert result.hotkey == sample_commitment.hotkey
        assert result.error == error


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


class TestRunCatchUp:
    """Tests for run_catch_up method."""

    @pytest.mark.asyncio
    async def test_downloads_new_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Downloads commitments that weren't known before."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.download_model = AsyncMock(return_value=Path("/cache/model.onnx"))

        # New commitment not in known_commitments
        new_commitment = MagicMock()
        new_commitment.hotkey = "5NewHotkey"
        new_commitment.model_hash = "newhash"
        new_commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[new_commitment])
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=10000)
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Known commitments is empty - so new_commitment is new
        scheduler._known_commitments = {}

        results = await scheduler.run_catch_up()

        assert "5NewHotkey" in results
        assert results["5NewHotkey"].success is True
        mock_downloader.download_model.assert_called_once_with(new_commitment)

    @pytest.mark.asyncio
    async def test_downloads_changed_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Downloads commitments where hash has changed."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.download_model = AsyncMock(return_value=Path("/cache/model.onnx"))

        # Commitment with changed hash
        changed_commitment = MagicMock()
        changed_commitment.hotkey = "5ExistingHotkey"
        changed_commitment.model_hash = "newhash"
        changed_commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[changed_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=10000)
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Same hotkey but different hash in known
        old_commitment = MagicMock()
        old_commitment.model_hash = "oldhash"
        scheduler._known_commitments = {"5ExistingHotkey": old_commitment}

        results = await scheduler.run_catch_up()

        assert "5ExistingHotkey" in results
        mock_downloader.download_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_too_recent_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Skips commitments that are too recent (after cutoff block)."""
        scheduler_config.min_commitment_age_blocks = 100
        mock_downloader = MagicMock()
        mock_downloader.download_model = AsyncMock()

        # Commitment at block 9950, current block 10000, cutoff 9900
        recent_commitment = MagicMock()
        recent_commitment.hotkey = "5RecentHotkey"
        recent_commitment.model_hash = "hash"
        recent_commitment.block_number = 9950  # Too recent

        mock_chain_client.get_all_commitments = AsyncMock(
            return_value=[recent_commitment]
        )
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=10000)
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {}

        results = await scheduler.run_catch_up()

        assert results == {}
        mock_downloader.download_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_known_commitments(
        self,
        scheduler_config: SchedulerConfig,
        mock_chain_client: MagicMock,
    ) -> None:
        """Updates known_commitments after catch-up."""
        scheduler_config.min_delay_between_downloads_seconds = 0
        mock_downloader = MagicMock()
        mock_downloader.download_model = AsyncMock(return_value=Path("/cache/model.onnx"))

        commitment = MagicMock()
        commitment.hotkey = "5Hotkey"
        commitment.model_hash = "hash"
        commitment.block_number = 1000

        mock_chain_client.get_all_commitments = AsyncMock(return_value=[commitment])
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=10000)
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        scheduler._known_commitments = {}

        await scheduler.run_catch_up()

        assert "5Hotkey" in scheduler._known_commitments
        assert scheduler._known_commitments["5Hotkey"] == commitment

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
        mock_chain_client.get_metagraph = AsyncMock(
            return_value=MagicMock(block=10000)
        )

        scheduler = ModelDownloadScheduler(
            scheduler_config, mock_downloader, mock_chain_client
        )
        # Same commitment already known
        scheduler._known_commitments = {"5Hotkey": commitment}

        results = await scheduler.run_catch_up()

        assert results == {}


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
