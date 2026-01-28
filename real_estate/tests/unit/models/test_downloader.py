"""Unit tests for ModelDownloader."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from real_estate.models import (
    CircuitBreakerOpenError,
    DownloadConfig,
    InsufficientDiskSpaceError,
    ModelDownloader,
    ModelDownloadError,
)


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_closed_initially(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Circuit breaker should be closed when downloader is created."""
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        assert downloader._is_circuit_breaker_open() is False
        assert downloader._circuit_breaker.consecutive_failures == 0
        assert downloader._circuit_breaker.open_until is None

    def test_circuit_breaker_opens_after_threshold_failures(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Circuit breaker opens after consecutive failures reach threshold."""
        download_config.circuit_breaker_threshold = 3
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Record failures up to threshold
        for _ in range(3):
            downloader._record_failure()

        assert downloader._is_circuit_breaker_open() is True
        assert downloader._circuit_breaker.open_until is not None

    def test_circuit_breaker_does_not_open_before_threshold(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Circuit breaker stays closed before reaching threshold."""
        download_config.circuit_breaker_threshold = 3
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Record failures below threshold
        for _ in range(2):
            downloader._record_failure()

        assert downloader._is_circuit_breaker_open() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_raises_error_when_open(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Download raises CircuitBreakerOpenError when circuit is open."""
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Force circuit breaker open
        downloader._circuit_breaker.open_until = datetime.now(UTC) + timedelta(
            minutes=5
        )

        with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker open"):
            await downloader.download_model(sample_commitment)

    def test_circuit_breaker_resets_on_success(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Circuit breaker failure count resets after success."""
        download_config.circuit_breaker_threshold = 5
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Record some failures
        for _ in range(3):
            downloader._record_failure()

        assert downloader._circuit_breaker.consecutive_failures == 3

        # Record success
        downloader._record_success()

        assert downloader._circuit_breaker.consecutive_failures == 0

    def test_circuit_breaker_closes_after_pause_duration(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Circuit breaker closes after pause duration expires."""
        download_config.circuit_breaker_pause_minutes = 1
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Set circuit breaker to have expired 1 minute ago
        downloader._circuit_breaker.open_until = datetime.now(UTC) - timedelta(
            minutes=1
        )
        downloader._circuit_breaker.consecutive_failures = 5

        # Should be closed now
        assert downloader._is_circuit_breaker_open() is False
        # And state should be reset
        assert downloader._circuit_breaker.consecutive_failures == 0
        assert downloader._circuit_breaker.open_until is None

    def test_get_circuit_breaker_remaining_seconds(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Should return correct remaining seconds until circuit closes."""
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        # Set to close in 60 seconds
        downloader._circuit_breaker.open_until = datetime.now(UTC) + timedelta(
            seconds=60
        )

        remaining = downloader._get_circuit_breaker_remaining_seconds()
        assert 58 <= remaining <= 62  # Allow small timing variance


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retries_on_http_error(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Retries download on transient HTTP errors."""
        download_config.max_retries = 3
        download_config.initial_retry_delay_seconds = 0
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        call_count = 0
        downloaded_file = tmp_path / "model.onnx"
        downloaded_file.write_bytes(b"model content")

        def mock_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPError("Transient error")
            return str(downloaded_file)

        with patch(
            "real_estate.models.downloader.hf_hub_download", side_effect=mock_download
        ):
            path = await downloader._download_with_retry("user/repo", "model.onnx")

        assert call_count == 3
        assert path == downloaded_file

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Verifies exponential backoff between retries."""
        download_config.max_retries = 4
        download_config.initial_retry_delay_seconds = 1
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        delays: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            delays.append(seconds)

        with (
            patch("asyncio.sleep", new=mock_sleep),
            patch(
                "real_estate.models.downloader.hf_hub_download",
                side_effect=httpx.HTTPError("Error"),
            ),
            pytest.raises(ModelDownloadError),
        ):
            await downloader._download_with_retry("user/repo", "model.onnx")

        # 4 retries = 3 sleeps with exponential backoff: 1, 2, 4
        assert delays == [1, 2, 4]

    @pytest.mark.asyncio
    async def test_stops_retrying_after_max_attempts(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Stops retrying after max attempts and raises ModelDownloadError."""
        download_config.max_retries = 2
        download_config.initial_retry_delay_seconds = 0
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        call_count = 0

        def mock_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPError("Persistent error")

        with (
            patch(
                "real_estate.models.downloader.hf_hub_download",
                side_effect=mock_download,
            ),
            pytest.raises(ModelDownloadError, match="Failed to download model"),
        ):
            await downloader._download_with_retry("user/repo", "model.onnx")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_on_repository_not_found(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Does not retry on RepositoryNotFoundError (permanent failure)."""
        download_config.max_retries = 3
        download_config.initial_retry_delay_seconds = 0
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        call_count = 0

        # Create a proper mock for the exception (requires response kwarg)
        mock_response = MagicMock()
        mock_response.status_code = 404

        def mock_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RepositoryNotFoundError("Repo not found", response=mock_response)

        with (
            patch(
                "real_estate.models.downloader.hf_hub_download",
                side_effect=mock_download,
            ),
            pytest.raises(ModelDownloadError, match="Repository not found"),
        ):
            await downloader._download_with_retry("user/repo", "model.onnx")

        # Should fail immediately without retrying
        assert call_count == 1
        # Should NOT count toward circuit breaker (permanent error, not transient)
        assert downloader._circuit_breaker.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_does_not_retry_on_entry_not_found(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Does not retry on EntryNotFoundError (permanent failure)."""
        download_config.max_retries = 3
        download_config.initial_retry_delay_seconds = 0
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        call_count = 0

        def mock_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # EntryNotFoundError is a simple exception (doesn't extend HfHubHTTPError)
            raise EntryNotFoundError("model.onnx not found")

        with (
            patch(
                "real_estate.models.downloader.hf_hub_download",
                side_effect=mock_download,
            ),
            pytest.raises(ModelDownloadError, match="model.onnx not found"),
        ):
            await downloader._download_with_retry("user/repo", "model.onnx")

        # Should fail immediately without retrying
        assert call_count == 1
        # Should NOT count toward circuit breaker (permanent error, not transient)
        assert downloader._circuit_breaker.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_retries_on_timeout(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Retries download when timeout occurs."""
        import asyncio

        download_config.max_retries = 3
        download_config.initial_retry_delay_seconds = 0
        download_config.download_timeout_seconds = 1  # 1 second timeout
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        call_count = 0

        def slow_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError("Download timed out")

        delays: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            delays.append(seconds)

        with (
            patch("asyncio.sleep", new=mock_sleep),
            patch(
                "real_estate.models.downloader.hf_hub_download",
                side_effect=slow_download,
            ),
            pytest.raises(ModelDownloadError, match="Failed to download model"),
        ):
            await downloader._download_with_retry("user/repo", "model.onnx")

        # Should retry all attempts
        assert call_count == 3
        # Should count toward circuit breaker (transient error)
        assert downloader._circuit_breaker.consecutive_failures == 3


class TestDiskSpaceCheck:
    """Tests for disk space verification."""

    @pytest.mark.asyncio
    async def test_raises_error_when_insufficient_disk_space(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Raises InsufficientDiskSpaceError when disk is full."""
        mock_cache.get_free_disk_space.return_value = 50_000_000  # 50MB free
        mock_verifier.find_onnx_file = AsyncMock(
            return_value=("model.onnx", 100_000_000)
        )

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with pytest.raises(InsufficientDiskSpaceError, match="Insufficient disk space"):
            await downloader.download_model(sample_commitment)

    @pytest.mark.asyncio
    async def test_includes_buffer_in_space_calculation(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
    ) -> None:
        """Disk space check includes 100MB safety buffer."""
        # Model is 50MB, but with 100MB buffer we need 150MB
        mock_verifier.find_onnx_file = AsyncMock(
            return_value=("model.onnx", 50_000_000)
        )
        mock_cache.get_free_disk_space.return_value = (
            120_000_000  # 120MB free (< 150MB)
        )

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with pytest.raises(InsufficientDiskSpaceError):
            await downloader.download_model(sample_commitment)

    @pytest.mark.asyncio
    async def test_passes_when_sufficient_disk_space(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Proceeds with download when disk space is sufficient."""
        mock_verifier.find_onnx_file = AsyncMock(
            return_value=("model.onnx", 50_000_000)
        )
        mock_cache.get_free_disk_space.return_value = 500_000_000  # 500MB free

        # Create mock downloaded file
        downloaded_file = tmp_path / "model.onnx"
        downloaded_file.write_bytes(b"model content")

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with (
            patch(
                "real_estate.models.downloader.hf_hub_download",
                return_value=str(downloaded_file),
            ),
            contextlib.suppress(Exception),
        ):
            # Should get past disk space check and proceed to download
            # (will fail at hash verification since we're not setting that up)
            await downloader.download_model(sample_commitment)

        # Verify disk space was checked
        mock_cache.get_free_disk_space.assert_called_once()


class TestDownloadModel:
    """Tests for the main download_model method."""

    @pytest.mark.asyncio
    async def test_returns_cached_path_when_valid_cache(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Returns cached model path when cache is valid."""
        cached_path = tmp_path / "cached" / "model.onnx"
        cached_model = MagicMock()
        cached_model.path = cached_path
        cached_model.metadata.commit_block = 1000  # Set commit_block for cache hit

        mock_cache.is_valid.return_value = True
        mock_cache.get.return_value = cached_model

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)
        result = await downloader.download_model(sample_commitment)

        assert result.path == cached_path
        assert result.commit_block == 1000
        # No network calls needed - commit_block is in cache
        mock_verifier.verify_extrinsic_record.assert_not_called()
        mock_verifier.check_license.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_all_verifications_in_order(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Calls all verification steps in correct order."""
        mock_cache.is_valid.return_value = False
        mock_verifier.find_onnx_file = AsyncMock(return_value=("model.onnx", 1000))

        downloaded_file = tmp_path / "model.onnx"
        downloaded_file.write_bytes(b"model content")

        cached_path = tmp_path / "cached" / "model.onnx"
        mock_cache.put.return_value = cached_path

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            return_value=str(downloaded_file),
        ):
            result = await downloader.download_model(sample_commitment)

        # Verify all steps were called
        mock_verifier.check_license.assert_called_once_with(
            sample_commitment.hf_repo_id
        )
        mock_verifier.find_onnx_file.assert_called_once()
        mock_verifier.verify_extrinsic_record.assert_called_once()
        mock_verifier.verify_hash.assert_called_once()
        mock_cache.put.assert_called_once()
        assert result.path == cached_path
        assert result.commit_block == 1000

    @pytest.mark.asyncio
    async def test_passes_commit_block_to_cache_put(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Passes commit_block from Pylon to cache.put() for new downloads."""
        mock_cache.is_valid.return_value = False
        mock_verifier.find_onnx_file = AsyncMock(return_value=("model.onnx", 1000))
        mock_verifier.verify_extrinsic_record = AsyncMock(return_value=7500)

        downloaded_file = tmp_path / "model.onnx"
        downloaded_file.write_bytes(b"model content")

        cached_path = tmp_path / "cached" / "model.onnx"
        mock_cache.put.return_value = cached_path

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            return_value=str(downloaded_file),
        ):
            result = await downloader.download_model(sample_commitment)

        # Verify commit_block from Pylon is passed to cache.put()
        put_call_kwargs = mock_cache.put.call_args[1]
        assert put_call_kwargs["commit_block"] == 7500
        assert result.commit_block == 7500

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file_on_hash_failure(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Cleans up temp file when hash verification fails."""
        from real_estate.models import HashMismatchError

        mock_cache.is_valid.return_value = False
        mock_verifier.find_onnx_file = AsyncMock(return_value=("model.onnx", 1000))

        # Create temp file that will be cleaned up
        downloaded_file = tmp_path / "model.onnx"
        downloaded_file.write_bytes(b"model content")

        # Make hash verification fail
        mock_verifier.verify_hash.side_effect = HashMismatchError(
            "Hash mismatch: computed abc, expected xyz"
        )

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with (
            patch(
                "real_estate.models.downloader.hf_hub_download",
                return_value=str(downloaded_file),
            ),
            pytest.raises(HashMismatchError),
        ):
            await downloader.download_model(sample_commitment)

        # Temp file should be cleaned up
        assert not downloaded_file.exists()

    @pytest.mark.asyncio
    async def test_downloads_custom_onnx_filename(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        sample_commitment: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Downloads using the custom filename from find_onnx_file."""
        mock_cache.is_valid.return_value = False
        # Verifier returns custom filename
        mock_verifier.find_onnx_file = AsyncMock(
            return_value=("my_custom_model.onnx", 1000)
        )

        downloaded_file = tmp_path / "my_custom_model.onnx"
        downloaded_file.write_bytes(b"model content")

        cached_path = tmp_path / "cached" / "model.onnx"
        mock_cache.put.return_value = cached_path

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            return_value=str(downloaded_file),
        ) as mock_download:
            await downloader.download_model(sample_commitment)

        # Verify hf_hub_download was called with custom filename
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["filename"] == "my_custom_model.onnx"


class TestIsCached:
    """Tests for is_cached method."""

    def test_returns_true_when_cached_and_hash_matches(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Returns True when model is cached with matching hash."""
        mock_cache.is_valid.return_value = True

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)
        result = downloader.is_cached("test_hotkey", "expected_hash")

        assert result is True
        mock_cache.is_valid.assert_called_once_with("test_hotkey", "expected_hash")

    def test_returns_false_when_not_cached(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Returns False when model is not cached."""
        mock_cache.is_valid.return_value = False

        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)
        result = downloader.is_cached("test_hotkey", "expected_hash")

        assert result is False
