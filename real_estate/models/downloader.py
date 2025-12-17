"""HuggingFace model downloader with retry and circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from .cache import ModelCache
from .errors import (
    CircuitBreakerOpenError,
    InsufficientDiskSpaceError,
    ModelDownloadError,
)
from .verifier import ModelVerifier

if TYPE_CHECKING:
    from ..chain.models import ChainModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for model downloads."""

    max_model_size_bytes: int = 200 * 1024 * 1024  # 200 MB
    max_retries: int = 4
    initial_retry_delay_seconds: int = 30  # 30s -> 1min -> 2min -> 4min
    circuit_breaker_threshold: int = 5  # consecutive failures to open circuit
    circuit_breaker_pause_minutes: int = 5
    download_timeout_seconds: int = 300


@dataclass
class _CircuitBreakerState:
    """Internal state for circuit breaker."""

    consecutive_failures: int = 0
    open_until: datetime | None = None


class ModelDownloader:
    """
    Download and cache ONNX models from HuggingFace.

    Features:
    - Pre-download checks (license, size) via HF API
    - Extrinsic verification before download
    - Hash verification after download
    - Exponential backoff retry (30s -> 1min -> 2min -> 4min)
    - Circuit breaker for rate limiting (5 failures -> pause 5 min)
    - Atomic file operations (temp -> move)

    Note: ONNX integrity is verified later in the sandboxed Docker evaluator.
    """

    def __init__(
        self,
        config: DownloadConfig,
        cache: ModelCache,
        verifier: ModelVerifier,
    ):
        """
        Initialize downloader.

        Args:
            config: Download configuration
            cache: Model cache for storing downloads
            verifier: Verifier for pre/post-download checks
        """
        self._config = config
        self._cache = cache
        self._verifier = verifier
        self._circuit_breaker = _CircuitBreakerState()

    async def download_model(self, commitment: ChainModelMetadata) -> Path:
        """
        Download and cache a model.

        Steps:
        1. Check circuit breaker
        2. Check cache (skip if hash matches)
        3. Check license via HF API
        4. Check size via HF API
        5. Verify extrinsic_record.json
        6. Download to temp file
        7. Verify hash matches commitment
        8. Atomic move to cache

        Args:
            commitment: Chain commitment with hotkey, hf_repo_id, model_hash

        Returns:
            Path to cached model

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            LicenseError: If license check fails
            ModelTooLargeError: If model exceeds size limit
            ExtrinsicVerificationError: If extrinsic check fails
            HashMismatchError: If hash doesn't match
            ModelDownloadError: If download fails after retries
        """
        hotkey = commitment.hotkey
        hf_repo_id = commitment.hf_repo_id
        expected_hash = commitment.model_hash

        if self._is_circuit_breaker_open():
            remaining = self._get_circuit_breaker_remaining_seconds()
            raise CircuitBreakerOpenError(
                f"Circuit breaker open. Retry in {remaining:.0f} seconds."
            )

        if self._cache.is_valid(hotkey, expected_hash):
            cached = self._cache.get(hotkey)
            if cached:
                logger.info(f"Using cached model for {hotkey}")
                return cached.path

        logger.info(f"Downloading model for {hotkey} from {hf_repo_id}")

        await self._verifier.check_license(hf_repo_id)

        size = await self._verifier.check_size(
            hf_repo_id, self._config.max_model_size_bytes
        )

        disk_buffer = 100_000_000  # 100MB safety margin
        free_space = self._cache.get_free_disk_space()
        if free_space < size + disk_buffer:
            raise InsufficientDiskSpaceError(
                f"Insufficient disk space: {free_space} bytes free, "
                f"need {size + disk_buffer} bytes ({size} + {disk_buffer} buffer)"
            )

        # 6. Verify extrinsic record
        await self._verifier.verify_extrinsic_record(
            hotkey=hotkey,
            hf_repo_id=hf_repo_id,
            expected_hash=expected_hash,
        )

        # 7. Download with retry
        temp_path = await self._download_with_retry(hf_repo_id)

        try:
            # 8. Verify hash
            self._verifier.verify_hash(temp_path, expected_hash)

            # 9. Move to cache
            actual_size = temp_path.stat().st_size
            temp_dir = temp_path.parent
            cached_path = self._cache.put(
                hotkey=hotkey,
                temp_model_path=temp_path,
                model_hash=expected_hash,
                size_bytes=actual_size,
            )

            # 10. Clean up temp directory (file was moved, dir may have HF cache files)
            self._cleanup_temp_dir(str(temp_dir))

            self._record_success()
            logger.info(f"Successfully downloaded and cached model for {hotkey}")
            return cached_path

        except Exception:
            # Clean up temp file on any verification failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def _download_with_retry(self, hf_repo_id: str) -> Path:
        """
        Download model.onnx with exponential backoff retry.

        Args:
            hf_repo_id: HuggingFace repository ID

        Returns:
            Path to downloaded file in temp directory

        Raises:
            ModelDownloadError: If all retries exhausted
        """
        last_error: Exception | None = None
        delay = self._config.initial_retry_delay_seconds

        for attempt in range(self._config.max_retries):
            temp_dir: str | None = None
            try:
                temp_dir = tempfile.mkdtemp(prefix="model_download_")

                # Run synchronous hf_hub_download in thread pool with timeout
                downloaded_path = await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_hub_download,
                        repo_id=hf_repo_id,
                        filename="model.onnx",
                        local_dir=temp_dir,
                        local_dir_use_symlinks=False,
                    ),
                    timeout=self._config.download_timeout_seconds,
                )

                return Path(downloaded_path)

            except RepositoryNotFoundError as e:
                self._cleanup_temp_dir(temp_dir)
                # Don't record failure - this is a permanent config error, not transient
                raise ModelDownloadError(f"Repository not found: {hf_repo_id}") from e

            except EntryNotFoundError as e:
                self._cleanup_temp_dir(temp_dir)
                # Don't record failure - this is a permanent config error, not transient
                raise ModelDownloadError(f"model.onnx not found in {hf_repo_id}") from e

            except asyncio.TimeoutError as e:
                self._cleanup_temp_dir(temp_dir)
                last_error = e
                self._record_failure()

                if attempt < self._config.max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} timed out for {hf_repo_id} "
                        f"(>{self._config.download_timeout_seconds}s). Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff

            except HfHubHTTPError as e:
                self._cleanup_temp_dir(temp_dir)
                last_error = e
                self._record_failure()

                if attempt < self._config.max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed for {hf_repo_id}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff

            except Exception as e:
                self._cleanup_temp_dir(temp_dir)
                last_error = e
                self._record_failure()

                if attempt < self._config.max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed for {hf_repo_id}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2

        raise ModelDownloadError(
            f"Failed to download model from {hf_repo_id} after "
            f"{self._config.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _cleanup_temp_dir(temp_dir: str | None) -> None:
        """Remove temp directory if it exists."""
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        state = self._circuit_breaker

        if state.open_until is None:
            return False

        now = datetime.now(UTC)
        if now < state.open_until:
            return True

        # Circuit breaker has expired - reset
        logger.info("Circuit breaker closed, resuming downloads")
        state.open_until = None
        state.consecutive_failures = 0
        return False

    def _get_circuit_breaker_remaining_seconds(self) -> float:
        """Get seconds until circuit breaker closes."""
        state = self._circuit_breaker
        if state.open_until is None:
            return 0.0
        remaining = (state.open_until - datetime.now(UTC)).total_seconds()
        return max(0.0, remaining)

    def _record_failure(self) -> None:
        """Record a download failure, potentially opening circuit breaker."""
        state = self._circuit_breaker
        state.consecutive_failures += 1

        if state.consecutive_failures >= self._config.circuit_breaker_threshold:
            state.open_until = datetime.now(UTC) + timedelta(
                minutes=self._config.circuit_breaker_pause_minutes
            )
            logger.warning(
                f"Circuit breaker opened after {state.consecutive_failures} "
                f"consecutive failures. Pausing downloads for "
                f"{self._config.circuit_breaker_pause_minutes} minutes."
            )

    def _record_success(self) -> None:
        """Record a successful download, resetting failure counter."""
        self._circuit_breaker.consecutive_failures = 0

    def is_cached(self, hotkey: str, expected_hash: str) -> bool:
        """Check if model is cached with matching hash."""
        return self._cache.is_valid(hotkey, expected_hash)

    def get_cached_path(self, hotkey: str) -> Path | None:
        """Get path to cached model if exists."""
        cached = self._cache.get(hotkey)
        return cached.path if cached else None

    def cleanup_stale_cache(self, active_hotkeys: set[str]) -> list[str]:
        """
        Remove cached models for hotkeys no longer on chain.

        Args:
            active_hotkeys: Set of hotkeys with current commitments

        Returns:
            List of removed hotkeys
        """
        return self._cache.cleanup_stale(active_hotkeys)
