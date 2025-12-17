"""Shared fixtures for models unit tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from real_estate.models import (
    DownloadConfig,
    ModelCache,
    ModelVerifier,
    SchedulerConfig,
)
from real_estate.models.models import CachedModelMetadata


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "model_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir: Path) -> ModelCache:
    """Create ModelCache instance with temp directory."""
    return ModelCache(temp_cache_dir)


@pytest.fixture
def sample_metadata() -> CachedModelMetadata:
    """Sample cached model metadata."""
    return CachedModelMetadata(hash="abc12345", size_bytes=1000)


@pytest.fixture
def download_config() -> DownloadConfig:
    """Download config with fast settings for tests."""
    return DownloadConfig(
        max_model_size_bytes=100 * 1024 * 1024,  # 100MB
        max_retries=3,
        initial_retry_delay_seconds=0,  # Fast tests
        circuit_breaker_threshold=3,
        circuit_breaker_pause_minutes=1,
    )


@pytest.fixture
def scheduler_config() -> SchedulerConfig:
    """Scheduler config for tests."""
    return SchedulerConfig(
        pre_download_hours=1.0,
        catch_up_minutes=10.0,
        min_delay_between_downloads_seconds=0,
        min_commitment_age_blocks=100,
    )


@pytest.fixture
def mock_cache() -> MagicMock:
    """Mock ModelCache."""
    cache = MagicMock(spec=ModelCache)
    cache.is_valid.return_value = False
    cache.get.return_value = None
    cache.get_free_disk_space.return_value = 1_000_000_000  # 1GB
    return cache


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Mock ModelVerifier."""
    verifier = MagicMock(spec=ModelVerifier)
    verifier.check_license = AsyncMock()
    verifier.check_size = AsyncMock(return_value=1000)
    verifier.verify_extrinsic_record = AsyncMock()
    verifier.verify_hash = MagicMock()
    return verifier


@pytest.fixture
def mock_chain_client() -> MagicMock:
    """Mock ChainClient."""
    client = MagicMock()
    client.get_all_commitments = AsyncMock(return_value=[])
    client.get_metagraph = AsyncMock(return_value=MagicMock(block=10000))
    client.get_extrinsic = AsyncMock(return_value=None)
    return client


@pytest.fixture
def sample_commitment() -> MagicMock:
    """Create a mock ChainModelMetadata."""
    commitment = MagicMock()
    commitment.hotkey = "5TestHotkey123456789"
    commitment.hf_repo_id = "testuser/test-model"
    commitment.model_hash = "abc12345"
    commitment.block_number = 1000
    return commitment


@pytest.fixture
def sample_commitments() -> list[MagicMock]:
    """Create list of mock commitments."""
    c1 = MagicMock()
    c1.hotkey = "5Hotkey1"
    c1.hf_repo_id = "user/model1"
    c1.model_hash = "hash1111"
    c1.block_number = 1000

    c2 = MagicMock()
    c2.hotkey = "5Hotkey2"
    c2.hf_repo_id = "user/model2"
    c2.model_hash = "hash2222"
    c2.block_number = 1000

    return [c1, c2]
