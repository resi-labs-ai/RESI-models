"""Factory functions for creating model management components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .cache import ModelCache
from .downloader import DownloadConfig, ModelDownloader
from .scheduler import ModelDownloadScheduler, SchedulerConfig
from .verifier import ModelVerifier

if TYPE_CHECKING:
    from ..chain.client import ChainClient


def create_model_scheduler(
    chain_client: ChainClient,
    cache_dir: Path,
    download_config: DownloadConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
    required_license: str | None = None,
) -> ModelDownloadScheduler:
    """
    Create a fully-wired ModelDownloadScheduler.

    This is the main entry point for the models module.
    Handles all internal wiring of cache, verifier, and downloader.

    Args:
        chain_client: Chain client for commitment queries
        cache_dir: Directory for caching downloaded models
        download_config: Optional custom download config (uses defaults if None)
        scheduler_config: Optional custom scheduler config (uses defaults if None)
        required_license: Optional custom required license string

    Returns:
        Ready-to-use ModelDownloadScheduler

    Example:
        scheduler = create_model_scheduler(chain_client, Path("./models_cache"))
        results = await scheduler.run_pre_download(eval_time)
    """
    cache = ModelCache(cache_dir)

    verifier_kwargs = {}
    if required_license is not None:
        verifier_kwargs["required_license"] = required_license
    verifier = ModelVerifier(chain_client, **verifier_kwargs)

    downloader = ModelDownloader(
        config=download_config or DownloadConfig(),
        cache=cache,
        verifier=verifier,
    )

    return ModelDownloadScheduler(
        config=scheduler_config or SchedulerConfig(),
        downloader=downloader,
        chain_client=chain_client,
    )
