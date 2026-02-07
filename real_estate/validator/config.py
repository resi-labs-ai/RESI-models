"""
Validator configuration management.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import coloredlogs


def add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add validator arguments to the parser.

    Arguments can be overridden by environment variables.
    """

    parser.add_argument(
        "--netuid",
        type=int,
        help="Subnet netuid to validate on.",
        default=int(os.environ.get("NETUID", "46")),
    )

    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        help="Name of the wallet to use.",
        default=os.environ.get("WALLET_NAME", "validator"),
    )

    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        help="Name of the hotkey to use.",
        default=os.environ.get("WALLET_HOTKEY", "default"),
    )

    parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        type=str,
        help="Path to wallet directory.",
        default=os.environ.get("BITTENSOR_WALLET_PATH", "~/.bittensor/wallets"),
    )

    parser.add_argument(
        "--subtensor.network",
        dest="subtensor_network",
        type=str,
        help="Subtensor network (finney, test, local, or ws:// endpoint).",
        default=os.environ.get("SUBTENSOR_NETWORK", "finney"),
    )

    parser.add_argument(
        "--pylon.url",
        dest="pylon_url",
        type=str,
        help="URL of the Pylon service.",
        default=os.environ.get("PYLON_URL", "http://localhost:8000"),
    )

    parser.add_argument(
        "--validation_data.url",
        dest="validation_data_url",
        type=str,
        help="URL of the validation data API (dashboard).",
        default=os.environ.get("VALIDATION_DATA_URL", "https://dashboard.resilabs.ai"),
    )

    parser.add_argument(
        "--validation_data.schedule_hour",
        dest="validation_data_schedule_hour",
        type=int,
        help="Hour (UTC) for daily validation data fetch (0-23).",
        default=int(os.environ.get("VALIDATION_DATA_SCHEDULE_HOUR", "18")),
    )

    parser.add_argument(
        "--validation_data.schedule_minute",
        dest="validation_data_schedule_minute",
        type=int,
        help="Minute for daily validation data fetch (0-59).",
        default=int(os.environ.get("VALIDATION_DATA_SCHEDULE_MINUTE", "0")),
    )

    parser.add_argument(
        "--validation_data.max_retries",
        dest="validation_data_max_retries",
        type=int,
        help="Max retry attempts for failed validation data fetches.",
        default=int(os.environ.get("VALIDATION_DATA_MAX_RETRIES", "24")),
    )

    parser.add_argument(
        "--validation_data.retry_delay",
        dest="validation_data_retry_delay",
        type=int,
        help="Delay in seconds between validation data retry attempts.",
        default=int(os.environ.get("VALIDATION_DATA_RETRY_DELAY", "300")),
    )

    parser.add_argument(
        "--validation_data.download_raw",
        dest="validation_data_download_raw",
        action="store_true",
        help="Download raw state files for verification.",
        default=os.environ.get("VALIDATION_DATA_DOWNLOAD_RAW", "false").lower()
        == "true",
    )

    parser.add_argument(
        "--pylon.token",
        dest="pylon_token",
        type=str,
        help="Authentication token for Pylon.",
        default=os.environ.get("PYLON_TOKEN", ""),
    )

    parser.add_argument(
        "--pylon.identity",
        dest="pylon_identity",
        type=str,
        help="Identity name configured in Pylon.",
        default=os.environ.get("PYLON_IDENTITY", ""),
    )

    parser.add_argument(
        "--epoch_length",
        type=int,
        help="Number of blocks between metagraph syncs and weight setting.",
        default=int(os.environ.get("EPOCH_LENGTH", "361")),
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        help="Score threshold for winner set. Models within this of best are equivalent.",
        default=float(os.environ.get("SCORE_THRESHOLD", "0.002")),
    )

    parser.add_argument(
        "--disable_set_weights",
        action="store_true",
        help="Disable automatic weight setting.",
        default=os.environ.get("DISABLE_SET_WEIGHTS", "false").lower() == "true",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
        default=os.environ.get("LOG_LEVEL", "DEBUG"),
    )

    parser.add_argument(
        "--wandb.off",
        dest="wandb_off",
        action="store_true",
        help="Disable WandB logging.",
        default=os.environ.get("WANDB_OFF", "false").lower() == "true",
    )

    parser.add_argument(
        "--wandb.project",
        dest="wandb_project",
        type=str,
        help="WandB project name.",
        default=os.environ.get("WANDB_PROJECT", "subnet-46-evaluations-mainnet"),
    )

    parser.add_argument(
        "--wandb.entity",
        dest="wandb_entity",
        type=str,
        help="WandB entity.",
        default=os.environ.get("WANDB_ENTITY", "resi-labs-org"),
    )

    parser.add_argument(
        "--wandb.api_key",
        dest="wandb_api_key",
        type=str,
        help="WandB API key. Can also be set via WANDB_API_KEY env var.",
        default=os.environ.get("WANDB_API_KEY", ""),
    )

    parser.add_argument(
        "--wandb.offline",
        dest="wandb_offline",
        action="store_true",
        help="Run WandB in offline mode (logs saved locally).",
        default=os.environ.get("WANDB_OFFLINE", "false").lower() == "true",
    )

    parser.add_argument(
        "--wandb.log_predictions",
        dest="wandb_log_predictions",
        action="store_true",
        help="Enable logging per-property predictions table to WandB (disabled by default).",
        default=os.environ.get("WANDB_LOG_PREDICTIONS", "false").lower() == "true",
    )

    # Model download settings
    parser.add_argument(
        "--model.cache_path",
        dest="model_cache_path",
        type=str,
        help="Path to cache downloaded models.",
        default=os.environ.get("MODEL_CACHE_PATH", "./model_cache"),
    )

    parser.add_argument(
        "--model.max_size_mb",
        dest="model_max_size_mb",
        type=int,
        help="Maximum model size in MB.",
        default=int(os.environ.get("MODEL_MAX_SIZE_MB", "200")),
    )

    parser.add_argument(
        "--model.min_commitment_age_blocks",
        dest="model_min_commitment_age_blocks",
        type=int,
        help="Minimum age in blocks for commitments to be eligible (~28h = 8400 blocks at 12s/block).",
        default=int(os.environ.get("MODEL_MIN_COMMITMENT_AGE_BLOCKS", "8400")),
    )

    # Docker execution settings
    parser.add_argument(
        "--docker.memory",
        dest="docker_memory",
        type=str,
        help="Docker container memory limit (e.g., '2g', '4g').",
        default=os.environ.get("DOCKER_MEMORY", "2g"),
    )

    parser.add_argument(
        "--docker.cpu",
        dest="docker_cpu",
        type=float,
        help="Docker container CPU limit (1.0 = 1 core).",
        default=float(os.environ.get("DOCKER_CPU", "1.0")),
    )

    parser.add_argument(
        "--docker.timeout",
        dest="docker_timeout",
        type=int,
        help="Docker inference timeout in seconds.",
        default=int(os.environ.get("DOCKER_TIMEOUT", "300")),
    )

    parser.add_argument(
        "--docker.max_concurrent",
        dest="docker_max_concurrent",
        type=int,
        help="Maximum concurrent Docker evaluations.",
        default=int(os.environ.get("DOCKER_MAX_CONCURRENT", "4")),
    )

    # Scheduler settings
    parser.add_argument(
        "--scheduler.pre_download_hours",
        dest="scheduler_pre_download_hours",
        type=float,
        help="Hours before eval to start downloading (default: 3.0).",
        default=float(os.environ.get("SCHEDULER_PRE_DOWNLOAD_HOURS", "3.0")),
    )

    parser.add_argument(
        "--scheduler.catch_up_minutes",
        dest="scheduler_catch_up_minutes",
        type=float,
        help="Minutes before eval reserved for catch-up phase (default: 30.0).",
        default=float(os.environ.get("SCHEDULER_CATCH_UP_MINUTES", "30.0")),
    )

    # Test mode settings
    parser.add_argument(
        "--test-data-path",
        dest="test_data_path",
        type=str,
        help="Path to local JSON file with test validation data. Bypasses API fetch.",
        default=os.environ.get("TEST_DATA_PATH", ""),
    )

    parser.add_argument(
        "--test-mode",
        dest="test_mode",
        action="store_true",
        help="Enable test mode: skip scheduling, run evaluation immediately.",
        default=os.environ.get("TEST_MODE", "false").lower() == "true",
    )

    # Burn settings (emission burning via subnet owner UID)
    parser.add_argument(
        "--burn_amount",
        type=float,
        help="Fraction of emissions to burn (0.0-1.0). Allocated to burn_uid, rest distributed normally.",
        default=float(os.environ.get("BURN_AMOUNT", "0.5")),
    )

    parser.add_argument(
        "--burn_uid",
        type=int,
        help="UID of subnet owner to receive burn allocation (protocol burns this emission).",
        default=int(os.environ.get("BURN_UID", "238")),
    )


def get_config() -> argparse.Namespace:
    """Parse arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Real Estate Subnet Validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    config = parser.parse_args()

    # Convert paths to Path objects
    config.model_cache_path = Path(config.model_cache_path)

    return config


def check_config(config: argparse.Namespace) -> None:
    """
    Validate configuration.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.wallet_name:
        raise ValueError("--wallet.name is required (or set WALLET_NAME env var)")

    if not config.wallet_hotkey:
        raise ValueError("--wallet.hotkey is required (or set WALLET_HOTKEY env var)")

    if not config.pylon_token:
        raise ValueError("--pylon.token is required (or set PYLON_TOKEN env var)")

    if not config.pylon_identity:
        raise ValueError("--pylon.identity is required (or set PYLON_IDENTITY env var)")

    # Validate burn settings
    if config.burn_amount < 0.0 or config.burn_amount > 1.0:
        raise ValueError("--burn_amount must be between 0.0 and 1.0")

    if config.burn_amount > 0.0 and config.burn_uid < 0:
        raise ValueError("--burn_uid is required when --burn_amount > 0")


def config_to_dict(config: argparse.Namespace) -> dict[str, Any]:
    """Convert config to dictionary for logging."""
    return {
        "netuid": config.netuid,
        "wallet_name": config.wallet_name,
        "wallet_hotkey": config.wallet_hotkey,
        "wallet_path": config.wallet_path,
        "subtensor_network": config.subtensor_network,
        "pylon_url": config.pylon_url,
        "pylon_token": "***" if config.pylon_token else "",
        "pylon_identity": config.pylon_identity,
        "validation_data_url": config.validation_data_url,
        "validation_data_schedule_hour": config.validation_data_schedule_hour,
        "validation_data_schedule_minute": config.validation_data_schedule_minute,
        "validation_data_max_retries": config.validation_data_max_retries,
        "validation_data_retry_delay": config.validation_data_retry_delay,
        "validation_data_download_raw": config.validation_data_download_raw,
        "epoch_length": config.epoch_length,
        "score_threshold": config.score_threshold,
        "disable_set_weights": config.disable_set_weights,
        "log_level": config.log_level,
        "wandb_off": config.wandb_off,
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "wandb_api_key": "***" if config.wandb_api_key else "",
        "wandb_offline": config.wandb_offline,
        "wandb_log_predictions": config.wandb_log_predictions,
        "model_cache_path": str(config.model_cache_path),
        "model_max_size_mb": config.model_max_size_mb,
        "model_min_commitment_age_blocks": config.model_min_commitment_age_blocks,
        "docker_memory": config.docker_memory,
        "docker_cpu": config.docker_cpu,
        "docker_timeout": config.docker_timeout,
        "docker_max_concurrent": config.docker_max_concurrent,
        "scheduler_pre_download_hours": config.scheduler_pre_download_hours,
        "scheduler_catch_up_minutes": config.scheduler_catch_up_minutes,
        "test_data_path": config.test_data_path,
        "test_mode": config.test_mode,
        "burn_amount": config.burn_amount,
        "burn_uid": config.burn_uid,
    }


def setup_logging(level: str) -> None:
    """
    Configure logging with colored output.

    Levels:
    - TRACE: Everything including third-party debug (websockets, httpcore, etc.)
    - DEBUG: Only real_estate.* debug logs, third-party at WARNING
    - INFO/WARNING/ERROR: Standard behavior
    """
    # Add custom TRACE level (below DEBUG)
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    if level.upper() == "TRACE":
        # TRACE = show everything
        coloredlogs.install(
            level=TRACE,
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        coloredlogs.install(
            level=getattr(logging, level),
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        # At DEBUG level, quiet noisy third-party loggers
        if level.upper() == "DEBUG":
            noisy_loggers = [
                "websockets",
                "httpcore",
                "httpx",
                "docker",
                "urllib3",
                "asyncio",
                "filelock",
            ]
            for name in noisy_loggers:
                logging.getLogger(name).setLevel(logging.WARNING)
