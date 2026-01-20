"""
Validator configuration management.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any


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
        default=os.environ.get("WALLET_NAME", "default"),
    )

    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        help="Name of the hotkey to use.",
        default=os.environ.get("WALLET_HOTKEY", "default"),
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
        default=int(os.environ.get("VALIDATION_DATA_SCHEDULE_HOUR", "2")),
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
        default=int(os.environ.get("VALIDATION_DATA_MAX_RETRIES", "3")),
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
        default=int(os.environ.get("EPOCH_LENGTH", "360")),
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        help="Score threshold for winner set. Models within this of best are equivalent.",
        default=float(os.environ.get("SCORE_THRESHOLD", "0.005")),
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
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
        default=os.environ.get("LOG_LEVEL", "INFO"),
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
        default=os.environ.get("WANDB_PROJECT", "real-estate-subnet"),
    )

    parser.add_argument(
        "--wandb.entity",
        dest="wandb_entity",
        type=str,
        help="WandB entity.",
        default=os.environ.get("WANDB_ENTITY", ""),
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
        "--model.required_license",
        dest="model_required_license",
        type=str,
        help="Required license text for models.",
        default=os.environ.get("MODEL_REQUIRED_LICENSE", "Lorem Ipsum"),
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


def config_to_dict(config: argparse.Namespace) -> dict[str, Any]:
    """Convert config to dictionary for logging."""
    return {
        "netuid": config.netuid,
        "wallet_name": config.wallet_name,
        "wallet_hotkey": config.wallet_hotkey,
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
        "model_cache_path": str(config.model_cache_path),
        "model_max_size_mb": config.model_max_size_mb,
        "model_required_license": config.model_required_license,
    }


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
