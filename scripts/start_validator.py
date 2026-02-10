#!/usr/bin/env python3
"""
Auto-updating validator script for RESI subnet.

Based on: https://github.com/macrocosm-os/pretraining/blob/main/scripts/start_validator.py

This script runs the validator and automatically updates it when a new version is released.
Auto-updates check git every 5 minutes, pull changes, and restart if needed.

Prerequisites:
  - PM2 installed: npm install -g pm2
  - Pylon running: docker compose up -d

Usage:
    # Option 1: Use .env file (recommended)
    cp .env.example .env && edit .env
    set -a && source .env && set +a
    python scripts/start_validator.py

    # Option 2: Pass args directly
    python scripts/start_validator.py \\
        --pylon.token $PYLON_TOKEN \\
        --pylon.identity $PYLON_IDENTITY
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from shlex import split

log = logging.getLogger(__name__)

# Configuration
UPDATES_CHECK_TIME = timedelta(minutes=5)
PROJECT_ROOT = Path(__file__).parent.parent
ECOSYSTEM_CONFIG_PATH = PROJECT_ROOT / "config" / "ecosystem.config.js"
VALIDATOR_MODULE = "real_estate.validator.validator"


def check_pm2_installed() -> None:
    """Check if PM2 is installed."""
    try:
        subprocess.run(
            ["pm2", "--version"],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        log.error("PM2 is not installed. Install with: npm install -g pm2")
        sys.exit(1)


def get_version() -> str:
    """Get current git commit hash (short)."""
    result = subprocess.run(
        split("git rev-parse HEAD"),
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f"Invalid commit hash: {commit}"
    return commit[:8]


def pull_latest_version() -> None:
    """
    Pull latest version from git.

    Uses `git pull --rebase --autostash` to preserve local changes.
    Aborts rebase on conflict.
    """
    try:
        subprocess.run(
            split("git pull --rebase --autostash"),
            check=True,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to pull, reverting: %s", exc)
        subprocess.run(
            split("git rebase --abort"),
            check=False, # OK to fail if no rebase to abort  
            cwd=PROJECT_ROOT,
        )


def upgrade_packages() -> None:
    """Upgrade packages using uv sync."""
    log.info("Upgrading packages with uv sync...")
    try:
        subprocess.run(split("uv sync"), check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        log.error("Failed to upgrade packages: %s", exc)


def generate_pm2_config(pm2_name: str, args: list[str]) -> None:
    """Generate PM2 ecosystem config file."""
    ECOSYSTEM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    validator_args = " ".join(args)
    config_content = f"""
module.exports = {{
    apps: [{{
        name: '{pm2_name}',
        script: '{sys.executable}',
        args: '-m {VALIDATOR_MODULE} {validator_args}',
        cwd: '{PROJECT_ROOT}',
        autorestart: true,
        restart_delay: 30000,
        max_restarts: 100,
        env: {{
            PYTHONPATH: '{os.environ.get("PYTHONPATH", "")}:{PROJECT_ROOT}',
        }},
    }}]
}};
"""
    ECOSYSTEM_CONFIG_PATH.write_text(config_content)
    log.info("Generated PM2 config: %s", ECOSYSTEM_CONFIG_PATH)


def start_validator(pm2_name: str, args: list[str]) -> None:
    """Start validator using PM2."""
    generate_pm2_config(pm2_name, args)
    log.info("Starting validator with PM2: %s", pm2_name)
    subprocess.run(
        ["pm2", "start", str(ECOSYSTEM_CONFIG_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def stop_validator(pm2_name: str) -> None:
    """Stop validator PM2 process."""
    subprocess.run(
        ["pm2", "delete", pm2_name],
        cwd=PROJECT_ROOT,
        check=False,  # Don't fail if not running
    )


# Mapping from argparse dest to CLI arg name
ARG_NAME_MAP = {
    "wallet_name": "wallet.name",
    "wallet_hotkey": "wallet.hotkey",
    "subtensor_network": "subtensor.network",
    "netuid": "netuid",
    "pylon_url": "pylon.url",
    "pylon_token": "pylon.token",
    "pylon_identity": "pylon.identity",
    "log_level": "log_level",
    "wandb_api_key": "wandb.api_key",
    "wandb_project": "wandb.project",
    "wandb_entity": "wandb.entity",
    "burn_amount": "burn_amount",
    "burn_uid": "burn_uid",
}


def build_args_list(
    args_namespace: argparse.Namespace, extra_args: list[str]
) -> list[str]:
    """Convert parsed args to command-line format."""
    args_list = []
    skip_keys = {"pm2_name"}

    for key, value in vars(args_namespace).items():
        if key in skip_keys:
            continue
        if value is not None and value != "":
            arg_key = ARG_NAME_MAP.get(key, key)
            args_list.append(f"--{arg_key}")
            if not isinstance(value, bool):
                args_list.append(str(value))

    args_list.extend(extra_args)
    return args_list


def main(pm2_name: str, args_namespace: argparse.Namespace, extra_args: list[str]) -> None:
    """
    Main loop: run validator and auto-update on new versions.

    Checks for updates every UPDATES_CHECK_TIME. If version changed,
    upgrades packages and restarts the validator.
    """
    check_pm2_installed()
    args_list = build_args_list(args_namespace, extra_args)

    start_validator(pm2_name, args_list)
    current_version = get_version()
    log.info("Started validator, version: %s", current_version)

    try:
        while True:
            time.sleep(UPDATES_CHECK_TIME.total_seconds())

            pull_latest_version()
            latest_version = get_version()

            if latest_version != current_version:
                log.info("Update available: %s -> %s", current_version, latest_version)
                upgrade_packages()
                stop_validator(pm2_name)
                start_validator(pm2_name, args_list)
                current_version = latest_version
                log.info("Validator restarted, version: %s", current_version)
            else:
                log.debug("No update, current: %s", current_version)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        stop_validator(pm2_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Auto-updating RESI validator runner.",
        epilog=(
            "Example:\n"
            "  python scripts/start_validator.py \\\n"
            "    --pylon.token $PYLON_TOKEN \\\n"
            "    --pylon.identity $PYLON_IDENTITY"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Script-specific args
    parser.add_argument(
        "--pm2_name",
        default="resi_validator",
        help="PM2 process name (default: resi_validator)",
    )

    # Validator args (forwarded to validator)
    # Defaults read from environment variables, then fall back to hardcoded defaults
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        default=os.environ.get("WALLET_NAME", "validator"),
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        default=os.environ.get("WALLET_HOTKEY", "default"),
    )
    parser.add_argument(
        "--subtensor.network",
        dest="subtensor_network",
        default=os.environ.get("SUBTENSOR_NETWORK", "finney"),
    )
    parser.add_argument(
        "--netuid",
        default=os.environ.get("NETUID", "46"),
    )
    parser.add_argument(
        "--pylon.url",
        dest="pylon_url",
        default=os.environ.get("PYLON_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--pylon.token",
        dest="pylon_token",
        default=os.environ.get("PYLON_TOKEN", ""),
    )
    parser.add_argument(
        "--pylon.identity",
        dest="pylon_identity",
        default=os.environ.get("PYLON_IDENTITY", ""),
    )
    parser.add_argument(
        "--log_level",
        default=os.environ.get("LOG_LEVEL", "DEBUG"),
    )
    parser.add_argument(
        "--wandb.api_key",
        dest="wandb_api_key",
        default=os.environ.get("WANDB_API_KEY", ""),
    )
    parser.add_argument(
        "--wandb.project",
        dest="wandb_project",
        default=os.environ.get("WANDB_PROJECT", "subnet-46-evaluations-mainnet"),
    )
    parser.add_argument(
        "--wandb.entity",
        dest="wandb_entity",
        default=os.environ.get("WANDB_ENTITY", "resi-labs"),
    )
    parser.add_argument(
        "--burn_amount",
        dest="burn_amount",
        type=float,
        default=float(os.environ.get("BURN_AMOUNT", "0.5")),
    )
    parser.add_argument(
        "--burn_uid",
        dest="burn_uid",
        type=int,
        default=int(os.environ.get("BURN_UID", "238")),
    )

    flags, extra_args = parser.parse_known_args()
    main(flags.pm2_name, flags, extra_args)
