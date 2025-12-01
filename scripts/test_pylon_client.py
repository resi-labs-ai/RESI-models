#!/usr/bin/env python3
"""
Test PylonClient commitment read/write functionality.

Usage:
    uv run python scripts/test_pylon_client.py
    uv run python scripts/test_pylon_client.py --pylon-url http://localhost:8000
    uv run python scripts/test_pylon_client.py --netuid 2 --token my_token

Environment variables (alternative to CLI args):
    PYLON_URL       - Pylon server URL
    PYLON_TOKEN     - Authentication token
    NETUID          - Subnet UID

Requirements:
    - Pylon running and accessible
"""

import argparse
import asyncio
import json
import os
import sys

from real_estate.chain import PylonClient, PylonConfig
from real_estate.chain.errors import (
    AuthenticationError,
    ChainConnectionError,
    CommitmentError,
)

# Defaults
DEFAULT_PYLON_URL = "http://localhost:8000"
DEFAULT_TOKEN = "test_token_123"
DEFAULT_NETUID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test PylonClient commitment read/write functionality"
    )
    parser.add_argument(
        "--pylon-url",
        default=os.environ.get("PYLON_URL", DEFAULT_PYLON_URL),
        help=f"Pylon server URL (default: {DEFAULT_PYLON_URL})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PYLON_TOKEN", DEFAULT_TOKEN),
        help="Pylon authentication token",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.environ.get("NETUID", DEFAULT_NETUID)),
        help=f"Subnet UID (default: {DEFAULT_NETUID})",
    )
    return parser.parse_args()


def header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def step(num: int, desc: str) -> None:
    print()
    print(f"{num}. {desc}")


async def test_commitments(pylon_url: str, token: str, netuid: int) -> bool:
    """Test PylonClient commitment read/write functionality."""

    header("PylonClient Commitment Tests")
    print(f"Pylon: {pylon_url}")
    print(f"NetUID: {netuid}")

    config = PylonConfig(url=pylon_url, token=token)
    client = PylonClient(config)
    all_passed = True

    # 1. Health check
    step(1, "Health check...")
    try:
        healthy = await client.health_check()
        if healthy:
            print("   PASS: Pylon is running")
        else:
            print("   FAIL: Pylon not responding")
            return False
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # 2. Get all commitments
    step(2, "Fetching all commitments...")
    try:
        commitments = await client.get_all_commitments()
        print(f"   PASS: Found {len(commitments)} commitments")
        for hotkey, commitment in list(commitments.items())[:3]:
            print(f"   - {hotkey[:24]}...: {commitment.data[:40]}...")
    except AuthenticationError as e:
        print(f"   FAIL: Auth error: {e}")
        all_passed = False
    except Exception as e:
        print(f"   FAIL: {e}")
        all_passed = False

    # 3. Get commitment for specific hotkey
    step(3, "Fetching commitment for specific hotkey...")
    test_hotkey = None
    try:
        metagraph = await client.get_metagraph(netuid)
        if metagraph.hotkeys:
            test_hotkey = metagraph.hotkeys[0]
            print(f"   Using hotkey: {test_hotkey[:24]}...")
    except Exception as e:
        print(f"   WARN: Could not get metagraph: {e}")

    if test_hotkey:
        try:
            commitment = await client.get_commitment(test_hotkey)
            if commitment:
                print(f"   PASS: Found commitment")
                print(f"   Data: {commitment.data[:50]}...")
            else:
                print(f"   PASS: No commitment (not set yet)")
        except Exception as e:
            print(f"   FAIL: {e}")
            all_passed = False

    # 4. Get model metadata
    step(4, "Fetching model metadata...")
    if test_hotkey:
        try:
            metadata = await client.get_model_metadata(test_hotkey)
            if metadata:
                print(f"   PASS: Got metadata")
                print(f"   Repo: {metadata.repo}")
                print(f"   Hash: {metadata.hash}")
            else:
                print(f"   PASS: No metadata (not set yet)")
        except CommitmentError as e:
            print(f"   WARN: Failed to parse: {e}")
        except Exception as e:
            print(f"   FAIL: {e}")
            all_passed = False

    # 5. Set commitment
    step(5, "Setting a test commitment...")
    commitment_hex = ""
    try:
        model_metadata = {
            "h": "sha256:abc123def456",
            "r": "testuser/resi-model",
            "v": "1.0.0",
        }
        commitment_bytes = json.dumps(model_metadata, separators=(",", ":")).encode("utf-8")
        commitment_hex = commitment_bytes.hex()

        print(f"   Metadata: {model_metadata}")
        print(f"   Size: {len(commitment_bytes)} bytes")

        success = await client.set_commitment(commitment_hex)
        if success:
            print("   PASS: Commitment set")
        else:
            print("   FAIL: set_commitment returned False")
            all_passed = False
    except AuthenticationError as e:
        print(f"   FAIL: Auth error: {e}")
        all_passed = False
    except CommitmentError as e:
        print(f"   FAIL: {e}")
        all_passed = False
    except ChainConnectionError as e:
        print(f"   FAIL: Connection error: {e}")
        all_passed = False

    # 6. Verify commitment was set
    step(6, "Verifying commitment (waiting 30s for chain)...")
    try:
        print("   Waiting 30s for block finalization...")
        await asyncio.sleep(30)
        commitments = await client.get_all_commitments()
        print(f"   Commitments after set: {len(commitments)}")
        found = any(commitment_hex in c.data for c in commitments.values())
        if found:
            print("   PASS: Commitment verified on chain")
        else:
            print("   WARN: Commitment not found (may need more time)")
    except Exception as e:
        print(f"   FAIL: {e}")
        all_passed = False

    # Summary
    header("Summary")
    if all_passed:
        print("All tests PASSED")
        return True
    else:
        print("Some tests FAILED")
        return False


def main() -> None:
    args = parse_args()
    success = asyncio.run(test_commitments(args.pylon_url, args.token, args.netuid))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
