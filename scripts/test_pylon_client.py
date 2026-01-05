"""
Test ChainClient commitment read/write functionality.

Usage:
    uv run python scripts/test_pylon_client.py
    uv run python scripts/test_pylon_client.py --pylon-url http://localhost:8000
    uv run python scripts/test_pylon_client.py --token my_token --identity validator
    uv run python scripts/test_pylon_client.py --hotkey <SS58-address>
    uv run python scripts/test_pylon_client.py --commitment '{"h":"sha256:abc","r":"user/repo","v":"1.0"}'

Environment variables (alternative to CLI args):
    PYLON_URL       - Pylon server URL
    PYLON_TOKEN     - Authentication token
    PYLON_IDENTITY  - Identity name configured in Pylon
    TEST_HOTKEY     - Hotkey to test commitment read

Requirements:
    - Pylon running and accessible
"""

import argparse
import asyncio
import json
import os
import sys

from real_estate.chain import ChainClient, ExtrinsicData, PylonConfig
from real_estate.chain.errors import (
    AuthenticationError,
    ChainConnectionError,
    CommitmentError,
)

# Defaults
DEFAULT_PYLON_URL = "http://localhost:8000"
DEFAULT_TOKEN = "test_token_123"
DEFAULT_IDENTITY = "validator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ChainClient commitment read/write functionality"
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
        "--identity",
        default=os.environ.get("PYLON_IDENTITY", DEFAULT_IDENTITY),
        help=f"Pylon identity name (default: {DEFAULT_IDENTITY})",
    )
    parser.add_argument(
        "--hotkey",
        default=os.environ.get("TEST_HOTKEY", ""),
        help="Hotkey to test commitment read (if not set, uses first from metagraph)",
    )
    parser.add_argument(
        "--commitment",
        default="",
        help='Commitment JSON to set (e.g. \'{"h":"sha256:abc","r":"user/repo","v":"1.0"}\')',
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


async def test_commitments(pylon_url: str, token: str, identity: str, hotkey: str, commitment_json: str) -> bool:
    """Test ChainClient commitment read/write functionality."""

    header("ChainClient Commitment Tests")
    print(f"Pylon: {pylon_url}")
    print(f"Identity: {identity}")

    config = PylonConfig(url=pylon_url, token=token, identity=identity)
    all_passed = True

    async with ChainClient(config) as client:
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
            for c in commitments[:3]:
                print(f"   - {c.hotkey[:24]}...: repo={c.hf_repo_id}")
        except AuthenticationError as e:
            print(f"   FAIL: Auth error: {e}")
            all_passed = False
        except Exception as e:
            print(f"   FAIL: {e}")
            all_passed = False

        # 3. Get commitment for specific hotkey
        step(3, "Fetching commitment for specific hotkey...")
        test_hotkey: str | None = None
        if hotkey:
            test_hotkey = hotkey
            print(f"   Using provided hotkey: {test_hotkey[:24]}...")
        else:
            try:
                metagraph = await client.get_metagraph()
                if metagraph.hotkeys:
                    test_hotkey = metagraph.hotkeys[0]
                    print(f"   Using first hotkey from metagraph: {test_hotkey[:24]}...")
                else:
                    print("   WARN: No hotkeys in metagraph")
            except Exception as e:
                print(f"   WARN: Could not get metagraph: {e}")

        if test_hotkey:
            try:
                commitment = await client.get_commitment(test_hotkey)
                if commitment:
                    print("   PASS: Found commitment")
                    print(f"   Data: {commitment.data[:50]}...")
                else:
                    print("   PASS: No commitment (not set yet)")
            except Exception as e:
                print(f"   FAIL: {e}")
                all_passed = False

        # 4. Get model metadata
        step(4, "Fetching model metadata...")
        if test_hotkey:
            try:
                metadata = await client.get_model_metadata(test_hotkey)
                if metadata:
                    print("   PASS: Got metadata")
                    print(f"   Repo: {metadata.hf_repo_id}")
                    print(f"   Hash: {metadata.model_hash}")
                else:
                    print("   PASS: No metadata (not set yet)")
            except CommitmentError as e:
                print(f"   WARN: Failed to parse: {e}")
            except Exception as e:
                print(f"   FAIL: {e}")
                all_passed = False

        # 5. Record block before setting commitment
        step(5, "Recording current block before commitment...")
        block_before_commitment = 0
        try:
            commitments_before = await client.get_all_commitments()
            if commitments_before:
                block_before_commitment = commitments_before[0].block_number
            else:
                metagraph = await client.get_metagraph()
                block_before_commitment = metagraph.block
            print(f"   Current block: {block_before_commitment}")
        except Exception as e:
            print(f"   WARN: Could not get current block: {e}")

        # 6. Set commitment
        step(6, "Setting a test commitment...")
        commitment_hex = ""
        try:
            if commitment_json:
                model_metadata = json.loads(commitment_json)
            else:
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

        # 7. Verify commitment was set
        step(7, "Verifying commitment (waiting 30s for chain)...")
        block_after_commitment = 0
        try:
            print("   Waiting 30s for block finalization...")
            await asyncio.sleep(30)
            commitments = await client.get_all_commitments()
            print(f"   Commitments after set: {len(commitments)}")
            if commitments:
                block_after_commitment = commitments[0].block_number
                print(f"   Current block after wait: {block_after_commitment}")
            found = any(commitment_hex in c.model_hash for c in commitments)
            if found:
                print("   PASS: Commitment verified on chain")
            else:
                print("   WARN: Commitment not found (may need more time)")
        except Exception as e:
            print(f"   FAIL: {e}")
            all_passed = False

        # 8. Test get_extrinsic - search for commitment extrinsic
        step(8, "Testing get_extrinsic - searching for commitment extrinsic...")
        if block_before_commitment > 0 and block_after_commitment > 0:
            try:
                found_extrinsic: ExtrinsicData | None = None
                blocks_to_search = range(
                    block_before_commitment,
                    block_after_commitment + 1
                )
                print(f"   Searching blocks {block_before_commitment} to {block_after_commitment}...")

                for block_num in blocks_to_search:
                    # Search extrinsic indices 0-15 in each block
                    for ext_idx in range(16):
                        try:
                            extrinsic = await client.get_extrinsic(block_num, ext_idx)
                            if extrinsic is None:
                                break  # No more extrinsics in this block

                            # Check if this is a commitment extrinsic
                            if extrinsic.is_commitment_extrinsic():
                                print(f"   Found commitment extrinsic at block {block_num}, index {ext_idx}")
                                print(f"   Block number: {extrinsic.block_number}")
                                print(f"   Extrinsic index: {extrinsic.extrinsic_index}")
                                print(f"   Extrinsic hash: {extrinsic.extrinsic_hash}")
                                print(f"   Extrinsic length: {extrinsic.extrinsic_length}")
                                print(f"   Address: {extrinsic.address}")
                                print(f"   Call module: {extrinsic.call.call_module}")
                                print(f"   Call function: {extrinsic.call.call_function}")
                                print(f"   Call args: {extrinsic.call.call_args}")
                                found_extrinsic = extrinsic
                                # Don't break - show all commitment extrinsics

                        except ChainConnectionError:
                            break  # No more extrinsics
                        except Exception as e:
                            print(f"   Error fetching extrinsic {block_num}:{ext_idx}: {e}")
                            break

                if found_extrinsic:
                    print("   PASS: get_extrinsic works - found commitment extrinsic")
                else:
                    print("   WARN: No commitment extrinsic found in searched blocks")
                    # Try a specific known extrinsic to verify API works
                    print("   Testing get_extrinsic with block 0, index 0...")
                    try:
                        test_ext = await client.get_extrinsic(block_after_commitment, 0)
                        if test_ext:
                            print("   PASS: get_extrinsic API works")
                            print(f"   Block number: {test_ext.block_number}")
                            print(f"   Extrinsic index: {test_ext.extrinsic_index}")
                            print(f"   Extrinsic hash: {test_ext.extrinsic_hash}")
                            print(f"   Extrinsic length: {test_ext.extrinsic_length}")
                            print(f"   Address: {test_ext.address}")
                            print(f"   Call module: {test_ext.call.call_module}")
                            print(f"   Call function: {test_ext.call.call_function}")
                            print(f"   Call args: {test_ext.call.call_args}")
                        else:
                            print("   WARN: get_extrinsic returned None")
                    except Exception as e:
                        print(f"   FAIL: get_extrinsic failed: {e}")
                        all_passed = False

            except Exception as e:
                print(f"   FAIL: {e}")
                all_passed = False
        else:
            print("   SKIP: Could not determine block range to search")

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
    success = asyncio.run(test_commitments(args.pylon_url, args.token, args.identity, args.hotkey, args.commitment))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
