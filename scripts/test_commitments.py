#!/usr/bin/env python3
"""
Test script to verify commitment functionality against a local subtensor.

Usage:
    uv run python scripts/test_commitments.py

Requirements:
    - Local subtensor running at ws://68.183.141.180:80
    - A wallet with registered hotkey on the subnet
"""

import asyncio
import json
from pathlib import Path

# Use turbobt directly for testing (Pylon would need to run as a service)
from turbobt.client import Bittensor
from bittensor_wallet import Wallet


# Configuration
SUBTENSOR_ENDPOINT = "ws://68.183.141.180:80"
NETUID = 1  # Change this to your subnet's netuid


async def test_commitments():
    """Test commitment push/pull functionality."""

    print("=" * 60)
    print("Commitment Test Script")
    print("=" * 60)
    print(f"Subtensor: {SUBTENSOR_ENDPOINT}")
    print(f"NetUID: {NETUID}")
    print()

    # Create or load wallet
    # For testing, we'll create a simple wallet
    wallet = Wallet(name="test_validator", hotkey="default")

    print(f"Wallet: {wallet.name}")
    print(f"Hotkey: {wallet.hotkey.ss58_address if wallet.hotkey else 'Not created'}")
    print()

    # Connect to subtensor via turbobt
    async with Bittensor(uri=SUBTENSOR_ENDPOINT, wallet=wallet) as client:
        print("Connected to subtensor!")

        # Get latest block
        latest_block = await client.head.get()
        print(f"Latest block: {latest_block.number}")
        print()

        subnet = client.subnet(NETUID)

        # 1. Test fetching existing commitments
        print("1. Fetching existing commitments...")
        try:
            commitments = await subnet.commitments.fetch()
            if commitments:
                print(f"   Found {len(commitments)} commitments:")
                for hotkey, data in list(commitments.items())[:5]:  # Show first 5
                    print(f"   - {hotkey[:16]}...: {data.hex()[:32]}...")
            else:
                print("   No commitments found on this subnet yet.")
        except Exception as e:
            print(f"   Error fetching commitments: {e}")
        print()

        # 2. Test fetching commitment for our hotkey
        print("2. Fetching commitment for our hotkey...")
        try:
            our_hotkey = wallet.hotkey.ss58_address
            commitment = await subnet.commitments.get(our_hotkey)
            if commitment:
                print(f"   Our commitment: {commitment.hex()}")
            else:
                print("   No commitment set for our hotkey yet.")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # 3. Test setting a commitment (model metadata)
        print("3. Setting a test commitment...")
        try:
            # Create test model metadata
            model_metadata = {
                "model_hash": "sha256:abc123def456",
                "hf_repo": "test-user/real-estate-model",
                "version": "1.0.0",
                "timestamp": 1700000000,
            }
            # Convert to bytes
            commitment_data = json.dumps(model_metadata).encode("utf-8")
            print(f"   Metadata: {model_metadata}")
            print(f"   Bytes length: {len(commitment_data)}")

            # Set commitment on chain
            await subnet.commitments.set(commitment_data)
            print("   Commitment set successfully!")
        except Exception as e:
            print(f"   Error setting commitment: {e}")
            print("   (This might fail if wallet is not registered on subnet)")
        print()

        # 4. Verify the commitment was set
        print("4. Verifying commitment was set...")
        try:
            our_hotkey = wallet.hotkey.ss58_address
            commitment = await subnet.commitments.get(our_hotkey)
            if commitment:
                print(f"   Raw bytes: {commitment.hex()}")
                decoded = json.loads(commitment.decode("utf-8"))
                print(f"   Decoded: {decoded}")
            else:
                print("   Commitment not found (might need to wait for block)")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # 5. List neurons on subnet
        print("5. Listing neurons on subnet...")
        try:
            neurons = await subnet.list_neurons()
            print(f"   Found {len(neurons)} neurons")
            for neuron in neurons[:5]:  # Show first 5
                print(f"   - UID {neuron.uid}: {neuron.hotkey[:16]}...")
        except Exception as e:
            print(f"   Error: {e}")

    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_commitments())
