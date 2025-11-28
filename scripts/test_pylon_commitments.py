#!/usr/bin/env python3
"""
Test script to verify Pylon commitment endpoints.

Usage:
    1. Start Pylon locally:
       cd /path/to/bittensor-pylon && ./run_local.sh

    2. Run this test:
       uv run python scripts/test_pylon_commitments.py

Requirements:
    - Pylon running at http://localhost:8000
    - Subtensor running (localnet or testnet)
"""

import asyncio
import json
import httpx

# Configuration
PYLON_URL = "http://localhost:8000"
AUTH_TOKEN = "test_token_123"  # Must match PYLON_ID_LOCALTEST_BITTENSOR_TOKEN


async def test_pylon_commitments():
    """Test Pylon commitment endpoints via HTTP."""

    print("=" * 60)
    print("Pylon Commitment Endpoint Tests")
    print("=" * 60)
    print(f"Pylon URL: {PYLON_URL}")
    print()

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(base_url=PYLON_URL, headers=headers, timeout=30.0) as client:

        # 0. Check if Pylon is running
        print("0. Checking Pylon health...")
        try:
            response = await client.get("/schema/swagger")
            if response.status_code == 200:
                print("   Pylon is running!")
            else:
                print(f"   Warning: Got status {response.status_code}")
        except httpx.ConnectError:
            print("   ERROR: Cannot connect to Pylon!")
            print("   Make sure Pylon is running: cd bittensor-pylon && ./run_local.sh")
            return
        print()

        # 1. Get all commitments
        print("1. GET /api/v1/commitments - Fetch all commitments...")
        try:
            response = await client.get("/api/v1/commitments")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                commitments = data.get("commitments", {})
                print(f"   Found {len(commitments)} commitments")
                for hotkey, commitment_hex in list(commitments.items())[:3]:
                    print(f"   - {hotkey[:20]}...: {commitment_hex[:40]}...")
            else:
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # 2. Get commitment for a specific hotkey
        print("2. GET /api/v1/commitments/{hotkey} - Fetch specific commitment...")
        test_hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # Example hotkey
        try:
            response = await client.get(f"/api/v1/commitments/{test_hotkey}")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Hotkey: {data.get('hotkey', 'N/A')}")
                print(f"   Data: {data.get('data', 'N/A')}")
            elif response.status_code == 404:
                print("   No commitment found for this hotkey (expected if not set)")
            else:
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # 3. Set a commitment
        print("3. POST /api/v1/commitments - Set a commitment...")
        try:
            # Create compact model metadata (must be <= 128 bytes for Substrate Data type)
            # Using short keys to fit within limit
            model_metadata = {
                "h": "abc123def456",  # model hash (shortened)
                "r": "user/model-v1",  # hf repo (shortened)
                "v": "1.0.0",
                "t": 1700000000,
            }
            # Convert to hex string for the API
            commitment_bytes = json.dumps(model_metadata, separators=(",", ":")).encode("utf-8")
            commitment_hex = commitment_bytes.hex()

            print(f"   Metadata: {model_metadata}")
            print(f"   Bytes: {len(commitment_bytes)}")
            print(f"   Hex: {commitment_hex}")

            response = await client.post(
                "/api/v1/commitments",
                json={"data": commitment_hex}
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")

            if response.status_code == 201:
                print("   SUCCESS: Commitment set!")
            elif response.status_code == 401:
                print("   Auth error - check your token")
            elif response.status_code == 502:
                print("   Backend error - subtensor might not be ready")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # 4. Get neurons (verify Pylon is connected to chain)
        print("4. GET /api/v1/neurons/latest - Verify chain connection...")
        try:
            response = await client.get("/api/v1/neurons/latest")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                block = data.get("block", {})
                neurons = data.get("neurons", {})
                print(f"   Block: {block.get('number', 'N/A')}")
                print(f"   Neurons: {len(neurons)}")
            else:
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   Error: {e}")

    print()
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)


async def test_pylon_client_library():
    """Test using Pylon's Python client library."""

    print()
    print("=" * 60)
    print("Testing Pylon Python Client Library")
    print("=" * 60)

    try:
        from pylon.v1 import (
            AsyncPylonClient,
            AsyncPylonClientConfig,
            GetLatestNeuronsRequest,
        )
        from pylon._internal.common.requests import (
            GetCommitmentsRequest,
            SetCommitmentRequest,
        )

        config = AsyncPylonClientConfig(
            address=PYLON_URL,
            auth_token=AUTH_TOKEN,
        )

        async with AsyncPylonClient(config) as client:
            print("1. Getting latest neurons via client...")
            try:
                response = await client.request(GetLatestNeuronsRequest())
                print(f"   Block: {response.block.number}")
                print(f"   Neurons: {len(response.neurons)}")
            except Exception as e:
                print(f"   Error: {e}")

            print()
            print("2. Getting commitments via client...")
            try:
                response = await client.request(GetCommitmentsRequest())
                print(f"   Commitments: {len(response.commitments)}")
            except Exception as e:
                print(f"   Error: {e}")

    except ImportError as e:
        print(f"Could not import Pylon client: {e}")
        print("Make sure pylon is installed: pip install -e /path/to/bittensor-pylon")


if __name__ == "__main__":
    asyncio.run(test_pylon_commitments())
    # Uncomment to also test the client library:
    # asyncio.run(test_pylon_client_library())
