#!/usr/bin/env python3
"""
RESI Miner CLI - Evaluate and submit ONNX models to the RESI subnet.

Usage:
    miner-cli evaluate ./model.onnx
    miner-cli submit --hf_repo_id user/repo --wallet_name miner --wallet_hotkey default
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import bittensor as bt
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from scripts.compute_hash import compute_hash

from .chain import build_commitment, scan_for_extrinsic_id
from .config import MAX_REPO_BYTES, SCAN_MAX_BLOCKS
from .utils import check_hf_file_exists, validate_model_file

LICENSE_NOTICE = """
License Notice:
Your HuggingFace model repository should be licensed under MIT.
"""


async def evaluate_model(model_path: str) -> int:
    """Evaluate an ONNX model locally with dummy data."""
    print("Evaluating model locally...")

    if not validate_model_file(model_path):
        return 1

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Get model info
        input_info = session.get_inputs()[0]
        bt.logging.info(f"Model input: {input_info.name}, shape={input_info.shape}")

        # Generate dummy input (replace dynamic dims with 1)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_info.shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)

        # Run inference
        start = time.time()
        predictions = session.run(None, {input_info.name: dummy_input})[0]
        inference_ms = (time.time() - start) * 1000

        # Validate output (same checks as validator)
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            bt.logging.error("Model produced NaN/Inf - will fail on validator")
            return 1

        print(
            f"\nEvaluation passed! Inference: {inference_ms:.2f}ms, "
            f"Output shape: {predictions.shape}\n"
        )
        return 0
    except Exception as e:
        bt.logging.error(f"Evaluation failed: {e}")
        return 1


async def submit_model(
    hf_repo_id: str,
    hf_model_filename: str,
    hf_token: str | None,
    wallet: bt.wallet,
    subtensor: bt.subtensor,
    netuid: int,
    extrinsic_scan_blocks: int,
) -> int:
    """Submit a model commitment to the chain."""
    print(LICENSE_NOTICE)
    print("Submitting model to chain...")

    # Validate repo_id fits in chain metadata space
    repo_bytes = len(hf_repo_id.encode("utf-8"))
    if repo_bytes > MAX_REPO_BYTES:
        bt.logging.error(
            f"hf_repo_id too long: {repo_bytes} bytes (max {MAX_REPO_BYTES})"
        )
        return 2

    # Log wallet info
    bt.logging.info(f"Initialized Coldkey: {wallet.name}, Hotkey: {wallet.hotkey}")
    bt.logging.info(f"Hotkey Address: {wallet.hotkey.ss58_address}")
    bt.logging.info(f"Network: {subtensor.network}")

    # Check registration
    if not subtensor.is_hotkey_registered(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=netuid,
        block=None,
    ):
        bt.logging.error(
            f"Hotkey not registered on netuid {netuid}. "
            "Run `btcli subnets register` first"
        )
        return 1

    bt.logging.success(f"Hotkey is registered on netuid {netuid}")

    # Check if file exists in HuggingFace before downloading
    if not check_hf_file_exists(hf_repo_id, hf_model_filename, hf_token):
        return 1

    # Download model from HuggingFace
    try:
        bt.logging.info(f"Downloading {hf_repo_id}/{hf_model_filename}...")
        model_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_model_filename,
            token=hf_token,
        )
        bt.logging.success(f"Successfully downloaded model to {model_path}")
    except Exception as e:
        bt.logging.error(f"Failed to download model: {e}")
        return 1

    # Validate downloaded model (exists + size + ONNX format)
    if not validate_model_file(model_path):
        return 1

    # Compute hash
    try:
        model_hash = compute_hash(Path(model_path))
        bt.logging.info(f"Model hash: {model_hash}")
    except Exception as e:
        bt.logging.error(f"Failed to compute hash: {e}")
        return 1

    # Build commitment
    commitment = build_commitment(
        model_hash=model_hash,
        hf_repo_id=hf_repo_id,
    )
    bt.logging.info(f"Commitment: {commitment}")

    # Record current block
    current_block = subtensor.get_current_block()
    bt.logging.info(f"Current Block: {current_block}")

    # Submit to chain
    try:
        subtensor.commit(wallet, netuid, commitment)
        bt.logging.success("Commitment submitted to chain")
    except Exception as e:
        bt.logging.error(f"Failed to submit commitment: {e}")
        return 1

    # Scan for extrinsic ID
    extrinsic_id = scan_for_extrinsic_id(
        subtensor=subtensor,
        signer_hotkey=wallet.hotkey.ss58_address,
        start_block=current_block,
        max_blocks=extrinsic_scan_blocks,
    )

    # Print extrinsic ID for user
    if extrinsic_id:
        print("SUCCESS! Model committed to chain.")
        print(f"\nExtrinsic ID: {extrinsic_id}")
        print("\nAdd this to your HuggingFace model card:")
        print(f"  Repository: {hf_repo_id}")
        print(f"  Extrinsic:  {extrinsic_id}")
    else:
        bt.logging.warning(
            "Commitment submitted but extrinsic ID not found. "
            "Check a block explorer manually."
        )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments using subcommands."""
    parser = argparse.ArgumentParser(
        description="RESI Miner CLI - Evaluate and submit ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Action to perform"
    )

    # EVALUATE
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an ONNX model locally with dummy data"
    )
    eval_parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        metavar="MODEL_PATH",
        help="Path to ONNX model",
    )
    eval_parser.add_argument(
        "--model_path",
        "--model-path",
        "--model.path",
        "-m",
        dest="model_path_flag",
        default=None,
        help="Path to ONNX model (alternative to positional)",
    )

    # SUBMIT
    submit_parser = subparsers.add_parser(
        "submit", help="Submit a model commitment to the chain"
    )

    # HuggingFace args
    submit_parser.add_argument(
        "--hf_repo_id",
        "--hf-repo-id",
        "--hf.repo_id",
        dest="hf_repo_id",
        required=True,
        help='HuggingFace repo ID, e.g., "user/repo"',
    )
    submit_parser.add_argument(
        "--hf_model_filename",
        "--hf-model-filename",
        "--hf.model_filename",
        dest="hf_model_filename",
        default="model.onnx",
        help="Filename in HF repo (default: model.onnx)",
    )
    submit_parser.add_argument(
        "--hf_token",
        "--hf-token",
        "--hf.token",
        dest="hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)",
    )

    # Chain args
    submit_parser.add_argument(
        "--network",
        "--subtensor.network",
        "--chain",
        "--subtensor.chain_endpoint",
        dest="network",
        default="finney",
        help="Network (finney/test/local) or endpoint URL (ws://...)",
    )
    submit_parser.add_argument(
        "--netuid", type=int, default=46, help="Subnet UID (default: 46)"
    )

    # Wallet args
    submit_parser.add_argument(
        "--wallet_name",
        "--wallet-name",
        "--wallet.name",
        "--name",
        dest="wallet_name",
        required=True,
        help="Bittensor wallet name",
    )
    submit_parser.add_argument(
        "--wallet_hotkey",
        "--wallet-hotkey",
        "--wallet.hotkey",
        "--hotkey",
        dest="wallet_hotkey",
        required=True,
        help="Bittensor wallet hotkey",
    )

    # Scanning args
    submit_parser.add_argument(
        "--extrinsic_scan_blocks",
        "--extrinsic-scan-blocks",
        "--extrinsic.scan_blocks",
        type=int,
        default=SCAN_MAX_BLOCKS,
        help=f"Max blocks to scan for extrinsic (default: {SCAN_MAX_BLOCKS})",
    )

    return parser.parse_args()


async def main() -> int:
    """Async CLI entry point."""
    config = parse_args()

    if config.action == "evaluate":
        model_path = config.model_path or config.model_path_flag
        if not model_path:
            bt.logging.error(
                "Model path required. Usage: evaluate <model.onnx> or "
                "evaluate --model.path <model.onnx>"
            )
            return 2
        return await evaluate_model(model_path)

    elif config.action == "submit":
        wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
        subtensor = bt.subtensor(network=config.network)
        return await submit_model(
            hf_repo_id=config.hf_repo_id,
            hf_model_filename=config.hf_model_filename,
            hf_token=config.hf_token,
            wallet=wallet,
            subtensor=subtensor,
            netuid=config.netuid,
            extrinsic_scan_blocks=config.extrinsic_scan_blocks,
        )

    else:
        print(f"ERROR: Unknown action: {config.action}")
        return 2


def main_sync() -> int:
    """Synchronous CLI entry point for script installation."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(main_sync())
