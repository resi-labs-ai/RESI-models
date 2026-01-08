#!/usr/bin/env python3
"""
RESI Miner CLI - Evaluate and submit ONNX models to the RESI subnet.

Usage:
    python neurons/test_miner_cli.py evaluate --model_path ./model.onnx
    python neurons/test_miner_cli.py submit --hf_repo_id user/repo --wallet_name miner --wallet_hotkey default
"""

# Standard library
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

# Third-party
import bittensor as bt
import huggingface_hub
import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Local imports
from scripts.compute_hash import compute_hash

# Constants
SCAN_MAX_BLOCKS = 20
SCAN_MAX_EXTRINSICS_PER_BLOCK = 100
MAX_MODEL_SIZE_MB = 200
MAX_COMMITMENT_BYTES = 128  # Chain metadata limit
HASH_LENGTH = 64  # SHA-256 hash
MAX_REPO_BYTES = 51
REQUIRED_ONNX_VERSION = "1.20.0"
REQUIRED_ONNXRUNTIME_VERSION = "1.20.1"

LICENSE_NOTICE = """
License Notice:
Your HuggingFace model repository should be licensed under MIT.
"""


def build_commitment(model_hash: str, hf_repo_id: str) -> str:
    """
    Build RESI commitment as compact JSON string.

    Args:
        model_hash: 64-char SHA-256 hash
        hf_repo_id: HuggingFace repo ID (e.g., "user/repo")

    Returns:
        Compact JSON string: {"h":"...","r":"..."}
    """
    return json.dumps({"h": model_hash, "r": hf_repo_id}, separators=(",", ":"))


def scan_for_extrinsic_id(
    subtensor: bt.subtensor,
    signer_hotkey: str,
    start_block: int,
    max_blocks: int = SCAN_MAX_BLOCKS,
    max_per_block: int = SCAN_MAX_EXTRINSICS_PER_BLOCK,
) -> str | None:
    """
    Scan blocks for a commitment extrinsic matching the signer.

    Args:
        subtensor: Bittensor subtensor instance
        signer_hotkey: SS58 address of the signer to match
        start_block: Block number to start scanning from
        max_blocks: Maximum number of blocks to scan forward
        max_per_block: Maximum extrinsic indices to check per block

    Returns:
        Extrinsic ID as "{block}-{index}" or None if not found
    """
    bt.logging.info(f"Scanning {max_blocks} blocks starting from {start_block}...")

    for block_offset in range(max_blocks):
        block_num = start_block + block_offset

        try:
            block_hash = subtensor.get_block_hash(block_num)
            if not block_hash:
                continue

            block = subtensor.substrate.get_block(block_hash)
            if not block or "extrinsics" not in block:
                continue

            extrinsics = block["extrinsics"]
            for idx, extrinsic in enumerate(extrinsics[:max_per_block]):
                try:
                    if not hasattr(extrinsic, "value") or not extrinsic.value:
                        continue

                    ext_value = extrinsic.value

                    if not extrinsic.signed:
                        continue

                    call = ext_value.get("call")
                    if not call:
                        continue

                    call_module = call.get("call_module", "")
                    call_function = call.get("call_function", "")

                    if (
                        call_module == "Commitments"
                        and call_function == "set_commitment"
                    ):
                        extrinsic_address = ext_value.get("address")
                        if extrinsic_address == signer_hotkey:
                            bt.logging.success(
                                f"Found commitment at block {block_num}, extrinsic {idx}"
                            )
                            return f"{block_num}-{idx}"

                except Exception as e:
                    bt.logging.debug(
                        f"Error processing extrinsic {idx} in block {block_num}: {e}"
                    )
                    continue

        except Exception as e:
            if "State discarded" in str(e) or "StateDiscardedError" in str(
                type(e).__name__
            ):
                bt.logging.warning(f"Block {block_num} too old (state discarded)")
                break
            else:
                bt.logging.warning(f"Error scanning block {block_num}: {e}")
                continue

    bt.logging.error(f"Commitment not found in {max_blocks} blocks")
    return None


def check_onnx_versions() -> bool:
    """
    Verify onnx and onnxruntime versions match validator requirements.
    Returns True if versions match, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        bt.logging.error(
            "onnxruntime not installed. Run: pip install onnxruntime==1.20.1"
        )
        return False

    onnx_version = onnx.__version__
    ort_version = ort.__version__

    errors = []

    if onnx_version != REQUIRED_ONNX_VERSION:
        errors.append(f"onnx=={onnx_version} (required: {REQUIRED_ONNX_VERSION})")

    if ort_version != REQUIRED_ONNXRUNTIME_VERSION:
        errors.append(
            f"onnxruntime=={ort_version} (required: {REQUIRED_ONNXRUNTIME_VERSION})"
        )

    if errors:
        bt.logging.error(
            f"Version mismatch - your model may fail on validator!\n"
            f"  Installed: {', '.join(errors)}\n"
            f"  Fix with: pip install onnx=={REQUIRED_ONNX_VERSION} onnxruntime=={REQUIRED_ONNXRUNTIME_VERSION}"
        )
        return False

    bt.logging.info(
        f"ONNX versions match validator (onnx=={onnx_version}, onnxruntime=={ort_version})"
    )
    return True


def validate_model_file(model_path: str) -> bool:
    """
    Validate a local ONNX model file.
    Checks existence, size limit, and ONNX format.
    """
    # Check ONNX versions match validator
    if not check_onnx_versions():
        return False

    if not os.path.exists(model_path):
        bt.logging.error(f"Model file does not exist: {model_path}")
        return False

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if size_mb > MAX_MODEL_SIZE_MB:
        bt.logging.error(
            f"Model too large: {size_mb:.2f}MB exceeds {MAX_MODEL_SIZE_MB}MB limit"
        )
        return False

    bt.logging.info(f"Model size: {size_mb:.2f}MB")

    try:
        onnx.checker.check_model(model_path)
        return True
    except onnx.checker.ValidationError as e:
        bt.logging.error(f"Invalid ONNX format: {e}")
        return False
    except Exception as e:
        bt.logging.error(f"Error reading ONNX model: {e}")
        return False


def check_hf_file_exists(repo_id: str, filename: str, token: str | None = None) -> bool:
    """Check if file exists in HuggingFace repo."""
    try:
        if not huggingface_hub.file_exists(
            repo_id=repo_id, filename=filename, repo_type="model", token=token
        ):
            bt.logging.error(
                f"File '{filename}' not found in repo '{repo_id}'. "
                f"Use --hf_model_filename if your model has a different name."
            )
            return False
        return True
    except huggingface_hub.utils.RepositoryNotFoundError:
        bt.logging.error(
            f"Repository '{repo_id}' not found or private.\n"
            f"  If private, provide --hf_token or set HF_TOKEN env var."
        )
        return False
    except Exception as e:
        bt.logging.error(f"Error checking HuggingFace repo: {e}")
        return False


async def evaluate_model(config: argparse.Namespace) -> int:
    """
    Evaluate an ONNX model locally with dummy data.

    Returns:
        Exit code (0 = success, 1 = error, 2 = invalid args)
    """
    print("🔍 Evaluating model locally...")

    # Support both positional and flag
    model_path = config.model_path or config.model_path_flag

    if not model_path:
        bt.logging.error(
            "Model path required. Usage: evaluate <model.onnx> or evaluate --model.path <model.onnx>"
        )
        return 2

    # Validate model file
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
            f"\n✅ Evaluation passed! Inference: {inference_ms:.2f}ms, Output shape: {predictions.shape}\n"
        )
        return 0
    except Exception as e:
        bt.logging.error(f"Evaluation failed: {e}")
        return 1


class MinerCLI:
    """
    MinerCLI class for evaluating and submitting miner models.
    """

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
        self.subtensor = bt.subtensor(network=config.network)

    async def submit_model(self) -> int:
        """
        Submit a model commitment to the chain.

        Returns:
            Exit code (0 = success, 1 = error, 2 = invalid args)
        """
        print(LICENSE_NOTICE)
        print("📤 Submitting model to chain...")
        if not self.config.hf_repo_id:
            bt.logging.error("--hf_repo_id is required.")
            return 2

        # Validate repo_id fits in chain metadata space
        repo_bytes = len(self.config.hf_repo_id.encode("utf-8"))
        if repo_bytes > MAX_REPO_BYTES:
            bt.logging.error(
                f"hf_repo_id too long: {repo_bytes} bytes (max {MAX_REPO_BYTES})"
            )
            return 2

        # Validate wallet info/ Load wallet and subtensor
        bt.logging.info(
            f"Initialized Coldkey: {self.wallet.name}, Hotkey: {self.wallet.hotkey}"
        )
        bt.logging.info(f"Hotkey Address: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"Network: {self.config.network}")

        # Check registration
        if not self.subtensor.is_hotkey_registered(
            hotkey_ss58=self.wallet.hotkey.ss58_address,
            netuid=self.config.netuid,
            block=None,
        ):
            bt.logging.error(
                f"Hotkey not registered on netuid {self.config.netuid}. Run `btcli subnets register` first"
            )
            return 1

        bt.logging.success(f"Hotkey is registered on netuid {self.config.netuid}")

        # Check if file exists in HuggingFace before downloading
        if not check_hf_file_exists(
            self.config.hf_repo_id, self.config.hf_model_filename, self.config.hf_token
        ):
            return 1

        # Download model from HuggingFace
        try:
            bt.logging.info(
                f"Downloading {self.config.hf_repo_id}/{self.config.hf_model_filename}..."
            )
            model_path = hf_hub_download(
                repo_id=self.config.hf_repo_id,
                filename=self.config.hf_model_filename,
                token=self.config.hf_token,
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
            hf_repo_id=self.config.hf_repo_id,
        )
        bt.logging.info(f"Commitment: {commitment}")

        # Record current block
        current_block = self.subtensor.get_current_block()
        bt.logging.info(f"Current Block: {current_block}")

        # Submit to chain
        try:
            self.subtensor.commit(self.wallet, self.config.netuid, commitment)
            bt.logging.success("Commitment submitted to chain")
        except Exception as e:
            bt.logging.error(f"Failed to submit commitment: {e}")
            return 1

        # Scan for extrinsic ID
        extrinsic_id = scan_for_extrinsic_id(
            subtensor=self.subtensor,
            signer_hotkey=self.wallet.hotkey.ss58_address,
            start_block=current_block,
            max_blocks=self.config.extrinsic_scan_blocks,
        )

        # Print extrinsic ID for user
        if extrinsic_id:
            print("✅ SUCCESS! Model committed to chain.")
            print(f"\nExtrinsic ID: {extrinsic_id}")
            print("\nAdd this to your HuggingFace model card:")
            print(f"  Repository: {self.config.hf_repo_id}")
            print(f"  Extrinsic:  {extrinsic_id}")
        else:
            bt.logging.warning(
                "Commitment submitted but extrinsic ID not found. "
                "Check a block explorer manually."
            )
        return 0


# MAIN & ARGUMENT PARSING
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
        "model_path",  # positional (no dashes)
        nargs="?",  # optional positional
        default=None,
        metavar="MODEL_PATH",
        help="Path to ONNX model",
    )
    eval_parser.add_argument(
        "--model_path",
        "--model-path",
        "--model.path",
        "-m",
        dest="model_path_flag",  # different dest!
        default=None,  # not required
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
        default=None,
        required=True,
        help="Bittensor wallet name",
    )
    submit_parser.add_argument(
        "--wallet_hotkey",
        "--wallet-hotkey",
        "--wallet.hotkey",
        "--hotkey",
        dest="wallet_hotkey",
        default=None,
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


def main_sync() -> int:
    """Synchronous CLI entry point for script installation."""
    return asyncio.run(main())


async def main() -> int:
    config = parse_args()
    if config.action == "evaluate":
        return await evaluate_model(config)
    elif config.action == "submit":
        cli = MinerCLI(config)
        return await cli.submit_model()
    else:
        print(f"ERROR: Unknown action: {config.action}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)
