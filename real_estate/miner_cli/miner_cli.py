#!/usr/bin/env python3
"""
RESI Miner CLI - Evaluate and submit ONNX models to the RESI subnet.

Usage:
    miner-cli evaluate --model.path ./model.onnx
    miner-cli submit --hf.repo_id user/repo --wallet.name miner --wallet.hotkey default
"""

import argparse
from pathlib import Path

import bittensor as bt
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from scripts.compute_hash import compute_hash

from .chain import build_commitment, scan_for_extrinsic_id
from .config import (
    EXPECTED_NUM_FEATURES,
    MAX_REPO_BYTES,
    NETWORK_NETUIDS,
    SCAN_MAX_BLOCKS,
    TEST_SAMPLES,
)
from .errors import (
    HashComputationError,
    HotkeyNotRegisteredError,
    InferenceError,
    InvalidPredictionError,
    MinerCLIError,
    ModelDownloadError,
    ModelInterfaceError,
)
from .utils import check_hf_file_exists, validate_model_file

LICENSE_NOTICE = """
License Notice:
Your HuggingFace model repository should be licensed under MIT.
"""


def validate_model_interface(session: ort.InferenceSession) -> tuple[str, int]:
    """
    Validate model input/output shapes match validator expectations.

    Args:
        session: ONNX runtime inference session.

    Returns:
        Tuple of (input_name, num_features).

    Raises:
        ModelInterfaceError: If interface doesn't match expected format.
    """
    # Check inputs
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ModelInterfaceError(
            f"Model has {len(inputs)} inputs, expected 1. "
            "Model should have a single input for features."
        )

    input_info = inputs[0]
    input_shape = input_info.shape

    # Expect shape like (batch, num_features) or (None, num_features)
    if len(input_shape) != 2:
        raise ModelInterfaceError(
            f"Input shape {input_shape} invalid. "
            f"Expected 2D shape (batch, {EXPECTED_NUM_FEATURES})."
        )

    # Check feature dimension
    feature_dim = input_shape[1]
    if isinstance(feature_dim, int) and feature_dim != EXPECTED_NUM_FEATURES:
        raise ModelInterfaceError(
            f"Model expects {feature_dim} features, validator expects {EXPECTED_NUM_FEATURES}. "
            "Ensure your model was trained with the correct feature set."
        )

    # Check outputs
    outputs = session.get_outputs()
    if len(outputs) != 1:
        raise ModelInterfaceError(
            f"Model has {len(outputs)} outputs, expected 1. "
            "Model should output a single price prediction."
        )

    output_shape = outputs[0].shape
    # Accept (batch,), (batch, 1), or dynamic shapes
    if len(output_shape) == 2:
        out_dim = output_shape[1]
        if isinstance(out_dim, int) and out_dim != 1:
            raise ModelInterfaceError(
                f"Output shape {output_shape} invalid. "
                "Expected (batch,) or (batch, 1) for price predictions."
            )

    return input_info.name, EXPECTED_NUM_FEATURES


def load_test_samples() -> tuple[np.ndarray, np.ndarray]:
    """
    Load embedded test samples.

    Returns:
        Tuple of (features, actual_prices) as numpy arrays.
    """
    features = np.array(
        [sample["features"] for sample in TEST_SAMPLES], dtype=np.float32
    )
    actual_prices = np.array(
        [sample["actual_price"] for sample in TEST_SAMPLES], dtype=np.float32
    )
    return features, actual_prices


def run_inference(
    session: ort.InferenceSession, input_name: str, features: np.ndarray
) -> np.ndarray:
    """
    Run inference on the model.

    Args:
        session: ONNX runtime inference session.
        input_name: Name of the model's input.
        features: Input features array of shape (batch, num_features).

    Returns:
        Predictions array of shape (batch,).

    Raises:
        InferenceError: If inference fails.
    """
    try:
        outputs = session.run(None, {input_name: features})
        predictions = outputs[0]
        # Flatten to 1D if needed (handles both (batch,) and (batch, 1))
        return predictions.flatten()  # type: ignore[no-any-return]
    except Exception as e:
        raise InferenceError(f"Inference failed: {e}") from e


def validate_predictions(predictions: np.ndarray) -> None:
    """
    Validate predictions are finite values.

    Args:
        predictions: Model predictions array.

    Raises:
        InvalidPredictionError: If predictions contain NaN or Inf.
    """
    if np.any(np.isnan(predictions)):
        raise InvalidPredictionError(
            "Model produced NaN values. Check your model for numerical instability."
        )
    if np.any(np.isinf(predictions)):
        raise InvalidPredictionError(
            "Model produced Inf values. Check your model for overflow issues."
        )
    if np.any(predictions <= 0):
        bt.logging.warning(
            "Model produced non-positive price predictions. "
            "This may indicate an issue with the model."
        )


def calculate_mape(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values.
        actual: Actual values.

    Returns:
        MAPE as a decimal (e.g., 0.05 for 5%).
    """
    return float(np.mean(np.abs((actual - predictions) / actual)))


def display_results(predictions: np.ndarray, actual_prices: np.ndarray) -> None:
    """
    Calculate metrics and display results.

    Args:
        predictions: Model predictions.
        actual_prices: Actual prices from test samples.
    """
    bt.logging.info("")
    bt.logging.info("Test inference results:")

    # Per-sample results
    for i, (pred, actual) in enumerate(zip(predictions, actual_prices)):
        error_pct = abs(actual - pred) / actual * 100
        bt.logging.info(
            f"  Sample {i + 1}: Predicted ${pred:,.0f} vs actual ${actual:,.0f} "
            f"({error_pct:.2f}% error)"
        )

    # Calculate MAPE and score
    mape = calculate_mape(predictions, actual_prices)
    score = 1.0 - mape

    bt.logging.info("")
    bt.logging.info("Metrics:")
    bt.logging.info(f"  MAPE:  {mape * 100:.2f}%")
    bt.logging.info(f"  Score: {score:.4f}")
    bt.logging.info("")

    # Pass/fail threshold (matching validator)
    if mape < 0.5:  # Less than 50% MAPE
        bt.logging.success("Model ready for submission")
    else:
        bt.logging.warning(
            "Model has high error rate. Consider improving before submission."
        )


def evaluate_model(model_path: str) -> None:
    """
    Evaluate an ONNX model locally with real sample data.

    Validates:
    - Model file exists, valid ONNX format, within size limit
    - Input shape matches expected features
    - Output shape is single price prediction
    - Predictions are valid (no NaN/Inf)

    Displays:
    - Interface validation results
    - Sample predictions vs actual prices
    - MAPE and score (matching validator scoring)

    Args:
        model_path: Path to the ONNX model file.

    Raises:
        ModelInterfaceError: If model interface doesn't match validator expectations.
        InferenceError: If model inference fails.
        InvalidPredictionError: If model produces NaN/Inf values.
    """
    bt.logging.info(f"Evaluating model: {model_path}")
    bt.logging.info("")

    # 1. Validate model file (exists, size, ONNX format)
    validate_model_file(model_path)
    bt.logging.success("Model file valid")

    # 2. Load model and validate interface
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise InferenceError(f"Failed to load model: {e}") from e

    bt.logging.info("")
    bt.logging.info("Interface validation:")
    input_name, num_features = validate_model_interface(session)

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    bt.logging.success(
        f"  Input: {input_info.shape} features - matches validator ({num_features} features)"
    )
    bt.logging.success(f"  Output: {output_info.shape} price prediction")

    # 3. Run inference on test samples
    features, actual_prices = load_test_samples()
    predictions = run_inference(session, input_name, features)

    # 4. Validate predictions
    validate_predictions(predictions)

    # 5. Calculate and display metrics
    display_results(predictions, actual_prices)


def submit_model(
    hf_repo_id: str,
    hf_model_filename: str,
    wallet: bt.wallet,
    subtensor: bt.subtensor,
    netuid: int,
    extrinsic_scan_blocks: int,
) -> None:
    """
    Submit a model commitment to the chain.

    Args:
        hf_repo_id: HuggingFace repository ID.
        hf_model_filename: Model filename in the repository.
        wallet: Bittensor wallet instance.
        subtensor: Bittensor subtensor instance.
        netuid: Subnet UID.
        extrinsic_scan_blocks: Maximum blocks to scan for extrinsic.

    Raises:
        ValueError: If repo_id exceeds maximum length.
        HotkeyNotRegisteredError: If hotkey is not registered on subnet.
        ModelDownloadError: If model download fails.
        HashComputationError: If hash computation fails.
    """
    bt.logging.info(LICENSE_NOTICE)
    bt.logging.info("Submitting model to chain...")

    # Validate repo_id fits in chain metadata space
    repo_bytes = len(hf_repo_id.encode("utf-8"))
    if repo_bytes > MAX_REPO_BYTES:
        raise ValueError(
            f"hf_repo_id too long: {repo_bytes} bytes (max {MAX_REPO_BYTES})"
        )

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
        raise HotkeyNotRegisteredError(
            f"Hotkey not registered on netuid {netuid}. "
            "Run `btcli subnets register` first"
        )

    bt.logging.success(f"Hotkey is registered on netuid {netuid}")

    # Check if file exists in HuggingFace before downloading
    check_hf_file_exists(hf_repo_id, hf_model_filename)

    # Download model from HuggingFace
    try:
        bt.logging.info(f"Downloading {hf_repo_id}/{hf_model_filename}...")
        model_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_model_filename,
        )
        bt.logging.success(f"Successfully downloaded model to {model_path}")
    except MinerCLIError:
        raise
    except Exception as e:
        raise ModelDownloadError(f"Failed to download model: {e}") from e

    # Validate downloaded model (exists + size + ONNX format)
    validate_model_file(model_path)

    # Compute hash
    try:
        model_hash = compute_hash(Path(model_path))
        bt.logging.info(f"Model hash: {model_hash}")
    except Exception as e:
        raise HashComputationError(f"Failed to compute hash: {e}") from e

    # Build commitment
    commitment = build_commitment(
        model_hash=model_hash,
        hf_repo_id=hf_repo_id,
    )
    bt.logging.info(f"Commitment: {commitment}")

    # Record current block
    current_block = subtensor.get_current_block()
    bt.logging.info(f"Current Block: {current_block}")

    # Submit to chain (let bittensor handle errors)
    subtensor.commit(wallet, netuid, commitment)
    bt.logging.success("Commitment submitted to chain")

    # Scan for extrinsic ID
    extrinsic_id = scan_for_extrinsic_id(
        subtensor=subtensor,
        signer_hotkey=wallet.hotkey.ss58_address,
        start_block=current_block,
        max_blocks=extrinsic_scan_blocks,
    )

    bt.logging.success("Model committed to chain.")

    # Always print extrinsic info (critical for user, even in quiet mode)
    print(f"\nExtrinsic ID: {extrinsic_id}")
    print("Add this to your HuggingFace model card:")
    print(f"  Repository: {hf_repo_id}")
    print(f"  Extrinsic:  {extrinsic_id}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments using subcommands."""
    parser = argparse.ArgumentParser(
        description="RESI Miner CLI - Evaluate and submit ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global args
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info logging (only show warnings and errors)",
    )

    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Action to perform"
    )

    # EVALUATE
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an ONNX model locally with real sample data"
    )
    eval_parser.add_argument(
        "--model.path",
        dest="model_path",
        required=True,
        help="Path to ONNX model",
    )

    # SUBMIT
    submit_parser = subparsers.add_parser(
        "submit", help="Submit a model commitment to the chain"
    )

    # HuggingFace args
    submit_parser.add_argument(
        "--hf.repo_id",
        dest="hf_repo_id",
        required=True,
        help='HuggingFace repo ID, e.g., "user/repo"',
    )
    submit_parser.add_argument(
        "--hf.model_filename",
        dest="hf_model_filename",
        default="model.onnx",
        help="Filename in HF repo (default: model.onnx)",
    )

    # Chain args
    submit_parser.add_argument(
        "--network",
        default="finney",
        help="Network (finney/mainnet/test/testnet) or endpoint URL (requires --netuid)",
    )
    submit_parser.add_argument(
        "--netuid", type=int, default=None,
        help="Subnet UID (optional, inferred from network for finney/test)"
    )

    # Wallet args
    submit_parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        required=True,
        help="Bittensor wallet name",
    )
    submit_parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        required=True,
        help="Bittensor wallet hotkey",
    )

    # Scanning args
    submit_parser.add_argument(
        "--extrinsic.scan_blocks",
        dest="extrinsic_scan_blocks",
        type=int,
        default=SCAN_MAX_BLOCKS,
        help=f"Max blocks to scan for extrinsic (default: {SCAN_MAX_BLOCKS})",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    config = parse_args()

    if not config.quiet:
        bt.logging.set_info(True)

    try:
        if config.action == "evaluate":
            evaluate_model(config.model_path)
            return 0

        elif config.action == "submit":
            netuid = config.netuid or NETWORK_NETUIDS.get(config.network)
            if netuid is None:
                bt.logging.error(
                    f"Unknown network '{config.network}'. Use --netuid to specify subnet UID."
                )
                return 2

            wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
            subtensor = bt.subtensor(network=config.network)
            submit_model(
                hf_repo_id=config.hf_repo_id,
                hf_model_filename=config.hf_model_filename,
                wallet=wallet,
                subtensor=subtensor,
                netuid=netuid,
                extrinsic_scan_blocks=config.extrinsic_scan_blocks,
            )
            return 0

        else:
            bt.logging.error(f"Unknown action: {config.action}")
            return 2

    except ValueError as e:
        bt.logging.error(str(e))
        return 2
    except MinerCLIError as e:
        bt.logging.error(str(e))
        return 1
    except Exception as e:
        bt.logging.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
