"""Validation utilities for miner CLI."""

import os

import bittensor as bt
import huggingface_hub
import onnx

# Constants
MAX_MODEL_SIZE_MB = 200
REQUIRED_ONNX_VERSION = "1.20.0"
REQUIRED_ONNXRUNTIME_VERSION = "1.20.1"


def check_onnx_versions() -> bool:
    """
    Verify onnx and onnxruntime versions match validator requirements.

    Returns:
        True if versions match, False otherwise.
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

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        True if valid, False otherwise.
    """
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


def check_hf_file_exists(
    repo_id: str, filename: str, token: str | None = None
) -> bool:
    """
    Check if file exists in HuggingFace repo.

    Args:
        repo_id: HuggingFace repository ID (e.g., "user/repo").
        filename: Name of the file to check.
        token: Optional HuggingFace token for private repos.

    Returns:
        True if file exists, False otherwise.
    """
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
