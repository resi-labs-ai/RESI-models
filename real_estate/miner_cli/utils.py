"""Validation utilities for miner CLI."""

from pathlib import Path

import bittensor as bt
import huggingface_hub
import onnx

from .config import (
    MAX_MODEL_SIZE_MB,
    REQUIRED_ONNX_VERSION,
    REQUIRED_ONNXRUNTIME_VERSION,
)
from .errors import (
    DependencyError,
    InvalidONNXFormatError,
    ModelFileNotFoundError,
    ModelNotFoundError,
    ModelSizeExceededError,
    RepoNotFoundError,
)


def check_onnx_versions() -> None:
    """
    Verify onnx and onnxruntime versions match validator requirements.

    Raises:
        DependencyError: If onnxruntime is not installed or versions don't match.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise DependencyError(
            f"onnxruntime not installed. Run: pip install onnxruntime=={REQUIRED_ONNXRUNTIME_VERSION}"
        ) from e

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
        raise DependencyError(
            f"Version mismatch - your model may fail on validator!\n"
            f"  Installed: {', '.join(errors)}\n"
            f"  Fix with: pip install onnx=={REQUIRED_ONNX_VERSION} onnxruntime=={REQUIRED_ONNXRUNTIME_VERSION}"
        )

    bt.logging.info(
        f"ONNX versions match validator (onnx=={onnx_version}, onnxruntime=={ort_version})"
    )


def validate_model_file(model_path: str) -> None:
    """
    Validate a local ONNX model file (existence, size limit, ONNX format).

    Args:
        model_path: Path to the ONNX model file.

    Raises:
        DependencyError: If onnx/onnxruntime versions don't match.
        ModelNotFoundError: If model file doesn't exist.
        ModelSizeExceededError: If model exceeds size limit.
        InvalidONNXFormatError: If model is not valid ONNX format.
    """
    check_onnx_versions()

    if not Path(model_path).exists():
        raise ModelNotFoundError(f"Model file does not exist: {model_path}")

    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    if size_mb > MAX_MODEL_SIZE_MB:
        raise ModelSizeExceededError(
            f"Model too large: {size_mb:.2f}MB exceeds {MAX_MODEL_SIZE_MB}MB limit"
        )

    bt.logging.info(f"Model size: {size_mb:.2f}MB")

    try:
        onnx.checker.check_model(model_path)
    except onnx.checker.ValidationError as e:
        raise InvalidONNXFormatError(f"Invalid ONNX format: {e}") from e
    except Exception as e:
        raise InvalidONNXFormatError(f"Error reading ONNX model: {e}") from e


def check_hf_file_exists(repo_id: str, filename: str) -> None:
    """
    Check if file exists in a public HuggingFace repo.

    Args:
        repo_id: HuggingFace repository ID (e.g., "user/repo").
        filename: Name of the file to check.

    Raises:
        RepoNotFoundError: If repository doesn't exist.
        ModelFileNotFoundError: If file doesn't exist in repository.
    """
    try:
        if not huggingface_hub.file_exists(
            repo_id=repo_id, filename=filename, repo_type="model"
        ):
            raise ModelFileNotFoundError(
                f"File '{filename}' not found in repo '{repo_id}'. "
                f"Use --hf.model_filename if your model has a different name."
            )
    except huggingface_hub.utils.RepositoryNotFoundError as e:
        raise RepoNotFoundError(
            f"Repository '{repo_id}' not found. "
            f"Ensure the repository exists and is public."
        ) from e
