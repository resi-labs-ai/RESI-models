"""Custom exceptions for miner CLI operations."""


class MinerCLIError(Exception):
    """Base exception for all miner CLI errors."""

    pass


# --- Dependency errors ---


class DependencyError(MinerCLIError):
    """
    Raised when required dependencies are missing or have wrong versions.

    This can happen when:
    - onnxruntime is not installed
    - onnx or onnxruntime version doesn't match validator requirements
    """

    pass


# --- Model validation errors ---


class ModelValidationError(MinerCLIError):
    """Base exception for model validation failures."""

    pass


class ModelNotFoundError(ModelValidationError):
    """
    Raised when model file doesn't exist at specified path.

    This can happen when:
    - Path is incorrect
    - File was moved or deleted
    """

    pass


class ModelSizeExceededError(ModelValidationError):
    """
    Raised when model exceeds maximum allowed size.

    This can happen when:
    - Model file exceeds configured max size (default 200MB)
    """

    pass


class InvalidONNXFormatError(ModelValidationError):
    """
    Raised when model file is not valid ONNX format.

    This can happen when:
    - File is corrupted
    - File is not a valid ONNX model
    - ONNX opset version incompatibility
    """

    pass


class HashComputationError(ModelValidationError):
    """
    Raised when model hash computation fails.

    This can happen when:
    - File read error during hashing
    - Unexpected error in hash algorithm
    """

    pass


class ModelInterfaceError(ModelValidationError):
    """
    Raised when model interface doesn't match validator expectations.

    This can happen when:
    - Model expects wrong number of input features
    - Model output shape is incorrect
    """

    pass


# --- Download errors ---


class DownloadError(MinerCLIError):
    """Base exception for model download operations."""

    pass


class RepoNotFoundError(DownloadError):
    """
    Raised when HuggingFace repository is not found.

    This can happen when:
    - Repository doesn't exist
    - Repository is not public
    """

    pass


class ModelFileNotFoundError(DownloadError):
    """
    Raised when model file is not found in HuggingFace repository.

    This can happen when:
    - Filename is incorrect
    - File was deleted from repository
    """

    pass


class ModelDownloadError(DownloadError):
    """
    Raised when model download from HuggingFace fails.

    This can happen when:
    - Network connectivity issues
    - HuggingFace rate limiting
    - Download interrupted
    """

    pass


# --- Chain errors ---


class ChainError(MinerCLIError):
    """Base exception for chain/blockchain operations."""

    pass


class HotkeyNotRegisteredError(ChainError):
    """
    Raised when hotkey is not registered on the subnet.

    This can happen when:
    - Hotkey has not been registered via btcli
    - Wrong netuid specified
    - Wrong network specified
    """

    pass


class ExtrinsicNotFoundError(ChainError):
    """
    Raised when extrinsic ID cannot be found after commitment submission.

    This can happen when:
    - Block scan range too small (commitment included in later block)
    - Chain indexing delay
    - Network issues during scan
    """

    pass


# --- Inference errors ---


class InferenceError(MinerCLIError):
    """
    Raised when model inference fails.

    This can happen when:
    - ONNX runtime session creation fails
    - Model execution fails
    - Input/output shape mismatch
    """

    pass


class InvalidPredictionError(MinerCLIError):
    """
    Raised when model produces invalid prediction values.

    This can happen when:
    - Model produces NaN values
    - Model produces Inf values
    - Output values are out of expected range
    """

    pass
