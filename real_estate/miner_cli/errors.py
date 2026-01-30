"""Custom exceptions for miner CLI operations."""


class MinerCLIError(Exception):
    """Base exception for all miner CLI errors."""

    pass


# --- Model validation errors ---


class ModelValidationError(MinerCLIError):
    """Base exception for model validation failures."""

    pass


class ModelNotFoundError(ModelValidationError):
    """Raised when model file doesn't exist at specified path."""

    pass


class ModelSizeExceededError(ModelValidationError):
    """Raised when model exceeds maximum allowed size."""

    pass


class InvalidONNXFormatError(ModelValidationError):
    """Raised when model file is not valid ONNX format."""

    pass


class ModelInterfaceError(ModelValidationError):
    """
    Raised when model interface doesn't match expected format.

    Expected:
    - Input: (batch, 73) float32 features
    - Output: (batch,) or (batch, 1) float32 prices
    """

    pass


# --- Evaluation errors ---


class EvaluationError(MinerCLIError):
    """
    Raised when model evaluation fails.

    This includes inference failures and invalid prediction outputs.
    """

    pass


# --- Chain errors ---


class ChainError(MinerCLIError):
    """Base exception for chain/blockchain operations."""

    pass


class HotkeyNotRegisteredError(ChainError):
    """Raised when hotkey is not registered on the subnet."""

    pass


class CommitmentError(ChainError):
    """Raised when commitment submission fails."""

    pass


class ExtrinsicNotFoundError(ChainError):
    """Raised when commitment extrinsic cannot be found after scanning."""

    pass


# --- Configuration errors ---


class ConfigurationError(MinerCLIError):
    """Raised when configuration files are missing or invalid."""

    pass
