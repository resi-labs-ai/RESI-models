"""Custom exceptions for model management module."""


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


# --- Verification errors ---


class LicenseError(ModelError):
    """
    Raised when model license doesn't match required license.

    This can happen when:
    - LICENSE file is missing from HF repo
    - LICENSE content doesn't match required text
    """

    pass


class HashMismatchError(ModelError):
    """
    Raised when downloaded model hash doesn't match commitment.

    This can happen when:
    - Model was modified after commitment
    - Wrong model file uploaded to HF
    - Corrupted download
    """

    pass


class ExtrinsicVerificationError(ModelError):
    """
    Raised when extrinsic record verification fails.

    This can happen when:
    - extrinsic_record.json missing from HF repo
    - Extrinsic ID doesn't exist on chain
    - Signer doesn't match hotkey
    - Commitment data doesn't match
    """

    pass


# --- Download errors ---


class ModelTooLargeError(ModelError):
    """
    Raised when model exceeds size limit.

    This can happen when:
    - Model file exceeds configured max size (default 200MB)
    - HF API reports size > limit before download
    """

    pass


class ModelDownloadError(ModelError):
    """
    Raised when model download from HuggingFace fails.

    This can happen when:
    - Network/connection error
    - HF rate limiting
    - Repository not found
    - File not found in repository
    """

    pass


class FeatureConfigValidationError(ModelError):
    """
    Raised when a miner's feature_config.json is present but invalid.

    Distinct from a missing feature_config.json (backward-compat path) — a
    miner who submits a config has explicitly opted into per-model feature
    selection, so any validation failure is a hard rejection rather than a
    silent fall-back to the default full feature encoder.

    This can happen when:
    - JSON parse error
    - Wrong/unsupported version
    - Missing required features
    - Unknown feature names
    - Duplicate feature names
    - Too few or too many features
    """

    pass


class RepoValidationError(ModelError):
    """
    Raised when HuggingFace repository fails LICENSE file validation.

    This can happen when:
    - LICENSE file is missing from the repository
    - LICENSE file content does not match the required exclusive license hash
    """

    pass


class ModelCorruptedError(ModelError):
    """
    Raised when model file is corrupted or not valid ONNX.

    This can happen when:
    - Download was incomplete
    - File is not valid ONNX format
    - ONNX Runtime cannot load the model
    """

    pass


# --- Resource errors ---


class InsufficientDiskSpaceError(ModelError):
    """
    Raised when there's not enough disk space for download.

    This can happen when:
    - Available disk space < model size + buffer
    - Cache directory partition is full
    """

    pass


class CircuitBreakerOpenError(ModelError):
    """
    Raised when circuit breaker is open due to repeated failures.

    This can happen when:
    - Too many consecutive HF API failures
    - Need to wait for circuit breaker cooldown
    """

    pass
