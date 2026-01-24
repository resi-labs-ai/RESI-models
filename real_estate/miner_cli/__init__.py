"""
Miner CLI for the RESI Real Estate Price Prediction Subnet.

This module provides a command-line interface for miners to:
- Evaluate ONNX models locally before submission
- Submit model commitments to the Bittensor chain

Usage:
    # Evaluate a model locally
    miner-cli evaluate --model.path ./model.onnx

    # Submit a model to chain
    miner-cli submit \\
        --model.path ./model.onnx \\
        --hf.repo_id user/repo \\
        --wallet.name miner \\
        --wallet.hotkey default

The evaluate command runs ONNX inference locally to validate your model
before submission.

The submit command hashes your local model, commits it to chain, and
scans for the extrinsic ID needed for extrinsic_record.json.
"""

from .chain import ExtrinsicInfo
from .errors import (
    ChainError,
    CommitmentError,
    ConfigurationError,
    EvaluationError,
    ExtrinsicNotFoundError,
    HotkeyNotRegisteredError,
    InvalidONNXFormatError,
    MinerCLIError,
    ModelInterfaceError,
    ModelNotFoundError,
    ModelSizeExceededError,
    ModelValidationError,
)
from .models import EvaluateResult, SubmitResult

__all__ = [
    # Results
    "EvaluateResult",
    "SubmitResult",
    "ExtrinsicInfo",
    # Errors
    "MinerCLIError",
    "ModelValidationError",
    "ModelNotFoundError",
    "ModelSizeExceededError",
    "InvalidONNXFormatError",
    "ModelInterfaceError",
    "EvaluationError",
    "ChainError",
    "HotkeyNotRegisteredError",
    "CommitmentError",
    "ExtrinsicNotFoundError",
    "ConfigurationError",
]
