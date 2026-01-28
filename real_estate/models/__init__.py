"""Model management module for downloading, verifying, and caching ONNX models."""

from .cache import ModelCache
from .downloader import DownloadConfig, ModelDownloader, ModelDownloadResult
from .errors import (
    CircuitBreakerOpenError,
    ExtrinsicVerificationError,
    HashMismatchError,
    InsufficientDiskSpaceError,
    LicenseError,
    ModelDownloadError,
    ModelError,
    ModelTooLargeError,
)
from .factory import create_model_scheduler
from .models import (
    CachedModel,
    CachedModelMetadata,
    DownloadResult,
    ExtrinsicRecord,
)
from .scheduler import ModelDownloadScheduler, SchedulerConfig
from .verifier import ModelVerifier

__all__ = [
    # Factory (main entry point)
    "create_model_scheduler",
    # Errors
    "ModelError",
    "LicenseError",
    "HashMismatchError",
    "ExtrinsicVerificationError",
    "ModelTooLargeError",
    "ModelDownloadError",
    "InsufficientDiskSpaceError",
    "CircuitBreakerOpenError",
    # Models
    "CachedModelMetadata",
    "CachedModel",
    "ExtrinsicRecord",
    "DownloadResult",
    "ModelDownloadResult",
    # Config
    "DownloadConfig",
    "SchedulerConfig",
    # Components (for advanced usage/testing)
    "ModelCache",
    "ModelVerifier",
    "ModelDownloader",
    "ModelDownloadScheduler",
]
