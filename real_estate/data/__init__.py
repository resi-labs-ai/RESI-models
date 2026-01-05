"""Data module for property feature encoding."""

from .errors import (
    DataError,
    FeatureConfigError,
    InvalidTransformValueError,
    MissingFieldError,
    MissingTransformFieldError,
    ValidationDataAuthError,
    ValidationDataError,
    ValidationDataNotFoundError,
    ValidationDataProcessingError,
    ValidationDataRateLimitError,
    ValidationDataRequestError,
)
from .feature_encoder import FeatureEncoder
from .feature_transforms import (
    feature_transform,
    get_registered_feature_transforms,
    reset_clock,
    set_clock,
)
from .models import PropertyData, ValidationDataset
from .validation_dataset_client import (
    RawFileInfo,
    ValidationDatasetClient,
    ValidationDatasetClientConfig,
    ValidationDatasetResponse,
)

__all__ = [
    # Errors
    "DataError",
    "FeatureConfigError",
    "InvalidTransformValueError",
    "MissingFieldError",
    "MissingTransformFieldError",
    "ValidationDataAuthError",
    "ValidationDataError",
    "ValidationDataNotFoundError",
    "ValidationDataProcessingError",
    "ValidationDataRateLimitError",
    "ValidationDataRequestError",
    # Feature encoding
    "FeatureEncoder",
    "feature_transform",
    "get_registered_feature_transforms",
    # Clock utilities (for testing)
    "set_clock",
    "reset_clock",
    # Models
    "PropertyData",
    "ValidationDataset",
    # Validation Dataset Client
    "ValidationDatasetClient",
    "ValidationDatasetClientConfig",
    "ValidationDatasetResponse",
    "RawFileInfo",
]
