"""Data module for property feature encoding."""

from .errors import (
    DataError,
    FeatureConfigError,
    InvalidTransformValueError,
    MissingFieldError,
    MissingTransformFieldError,
    UnknownCategoryError,
    ValidationDataAuthError,
    ValidationDataError,
    ValidationDataNotFoundError,
    ValidationDataProcessingError,
    ValidationDataRateLimitError,
    ValidationDataRequestError,
)
from .feature_encoder import FeatureEncoder
from .feature_transforms import (
    PROPERTY_TYPE_CATEGORY_ORDER,
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
    "UnknownCategoryError",
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
    "PROPERTY_TYPE_CATEGORY_ORDER",
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
