"""Data module for property feature encoding."""

from .config_encoder import (
    ConfigEncoder,
    FeatureConfig,
    FeatureLayout,
    create_default_feature_config,
    load_feature_config,
    parse_feature_config,
)
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
    ATHRecord,
    RawFileInfo,
    ValidationClient,
    ValidationClientConfig,
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
    # Config-driven feature encoding
    "ConfigEncoder",
    "FeatureConfig",
    "FeatureLayout",
    "create_default_feature_config",
    "load_feature_config",
    "parse_feature_config",
    # Legacy feature encoding
    "FeatureEncoder",
    "feature_transform",
    "get_registered_feature_transforms",
    # Clock utilities (for testing)
    "set_clock",
    "reset_clock",
    # Models
    "PropertyData",
    "ValidationDataset",
    # Validation Client
    "ATHRecord",
    "ValidationClient",
    "ValidationClientConfig",
    "ValidationDatasetResponse",
    "RawFileInfo",
]
