"""Data module for property feature encoding."""

from .config_encoder import (
    IMAGES_FEATURE_NAME,
    FeatureConfig,
    FeatureLayout,
    ImageBlockConfig,
    TabularEncoder,
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
from .image_bundle import (
    DecodedImageBundle,
    decode_for_model,
    parse_manifest,
    verify_bundle,
)
from .models import PropertyData, ValidationDataset
from .validation_dataset_client import (
    ATHRecord,
    ImageBundleResponse,
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
    "IMAGES_FEATURE_NAME",
    "TabularEncoder",
    "FeatureConfig",
    "FeatureLayout",
    "ImageBlockConfig",
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
    # Image bundle
    "DecodedImageBundle",
    "ImageBundleResponse",
    "decode_for_model",
    "parse_manifest",
    "verify_bundle",
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
