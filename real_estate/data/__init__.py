"""Data module for property feature encoding."""

from .errors import (
    DataError,
    FeatureConfigError,
    InvalidTransformValueError,
    MissingFieldError,
    MissingTransformFieldError,
    ScraperAuthError,
    ScraperError,
    ScraperRequestError,
    UnknownCategoryError,
    ValidationAuthError,
    ValidationError,
    ValidationNotFoundError,
    ValidationProcessingError,
    ValidationRateLimitError,
    ValidationRequestError,
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
from .scraper_client import (
    ScraperClient,
    ScraperConfig,
)
from .validation_client import (
    RawFileInfo,
    ValidationSetClient,
    ValidationSetConfig,
    ValidationSetResponse,
)

__all__ = [
    # Errors
    "DataError",
    "FeatureConfigError",
    "InvalidTransformValueError",
    "MissingFieldError",
    "MissingTransformFieldError",
    "ScraperAuthError",
    "ScraperError",
    "ScraperRequestError",
    "UnknownCategoryError",
    "ValidationAuthError",
    "ValidationError",
    "ValidationNotFoundError",
    "ValidationProcessingError",
    "ValidationRateLimitError",
    "ValidationRequestError",
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
    # Scraper
    "ScraperClient",
    "ScraperConfig",
    # Validation API
    "ValidationSetClient",
    "ValidationSetConfig",
    "ValidationSetResponse",
    "RawFileInfo",
]
