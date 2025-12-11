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
)
from .feature_encoder import FeatureEncoder
from .feature_transforms import (
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
    # Scraper
    "ScraperClient",
    "ScraperConfig",
]
