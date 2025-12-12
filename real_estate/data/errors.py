"""Custom exceptions for data module."""


class DataError(Exception):
    """Base exception for data-related errors."""

    pass


# --- Feature encoding errors ---


class FeatureConfigError(DataError):
    """
    Raised when feature configuration is invalid or cannot be loaded.

    This can happen when:
    - Config file not found
    - Invalid YAML/JSON syntax
    - Missing required keys in config
    - Mapping file not found
    """

    pass


class MissingFieldError(DataError):
    """
    Raised when a required field is missing from property data.

    This can happen when:
    - Property dict doesn't have a required numeric field
    - Property dict doesn't have a required categorical field
    """

    pass


class UnknownCategoryError(DataError):
    """
    Raised when a categorical value is not in the mapping.

    This can happen when:
    - Property has a category value not defined in the mapping JSON
    - New category appears in data that wasn't in training set
    """

    pass


# --- Feature transform errors ---


class MissingTransformFieldError(DataError):
    """
    Raised when a required field for a transform is missing.

    This can happen when:
    - Property dict doesn't have the source field needed by transform
    - Field is None when a value is required
    """

    pass


class InvalidTransformValueError(DataError):
    """
    Raised when a field value cannot be parsed by a transform.

    This can happen when:
    - Date string is not in ISO format
    - Value is wrong type for the transform
    """

    pass


# --- Scraper client errors ---


class ScraperError(DataError):
    """Base exception for scraper-related errors."""

    pass


class ScraperAuthError(ScraperError):
    """
    Raised when scraper authentication fails.

    This can happen when:
    - Invalid hotkey signature (401)
    - Hotkey not in whitelist (403)
    - Nonce validation failed
    """

    pass


class ScraperRequestError(ScraperError):
    """
    Raised when scraper request fails.

    This can happen when:
    - Connection error
    - HTTP error status
    - Invalid JSON response
    - Missing 'properties' key in response
    """

    pass
