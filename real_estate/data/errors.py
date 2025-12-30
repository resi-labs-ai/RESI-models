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


# --- Validation API errors ---


class ValidationError(DataError):
    """Base exception for validation API errors."""

    pass


class ValidationAuthError(ValidationError):
    """
    Raised when validation API authentication fails.

    HTTP 401: Invalid signature, expired nonce, unknown hotkey, etc.
    """

    pass


class ValidationNotFoundError(ValidationError):
    """
    Raised when no validation set exists for requested date.

    HTTP 404: No data available.
    """

    pass


class ValidationRateLimitError(ValidationError):
    """
    Raised when rate limit exceeded.

    HTTP 429: Too many requests (10/min limit).
    """

    pass


class ValidationRequestError(ValidationError):
    """
    Raised when validation API request fails.

    Connection errors, invalid responses, etc.
    """

    pass


class ValidationProcessingError(ValidationError):
    """
    Raised when validation set is still being processed.

    HTTP 200 with status: "processing" - data not ready yet, retry later.
    """

    def __init__(
        self,
        message: str,
        validation_date: str | None = None,
        estimated_ready_time: str | None = None,
        retry_after: int | None = None,
    ):
        """
        Initialize ValidationProcessingError.

        Args:
            message: Error message
            validation_date: Date of validation set being processed
            estimated_ready_time: ISO timestamp when data is estimated to be ready
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message)
        self.validation_date = validation_date
        self.estimated_ready_time = estimated_ready_time
        self.retry_after = retry_after
