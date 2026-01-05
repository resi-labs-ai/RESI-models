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
    - Property dict doesn't have a required field for a transform
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


# --- Validation Data API errors ---


class ValidationDataError(DataError):
    """Base exception for validation data API errors."""

    pass


class ValidationDataAuthError(ValidationDataError):
    """
    Raised when validation data API authentication fails.

    HTTP 401: Invalid signature, expired nonce, unknown hotkey, etc.
    """

    pass


class ValidationDataNotFoundError(ValidationDataError):
    """
    Raised when no validation set exists for requested date.

    HTTP 404: No data available.
    """

    pass


class ValidationDataRateLimitError(ValidationDataError):
    """
    Raised when rate limit exceeded.

    HTTP 429: Too many requests (10/min limit).
    """

    pass


class ValidationDataRequestError(ValidationDataError):
    """
    Raised when validation data API request fails.

    Connection errors, invalid responses, etc.
    """

    pass


class ValidationDataProcessingError(ValidationDataError):
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
        Initialize ValidationDataProcessingError.

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
