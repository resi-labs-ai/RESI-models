"""Custom exceptions for evaluation module."""


class EvaluationError(Exception):
    """Base exception for evaluation-related errors."""

    pass


# --- Metrics errors ---


class MetricsError(EvaluationError):
    """
    Raised when metrics calculation fails.

    This can happen when:
    - Input arrays have different lengths
    - No samples remain after filtering
    - Division by zero (e.g., all ground truth values are zero)
    """

    pass


class EmptyDatasetError(MetricsError):
    """
    Raised when dataset is empty or becomes empty after filtering.

    This can happen when:
    - No properties in validation dataset
    - All properties filtered out by min_price threshold
    """

    pass


# --- Docker execution errors ---


class DockerError(EvaluationError):
    """Base exception for Docker-related errors."""

    pass


class DockerNotAvailableError(DockerError):
    """
    Raised when Docker daemon is not available.

    This can happen when:
    - Docker is not installed
    - Docker daemon is not running
    - Permission denied to access Docker socket
    """

    pass


class DockerImageError(DockerError):
    """
    Raised when Docker image operations fail.

    This can happen when:
    - Image not found and cannot be pulled
    - Image pull fails (network error, auth error)
    - Image is corrupted
    """

    pass


class DockerExecutionError(DockerError):
    """
    Raised when container execution fails.

    This can happen when:
    - Container exits with non-zero code
    - Container times out
    - Resource limits exceeded (OOM, CPU)
    - Model inference fails inside container
    """

    def __init__(
        self, message: str, exit_code: int | None = None, logs: str | None = None
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.logs = logs


class InferenceTimeoutError(DockerExecutionError):
    """
    Raised when model inference exceeds timeout.

    This can happen when:
    - Model is too slow
    - Model is stuck in infinite loop
    - Container hangs
    """

    pass


class InvalidPredictionError(EvaluationError):
    """
    Raised when model produces invalid predictions.

    This can happen when:
    - Output shape doesn't match expected (N,) or (N, 1)
    - Predictions contain NaN or Inf (these break metrics calculations)
    """

    pass
