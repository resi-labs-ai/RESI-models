"""Custom exceptions for chain interactions."""


class ChainError(Exception):
    """Base exception for chain-related errors."""

    pass


class ChainConnectionError(ChainError):
    """
    Raised when connection to Pylon or chain fails.

    This can happen when:
    - Pylon service is not running
    - Network connectivity issues
    - Pylon authentication fails
    """

    pass


class CommitmentError(ChainError):
    """
    Raised when commitment operations fail.

    This can happen when:
    - Commitment data is malformed
    - Commitment not found for a hotkey
    - Failed to set commitment on chain
    """

    pass


class WeightSettingError(ChainError):
    """
    Raised when setting weights fails.

    This can happen when:
    - Validator doesn't have permission to set weights
    - Rate limiting (too many weight sets)
    - Invalid weight values
    - Chain transaction fails
    """

    pass


class MetagraphError(ChainError):
    """
    Raised when metagraph operations fail.

    This can happen when:
    - Failed to fetch metagraph from chain
    - Invalid netuid
    - Chain sync issues
    """

    pass


class AuthenticationError(ChainError):
    """
    Raised when Pylon authentication fails.

    This can happen when:
    - Invalid or expired token
    - Token not configured
    - Identity mismatch
    """

    pass
