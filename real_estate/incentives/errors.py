"""Exceptions for incentives module."""


class IncentiveError(Exception):
    """Base exception for incentive calculation errors."""

    pass


class NoValidModelsError(IncentiveError):
    """Raised when there are no valid models to evaluate."""

    pass
