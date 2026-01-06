"""Custom exceptions for duplicate detection module."""

from __future__ import annotations


class DuplicateDetectionError(Exception):
    """Base exception for duplicate detection-related errors."""

    pass


class PioneerDetectionError(DuplicateDetectionError):
    """
    Raised when pioneer detection fails due to missing metadata.

    This can happen when:
    - A hotkey in a duplicate group has no chain metadata
    - Metadata dict is incomplete
    """

    def __init__(self, message: str, missing_hotkeys: list[str] | None = None):
        super().__init__(message)
        self.missing_hotkeys = missing_hotkeys or []
