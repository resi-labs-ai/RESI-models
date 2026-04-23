"""Shared fixtures for unit tests."""

import sys

# Mock pylon_client submodules for CI environments where pylon may not be
# installed. Only mock if not already importable — when pylon IS installed
# (local dev), use the real package so isinstance checks work.
try:
    import pylon_client.artanis.unstable  # noqa: F401
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    # Build a mock tree that supports `from pylon_client.artanis.unstable import X`
    mock_unstable = MagicMock()
    mock_artanis = MagicMock()
    mock_artanis.unstable = mock_unstable
    mock_pylon = MagicMock()
    mock_pylon.artanis = mock_artanis

    sys.modules.setdefault("pylon_client", mock_pylon)
    sys.modules.setdefault("pylon_client.artanis", mock_artanis)
    sys.modules.setdefault("pylon_client.artanis.unstable", mock_unstable)
