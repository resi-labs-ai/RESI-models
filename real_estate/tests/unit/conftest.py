"""Shared fixtures for unit tests."""

import sys
from unittest.mock import MagicMock

# Mock pylon_client module to allow tests to run without pylon installed
# This must happen before any imports that require pylon_client
if "pylon_client" not in sys.modules:
    mock_pylon_client = MagicMock()
    mock_pylon_client.v1 = MagicMock()

    sys.modules["pylon_client"] = mock_pylon_client
    sys.modules["pylon_client.v1"] = mock_pylon_client.v1
