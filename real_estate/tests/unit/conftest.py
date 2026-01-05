"""Shared fixtures for unit tests."""

import sys
from unittest.mock import MagicMock

# Mock pylon module to allow tests to run without pylon installed
# This must happen before any imports that require pylon
if "pylon" not in sys.modules:
    mock_pylon = MagicMock()
    mock_pylon._internal = MagicMock()
    mock_pylon._internal.common = MagicMock()
    mock_pylon._internal.common.types = MagicMock()
    mock_pylon._internal.common.types.CommitmentDataHex = MagicMock()
    mock_pylon.v1 = MagicMock()

    sys.modules["pylon"] = mock_pylon
    sys.modules["pylon._internal"] = mock_pylon._internal
    sys.modules["pylon._internal.common"] = mock_pylon._internal.common
    sys.modules["pylon._internal.common.types"] = mock_pylon._internal.common.types
    sys.modules["pylon.v1"] = mock_pylon.v1
