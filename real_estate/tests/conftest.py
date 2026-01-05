"""Pytest configuration and shared fixtures."""

import shutil
from pathlib import Path

import pytest

from real_estate.data import reset_clock


@pytest.fixture(autouse=True)
def reset_clock_after_test():
    """
    Reset feature transform clock after each test.

    This prevents clock state from leaking between tests when using
    set_clock() in feature transform tests.

    Usage in tests:
        from datetime import UTC, datetime
        from real_estate.data import set_clock

        def test_days_since_last_sale():
            # Set fixed time for deterministic test
            set_clock(lambda: datetime(2024, 6, 15, tzinfo=UTC))

            # ... test code ...

            # Clock automatically resets after test via this fixture
    """
    yield
    reset_clock()


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after test session."""
    yield
    # Cleanup after all tests complete
    root = Path(__file__).parent.parent.parent
    for artifact in ["test_model_cache"]:
        path = root / artifact
        if path.exists():
            shutil.rmtree(path)
