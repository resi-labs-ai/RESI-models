"""Pytest configuration and shared fixtures."""

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
