"""Tests for feature transform functions."""

from datetime import UTC, datetime

import pytest

from real_estate.data import (
    InvalidTransformValueError,
    MissingTransformFieldError,
    reset_clock,
    set_clock,
)
from real_estate.data.feature_transforms import _FEATURE_TRANSFORM_REGISTRY


class TestDaysSinceLastSale:
    """Tests for days_since_last_sale transform."""

    def setup_method(self):
        """Get the transform function."""
        self.transform = _FEATURE_TRANSFORM_REGISTRY["days_since_last_sale"]

    def teardown_method(self):
        """Reset clock after each test."""
        reset_clock()

    def test_valid_date(self):
        """Computes correct days between dates."""
        set_clock(lambda: datetime(2024, 6, 15, tzinfo=UTC))

        prop = {"last_sale_date": "2024-01-01T00:00:00+00:00"}
        result = self.transform(prop)

        # Jan 1 to Jun 15 = 166 days
        assert result == 166.0

    def test_naive_datetime_raises_error(self):
        """Naive datetime (no timezone) raises InvalidTransformValueError."""
        prop = {"last_sale_date": "2024-01-01T00:00:00"}

        with pytest.raises(InvalidTransformValueError, match="must include timezone"):
            self.transform(prop)

    def test_missing_field_raises_error(self):
        """Raises MissingTransformFieldError when field is missing."""
        prop = {}

        with pytest.raises(MissingTransformFieldError, match="Missing required field"):
            self.transform(prop)

    def test_none_value_raises_error(self):
        """Raises MissingTransformFieldError when field is None."""
        prop = {"last_sale_date": None}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)

    def test_invalid_format_raises_error(self):
        """Raises InvalidTransformValueError on unparseable date string."""
        prop = {"last_sale_date": "not-a-date"}

        with pytest.raises(InvalidTransformValueError, match="Cannot parse"):
            self.transform(prop)
