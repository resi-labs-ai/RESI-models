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


class TestPropertyAge:
    """Tests for property_age transform."""

    def setup_method(self):
        """Get the transform function."""
        self.transform = _FEATURE_TRANSFORM_REGISTRY["property_age"]

    def teardown_method(self):
        """Reset clock after each test."""
        reset_clock()

    def test_valid_year(self):
        """Computes correct age from year_built."""
        set_clock(lambda: datetime(2025, 1, 1, tzinfo=UTC))

        prop = {"year_built": 2000}
        result = self.transform(prop)

        assert result == 25.0

    def test_string_year(self):
        """Handles year_built as string."""
        set_clock(lambda: datetime(2025, 1, 1, tzinfo=UTC))

        prop = {"year_built": "1990"}
        result = self.transform(prop)

        assert result == 35.0

    def test_missing_year_raises_error(self):
        """Raises MissingTransformFieldError when year_built is missing."""
        prop = {}

        with pytest.raises(MissingTransformFieldError, match="Missing required field"):
            self.transform(prop)

    def test_none_year_raises_error(self):
        """Raises MissingTransformFieldError when year_built is None."""
        prop = {"year_built": None}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)

    def test_invalid_year_raises_error(self):
        """Raises InvalidTransformValueError when year_built is not parseable."""
        prop = {"year_built": "not-a-year"}

        with pytest.raises(InvalidTransformValueError, match="Cannot parse"):
            self.transform(prop)

    def test_future_year_raises_error(self):
        """Raises InvalidTransformValueError when year_built is in the future."""
        set_clock(lambda: datetime(2025, 1, 1, tzinfo=UTC))

        prop = {"year_built": 2030}

        with pytest.raises(InvalidTransformValueError, match="in the future"):
            self.transform(prop)


class TestBedsPerBath:
    """Tests for beds_per_bath transform."""

    def setup_method(self):
        """Get the transform function."""
        self.transform = _FEATURE_TRANSFORM_REGISTRY["beds_per_bath"]

    def test_valid_ratio(self):
        """Computes correct beds per bath ratio."""
        prop = {"bedrooms": 3, "bathrooms": 2}
        result = self.transform(prop)

        assert result == 1.5

    def test_string_values(self):
        """Handles string values."""
        prop = {"bedrooms": "4", "bathrooms": "2.5"}
        result = self.transform(prop)

        assert result == 1.6

    def test_missing_bedrooms_raises_error(self):
        """Raises MissingTransformFieldError when bedrooms is missing."""
        prop = {"bathrooms": 2}

        with pytest.raises(MissingTransformFieldError, match="bedrooms"):
            self.transform(prop)

    def test_missing_bathrooms_raises_error(self):
        """Raises MissingTransformFieldError when bathrooms is missing."""
        prop = {"bedrooms": 3}

        with pytest.raises(MissingTransformFieldError, match="bathrooms"):
            self.transform(prop)

    def test_zero_bathrooms_raises_error(self):
        """Raises InvalidTransformValueError when bathrooms is 0 (divide by zero)."""
        prop = {"bedrooms": 3, "bathrooms": 0}

        with pytest.raises(InvalidTransformValueError, match="division by zero"):
            self.transform(prop)

    def test_none_bedrooms_raises_error(self):
        """Raises MissingTransformFieldError when bedrooms is None."""
        prop = {"bedrooms": None, "bathrooms": 2}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)

    def test_none_bathrooms_raises_error(self):
        """Raises MissingTransformFieldError when bathrooms is None."""
        prop = {"bedrooms": 3, "bathrooms": None}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)


class TestLotToLivingRatio:
    """Tests for lot_to_living_ratio transform."""

    def setup_method(self):
        """Get the transform function."""
        self.transform = _FEATURE_TRANSFORM_REGISTRY["lot_to_living_ratio"]

    def test_valid_ratio(self):
        """Computes correct lot to living ratio."""
        prop = {"lot_size_sqft": 10000, "living_area_sqft": 2000}
        result = self.transform(prop)

        assert result == 5.0

    def test_string_values(self):
        """Handles string values."""
        prop = {"lot_size_sqft": "8000", "living_area_sqft": "2000"}
        result = self.transform(prop)

        assert result == 4.0

    def test_missing_lot_size_raises_error(self):
        """Raises MissingTransformFieldError when lot_size_sqft is missing."""
        prop = {"living_area_sqft": 2000}

        with pytest.raises(MissingTransformFieldError, match="lot_size_sqft"):
            self.transform(prop)

    def test_missing_living_area_raises_error(self):
        """Raises MissingTransformFieldError when living_area_sqft is missing."""
        prop = {"lot_size_sqft": 10000}

        with pytest.raises(MissingTransformFieldError, match="living_area_sqft"):
            self.transform(prop)

    def test_zero_living_area_raises_error(self):
        """Raises InvalidTransformValueError when living_area_sqft is 0 (divide by zero)."""
        prop = {"lot_size_sqft": 10000, "living_area_sqft": 0}

        with pytest.raises(InvalidTransformValueError, match="division by zero"):
            self.transform(prop)

    def test_none_lot_size_raises_error(self):
        """Raises MissingTransformFieldError when lot_size_sqft is None."""
        prop = {"lot_size_sqft": None, "living_area_sqft": 2000}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)

    def test_none_living_area_raises_error(self):
        """Raises MissingTransformFieldError when living_area_sqft is None."""
        prop = {"lot_size_sqft": 10000, "living_area_sqft": None}

        with pytest.raises(MissingTransformFieldError, match="is None"):
            self.transform(prop)


class TestRegistryCompleteness:
    """Tests to ensure all transforms are properly registered."""

    def test_all_computed_fields_registered(self):
        """All computed fields have registered transforms."""
        expected_fields = [
            "days_since_last_sale",
            "property_age",
            "beds_per_bath",
            "lot_to_living_ratio",
        ]

        for field in expected_fields:
            assert field in _FEATURE_TRANSFORM_REGISTRY
