"""Tests for FeatureEncoder."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from real_estate.data import (
    FeatureConfigError,
    FeatureEncoder,
    MissingFieldError,
    UnknownCategoryError,
    reset_clock,
    set_clock,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "data"


@pytest.fixture
def encoder():
    """Create encoder with test config."""
    return FeatureEncoder(config_path=FIXTURES_DIR / "feature_config.yaml")


class TestInit:
    """Tests for FeatureEncoder initialization."""

    def test_loads_config(self, encoder):
        """Encoder loads config successfully."""
        assert encoder.get_feature_count() == 6
        assert encoder.get_feature_names() == [
            "price",
            "beds",
            "sqft",
            "days_since_last_sale",
            "city",
            "property_type",
        ]

    def test_missing_mapping_file_raises_error(self):
        """Missing mapping file raises FeatureConfigError."""
        with pytest.raises(FeatureConfigError, match="Mapping file not found"):
            FeatureEncoder(config_path=FIXTURES_DIR / "feature_config_missing_mapping.yaml")

    def test_unregistered_transform_raises_error(self):
        """Unregistered transform raises FeatureConfigError."""
        with pytest.raises(FeatureConfigError, match="no registered functions"):
            FeatureEncoder(
                config_path=FIXTURES_DIR / "feature_config_unregistered_transform.yaml"
            )

    def test_invalid_mapping_json_raises_error(self):
        """Invalid JSON in mapping file raises FeatureConfigError."""
        with pytest.raises(FeatureConfigError, match="Invalid JSON"):
            FeatureEncoder(
                config_path=FIXTURES_DIR / "feature_config_invalid_mapping.yaml"
            )


class TestEncode:
    """Tests for encoding properties."""

    def teardown_method(self):
        """Reset clock after each test."""
        reset_clock()

    def test_encode_numeric_fields(self, encoder):
        """Numeric fields are encoded correctly."""
        props = [{"price": 500000, "beds": 3, "sqft": 2000, "city": "Austin", "property_type": "single_family", "last_sale_date": "2024-01-01T00:00:00+00:00"}]
        result = encoder.encode(props)

        assert result.shape == (1, 6)
        assert result[0, 0] == 500000.0  # price
        assert result[0, 1] == 3.0  # beds
        assert result[0, 2] == 2000.0  # sqft

    def test_encode_categorical_fields(self, encoder):
        """Categorical fields are encoded to integers."""
        props = [{"price": 100, "beds": 1, "sqft": 500, "city": "Dallas", "property_type": "condo", "last_sale_date": "2024-01-01T00:00:00+00:00"}]
        result = encoder.encode(props)

        assert result[0, 4] == 1.0  # Dallas -> 1
        assert result[0, 5] == 1.0  # condo -> 1

    def test_encode_batch(self, encoder):
        """Batch encoding returns correct shape."""
        props = [
            {"price": 100, "beds": 1, "sqft": 500, "city": "Austin", "property_type": "single_family", "last_sale_date": "2024-01-01T00:00:00+00:00"},
            {"price": 200, "beds": 2, "sqft": 1000, "city": "Dallas", "property_type": "condo", "last_sale_date": "2024-01-01T00:00:00+00:00"},
            {"price": 300, "beds": 3, "sqft": 1500, "city": "Houston", "property_type": "townhouse", "last_sale_date": "2024-01-01T00:00:00+00:00"},
        ]
        result = encoder.encode(props)

        assert result.shape == (3, 6)
        assert result.dtype == np.float32

    def test_encode_missing_numeric_field_raises_error(self, encoder):
        """Missing numeric field raises MissingFieldError."""
        props = [{"price": 100, "sqft": 500, "city": "Austin", "property_type": "single_family"}]  # missing beds

        with pytest.raises(MissingFieldError, match="beds"):
            encoder.encode(props)

    def test_encode_missing_categorical_field_raises_error(self, encoder):
        """Missing categorical field raises MissingFieldError."""
        props = [{"price": 100, "beds": 1, "sqft": 500, "property_type": "single_family", "last_sale_date": "2024-01-01T00:00:00+00:00"}]  # missing city

        with pytest.raises(MissingFieldError, match="city"):
            encoder.encode(props)

    def test_encode_unknown_category_raises_error(self, encoder):
        """Unknown category value raises UnknownCategoryError."""
        props = [{"price": 100, "beds": 1, "sqft": 500, "city": "Unknown City", "property_type": "single_family", "last_sale_date": "2024-01-01T00:00:00+00:00"}]

        with pytest.raises(UnknownCategoryError, match="Unknown City"):
            encoder.encode(props)

    def test_encode_extra_fields_ignored(self, encoder):
        """Extra fields in property dict are ignored."""
        props = [{
            "price": 100,
            "beds": 1,
            "sqft": 500,
            "city": "Austin",
            "property_type": "single_family",
            "last_sale_date": "2024-01-01T00:00:00+00:00",
            "extra_field": "ignored",
            "another_extra": 999,
        }]
        result = encoder.encode(props)

        assert result.shape == (1, 6)

    def test_encode_preserves_feature_order(self, encoder):
        """Features are encoded in the order specified in config."""
        set_clock(lambda: datetime(2024, 6, 15, tzinfo=UTC))

        props = [{
            "property_type": "condo",  # listed last but should be at index 5
            "city": "Dallas",  # listed second-to-last but should be at index 4
            "sqft": 1500,
            "beds": 2,
            "price": 300000,
            "last_sale_date": "2024-01-01T00:00:00+00:00",
        }]
        result = encoder.encode(props)

        # Order from config: price, beds, sqft, days_since_last_sale, city, property_type
        assert result[0, 0] == 300000.0  # price
        assert result[0, 1] == 2.0  # beds
        assert result[0, 2] == 1500.0  # sqft
        assert result[0, 3] == 166.0  # days_since_last_sale (Jan 1 to Jun 15)
        assert result[0, 4] == 1.0  # Dallas -> 1
        assert result[0, 5] == 1.0  # condo -> 1


class TestHelpers:
    """Tests for helper methods."""

    def test_get_feature_names(self, encoder):
        """get_feature_names returns ordered list."""
        names = encoder.get_feature_names()
        assert names == ["price", "beds", "sqft", "days_since_last_sale", "city", "property_type"]

    def test_get_feature_count(self, encoder):
        """get_feature_count returns correct count."""
        assert encoder.get_feature_count() == 6

    def test_get_categorical_mapping(self, encoder):
        """get_categorical_mapping returns copy of mapping."""
        mapping = encoder.get_categorical_mapping("city")

        assert mapping == {"Austin": 0, "Dallas": 1, "Houston": 2}

        # Verify it's a copy
        mapping["NewCity"] = 99
        assert "NewCity" not in encoder.get_categorical_mapping("city")

    def test_get_categorical_mapping_unknown_field_raises_error(self, encoder):
        """get_categorical_mapping raises FeatureConfigError for unknown field."""
        with pytest.raises(FeatureConfigError, match="No mapping for field"):
            encoder.get_categorical_mapping("nonexistent")
