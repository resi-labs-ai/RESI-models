"""Tests for FeatureEncoder."""

from pathlib import Path

import numpy as np
import pytest

from real_estate.data import (
    FeatureConfigError,
    FeatureEncoder,
    MissingFieldError,
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
        assert encoder.get_feature_count() == 5
        assert encoder.get_feature_names() == [
            "price",
            "beds",
            "sqft",
            "property_age",
            "has_pool",
        ]

    def test_unregistered_transform_raises_error(self):
        """Unregistered transform raises FeatureConfigError."""
        with pytest.raises(FeatureConfigError, match="no registered functions"):
            FeatureEncoder(
                config_path=FIXTURES_DIR / "feature_config_unregistered_transform.yaml"
            )


class TestEncode:
    """Tests for encoding properties."""

    def test_encode_numeric_fields(self, encoder):
        """Numeric fields are encoded correctly."""
        props = [{"price": 500000, "beds": 3, "sqft": 2000, "property_age": 20, "has_pool": True}]
        result = encoder.encode(props)

        assert result.shape == (1, 5)
        assert result[0, 0] == 500000.0  # price
        assert result[0, 1] == 3.0  # beds
        assert result[0, 2] == 2000.0  # sqft

    def test_encode_boolean_fields(self, encoder):
        """Boolean fields are encoded correctly."""
        props_true = [{"price": 100, "beds": 1, "sqft": 500, "property_age": 20, "has_pool": True}]
        props_false = [{"price": 100, "beds": 1, "sqft": 500, "property_age": 20, "has_pool": False}]

        result_true = encoder.encode(props_true)
        result_false = encoder.encode(props_false)

        assert result_true[0, 4] == 1.0  # has_pool True
        assert result_false[0, 4] == 0.0  # has_pool False

    def test_encode_batch(self, encoder):
        """Batch encoding returns correct shape."""
        props = [
            {"price": 100, "beds": 1, "sqft": 500, "property_age": 20, "has_pool": True},
            {"price": 200, "beds": 2, "sqft": 1000, "property_age": 20, "has_pool": False},
            {"price": 300, "beds": 3, "sqft": 1500, "property_age": 20, "has_pool": True},
        ]
        result = encoder.encode(props)

        assert result.shape == (3, 5)
        assert result.dtype == np.float32

    def test_encode_missing_numeric_field_raises_error(self, encoder):
        """Missing numeric field raises MissingFieldError."""
        props = [{"price": 100, "sqft": 500, "has_pool": True}]  # missing beds

        with pytest.raises(MissingFieldError, match="beds"):
            encoder.encode(props)

    def test_encode_missing_boolean_field_raises_error(self, encoder):
        """Missing boolean field raises MissingFieldError."""
        props = [{"price": 100, "beds": 1, "sqft": 500, "property_age": 20}]  # missing has_pool

        with pytest.raises(MissingFieldError, match="has_pool"):
            encoder.encode(props)

    def test_encode_none_boolean_field_raises_error(self, encoder):
        """None boolean field raises MissingFieldError."""
        props = [{"price": 100, "beds": 1, "sqft": 500, "property_age": 20, "has_pool": None}]

        with pytest.raises(MissingFieldError, match="has_pool"):
            encoder.encode(props)

    def test_encode_extra_fields_ignored(self, encoder):
        """Extra fields in property dict are ignored."""
        props = [{
            "price": 100,
            "beds": 1,
            "sqft": 500,
            "property_age": 20,
            "has_pool": False,
            "extra_field": "ignored",
            "another_extra": 999,
        }]
        result = encoder.encode(props)

        assert result.shape == (1, 5)

    def test_encode_preserves_feature_order(self, encoder):
        """Features are encoded in the order specified in config."""
        props = [{
            "sqft": 1500,
            "beds": 2,
            "price": 300000,
            "property_age": 10,
            "has_pool": True,
        }]
        result = encoder.encode(props)

        # Order from config: price, beds, sqft, property_age, has_pool
        assert result[0, 0] == 300000.0  # price
        assert result[0, 1] == 2.0  # beds
        assert result[0, 2] == 1500.0  # sqft
        assert result[0, 3] == 10.0  # property_age (pre-computed by API)
        assert result[0, 4] == 1.0  # has_pool


class TestHelpers:
    """Tests for helper methods."""

    def test_get_feature_names(self, encoder):
        """get_feature_names returns ordered list."""
        names = encoder.get_feature_names()
        assert names == ["price", "beds", "sqft", "property_age", "has_pool"]

    def test_get_feature_count(self, encoder):
        """get_feature_count returns correct count."""
        assert encoder.get_feature_count() == 5
