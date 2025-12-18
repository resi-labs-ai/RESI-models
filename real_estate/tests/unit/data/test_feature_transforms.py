"""Tests for feature transform functions."""

from datetime import UTC, datetime

import pytest

from real_estate.data import (
    PROPERTY_TYPE_CATEGORY_ORDER,
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

    def test_missing_year_returns_null(self):
        """Returns -1.0 when year_built is missing."""
        prop = {}
        result = self.transform(prop)

        assert result == -1.0

    def test_none_year_returns_null(self):
        """Returns -1.0 when year_built is None."""
        prop = {"year_built": None}
        result = self.transform(prop)

        assert result == -1.0

    def test_invalid_year_returns_null(self):
        """Returns -1.0 when year_built is not parseable."""
        prop = {"year_built": "not-a-year"}
        result = self.transform(prop)

        assert result == -1.0

    def test_future_year_returns_null(self):
        """Returns -1.0 when year_built is in the future (negative age)."""
        set_clock(lambda: datetime(2025, 1, 1, tzinfo=UTC))

        prop = {"year_built": 2030}
        result = self.transform(prop)

        assert result == -1.0


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

    def test_missing_bedrooms_returns_null(self):
        """Returns -1.0 when bedrooms is missing."""
        prop = {"bathrooms": 2}
        result = self.transform(prop)

        assert result == -1.0

    def test_missing_bathrooms_returns_null(self):
        """Returns -1.0 when bathrooms is missing."""
        prop = {"bedrooms": 3}
        result = self.transform(prop)

        assert result == -1.0

    def test_zero_bathrooms_returns_null(self):
        """Returns -1.0 when bathrooms is 0 (divide by zero)."""
        prop = {"bedrooms": 3, "bathrooms": 0}
        result = self.transform(prop)

        assert result == -1.0

    def test_none_values_return_null(self):
        """Returns -1.0 when values are None."""
        prop = {"bedrooms": None, "bathrooms": 2}
        result = self.transform(prop)

        assert result == -1.0


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

    def test_missing_lot_size_returns_null(self):
        """Returns -1.0 when lot_size_sqft is missing."""
        prop = {"living_area_sqft": 2000}
        result = self.transform(prop)

        assert result == -1.0

    def test_missing_living_area_returns_null(self):
        """Returns -1.0 when living_area_sqft is missing."""
        prop = {"lot_size_sqft": 10000}
        result = self.transform(prop)

        assert result == -1.0

    def test_zero_living_area_returns_null(self):
        """Returns -1.0 when living_area_sqft is 0 (divide by zero)."""
        prop = {"lot_size_sqft": 10000, "living_area_sqft": 0}
        result = self.transform(prop)

        assert result == -1.0


class TestBooleanTransforms:
    """Tests for all boolean transform functions."""

    BOOL_FIELDS = [
        "has_pool",
        "has_any_pool_or_spa",
        "has_garage",
        "is_new_construction",
        "has_spa",
        "has_central_air",
        "has_open_parking",
        "has_natural_gas",
        "has_fireplace",
        "has_forced_air_heating",
        "has_tile_floors",
        "has_attached_garage",
        "has_home_warranty",
        "has_heating",
        "has_view",
        "has_cooling",
        "has_hardwood_floors",
        "is_senior_community",
        "has_waterfront_view",
        "has_basement",
    ]

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_true_value(self, field):
        """Boolean True returns 1.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: True}
        result = transform(prop)

        assert result == 1.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_false_value(self, field):
        """Boolean False returns 0.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: False}
        result = transform(prop)

        assert result == 0.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_none_value(self, field):
        """None returns -1.0 (null sentinel)."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: None}
        result = transform(prop)

        assert result == -1.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_missing_field(self, field):
        """Missing field returns -1.0 (null sentinel)."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {}
        result = transform(prop)

        assert result == -1.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_string_true(self, field):
        """String 'true' returns 1.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: "true"}
        result = transform(prop)

        assert result == 1.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_string_false(self, field):
        """String 'false' returns 0.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: "false"}
        result = transform(prop)

        assert result == 0.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_numeric_one(self, field):
        """Numeric 1 returns 1.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: 1}
        result = transform(prop)

        assert result == 1.0

    @pytest.mark.parametrize("field", BOOL_FIELDS)
    def test_numeric_zero(self, field):
        """Numeric 0 returns 0.0."""
        transform = _FEATURE_TRANSFORM_REGISTRY[field]
        prop = {field: 0}
        result = transform(prop)

        assert result == 0.0


class TestPropertyTypeOneHot:
    """Tests for property type one-hot encoding transforms."""

    def test_single_family_exact_match(self):
        """Single family residence maps to single_family category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_single_family"]
        prop = {
            "property_type": "Residential",
            "property_sub_type": "Single Family Residence",
        }
        result = transform(prop)

        assert result == 1.0

    def test_single_family_other_categories_zero(self):
        """Other categories return 0.0 for single_family property."""
        prop = {
            "property_type": "Residential",
            "property_sub_type": "Single Family Residence",
        }

        for category in PROPERTY_TYPE_CATEGORY_ORDER:
            transform = _FEATURE_TRANSFORM_REGISTRY[f"home_type_{category}"]
            result = transform(prop)

            if category == "single_family":
                assert result == 1.0
            else:
                assert result == 0.0

    def test_condo_mapping(self):
        """Condominium maps to condo_townhouse category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_condo_townhouse"]
        prop = {"property_type": "Residential", "property_sub_type": "Condominium"}
        result = transform(prop)

        assert result == 1.0

    def test_townhouse_mapping(self):
        """Townhouse maps to condo_townhouse category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_condo_townhouse"]
        prop = {"property_type": "Residential", "property_sub_type": "Townhouse"}
        result = transform(prop)

        assert result == 1.0

    def test_multi_family_mapping(self):
        """Duplex maps to multi_family category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_multi_family"]
        prop = {"property_type": "Residential", "property_sub_type": "Duplex"}
        result = transform(prop)

        assert result == 1.0

    def test_residential_income_mapping(self):
        """Residential Income type maps to multi_family category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_multi_family"]
        prop = {"property_type": "Residential Income", "property_sub_type": None}
        result = transform(prop)

        assert result == 1.0

    def test_manufactured_mapping(self):
        """Mobile Home maps to manufactured category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_manufactured"]
        prop = {"property_type": "Mobile Home", "property_sub_type": None}
        result = transform(prop)

        assert result == 1.0

    def test_land_mapping(self):
        """Land type maps to land category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_land"]
        prop = {"property_type": "Land", "property_sub_type": None}
        result = transform(prop)

        assert result == 1.0

    def test_farm_mapping(self):
        """Farm type without subtype maps to farm_ranch category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_farm_ranch"]
        prop = {"property_type": "Farm", "property_sub_type": "Ranch"}
        result = transform(prop)

        assert result == 1.0

    def test_commercial_mapping(self):
        """Commercial Sale maps to commercial category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_commercial"]
        prop = {"property_type": "Commercial Sale", "property_sub_type": "Office"}
        result = transform(prop)

        assert result == 1.0

    def test_other_fallback(self):
        """Unknown property type falls back to other category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_other"]
        prop = {"property_type": "Unknown Type", "property_sub_type": "Something"}
        result = transform(prop)

        assert result == 1.0

    def test_business_opportunity_mapping(self):
        """Business Opportunity maps to other category."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_other"]
        prop = {"property_type": "Business Opportunity", "property_sub_type": None}
        result = transform(prop)

        assert result == 1.0

    def test_missing_fields_default_to_other(self):
        """Missing property_type defaults to other."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_other"]
        prop = {}
        result = transform(prop)

        assert result == 1.0

    def test_none_values_default_to_other(self):
        """None property_type defaults to other."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_other"]
        prop = {"property_type": None, "property_sub_type": None}
        result = transform(prop)

        assert result == 1.0

    def test_type_only_match_fallback(self):
        """Falls back to type-only match when subtype doesn't match."""
        transform = _FEATURE_TRANSFORM_REGISTRY["home_type_single_family"]
        prop = {
            "property_type": "Residential",
            "property_sub_type": "Unknown Subtype",
        }
        result = transform(prop)

        # Residential with None subtype maps to single_family
        assert result == 1.0

    def test_one_hot_exclusivity(self):
        """Only one category should be 1.0 for any property."""
        test_cases = [
            {
                "property_type": "Residential",
                "property_sub_type": "Single Family Residence",
            },
            {"property_type": "Residential", "property_sub_type": "Condominium"},
            {"property_type": "Residential Income", "property_sub_type": "Duplex"},
            {"property_type": "Mobile Home", "property_sub_type": None},
            {"property_type": "Land", "property_sub_type": None},
            {"property_type": "Farm", "property_sub_type": "Ranch"},
            {"property_type": "Commercial Sale", "property_sub_type": "Office"},
            {"property_type": "Business Opportunity", "property_sub_type": None},
        ]

        for prop in test_cases:
            results = []
            for category in PROPERTY_TYPE_CATEGORY_ORDER:
                transform = _FEATURE_TRANSFORM_REGISTRY[f"home_type_{category}"]
                results.append(transform(prop))

            # Exactly one 1.0 and rest 0.0
            assert sum(results) == 1.0
            assert results.count(1.0) == 1
            assert results.count(0.0) == 7


class TestRegistryCompleteness:
    """Tests to ensure all transforms are properly registered."""

    def test_all_bool_fields_registered(self):
        """All boolean fields have registered transforms."""
        expected_fields = [
            "has_pool",
            "has_any_pool_or_spa",
            "has_garage",
            "is_new_construction",
            "has_spa",
            "has_central_air",
            "has_open_parking",
            "has_natural_gas",
            "has_fireplace",
            "has_forced_air_heating",
            "has_tile_floors",
            "has_attached_garage",
            "has_home_warranty",
            "has_heating",
            "has_view",
            "has_cooling",
            "has_hardwood_floors",
            "is_senior_community",
            "has_waterfront_view",
            "has_basement",
        ]

        for field in expected_fields:
            assert field in _FEATURE_TRANSFORM_REGISTRY

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

    def test_all_home_type_fields_registered(self):
        """All home type one-hot fields have registered transforms."""
        for category in PROPERTY_TYPE_CATEGORY_ORDER:
            field = f"home_type_{category}"
            assert field in _FEATURE_TRANSFORM_REGISTRY
