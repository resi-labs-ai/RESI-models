"""Tests for ConfigEncoder and parse_feature_config."""

from __future__ import annotations

import numpy as np
import pytest

from real_estate.data import (
    ConfigEncoder,
    FeatureConfig,
    FeatureConfigError,
    create_default_feature_config,
    parse_feature_config,
)

REQUIRED = ["living_area_sqft", "latitude", "longitude", "bedrooms", "bathrooms"]

# A minimal valid feature set (10 features — the minimum)
MINIMAL_FEATURES = REQUIRED + [
    "lot_size_sqft",
    "year_built",
    "has_pool",
    "has_garage",
    "stories",
]


def _make_raw(features: list[str] | None = None, version: str = "1.0") -> dict:
    """Helper to build a raw feature_config dict."""
    return {
        "version": version,
        "features": features if features is not None else MINIMAL_FEATURES,
    }


class TestParseFeatureConfig:
    def test_valid_minimal_config(self) -> None:
        fc = parse_feature_config(_make_raw())
        assert isinstance(fc, FeatureConfig)
        assert fc.version == "1.0"
        assert len(fc.features) == 10

    def test_wrong_version_rejected(self) -> None:
        with pytest.raises(FeatureConfigError, match="Unsupported version"):
            parse_feature_config(_make_raw(version="2.0"))

    def test_missing_version_rejected(self) -> None:
        with pytest.raises(FeatureConfigError, match="Unsupported version"):
            parse_feature_config({"features": MINIMAL_FEATURES})

    def test_features_must_be_list(self) -> None:
        with pytest.raises(FeatureConfigError, match="must be a list"):
            parse_feature_config({"version": "1.0", "features": "not_a_list"})

    def test_features_must_be_strings(self) -> None:
        with pytest.raises(FeatureConfigError, match="must be strings"):
            parse_feature_config({"version": "1.0", "features": [1, 2, 3]})

    def test_too_few_features_rejected(self) -> None:
        with pytest.raises(FeatureConfigError, match="Too few features"):
            parse_feature_config(_make_raw(features=REQUIRED))  # only 5

    def test_duplicate_features_rejected(self) -> None:
        duped = MINIMAL_FEATURES + ["has_pool"]  # has_pool appears twice
        with pytest.raises(FeatureConfigError, match="Duplicate"):
            parse_feature_config(_make_raw(features=duped))

    def test_unknown_feature_rejected(self) -> None:
        bad = MINIMAL_FEATURES + ["totally_fake_feature"]
        with pytest.raises(FeatureConfigError, match="Unknown feature"):
            parse_feature_config(_make_raw(features=bad))

    def test_missing_required_feature_rejected(self) -> None:
        # Remove "latitude" from required set
        features = [f for f in MINIMAL_FEATURES if f != "latitude"]
        features.append("has_basement")  # keep count at 10
        with pytest.raises(FeatureConfigError, match="Missing required"):
            parse_feature_config(_make_raw(features=features))

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(FeatureConfigError, match="must be a JSON object"):
            parse_feature_config("not a dict")  # type: ignore[arg-type]

    def test_extra_keys_ignored(self) -> None:
        raw = _make_raw()
        raw["extra_key"] = "ignored"
        raw["another"] = 123
        fc = parse_feature_config(raw)
        assert len(fc.features) == 10



class TestCreateDefaultConfig:
    def test_returns_all_79_features(self) -> None:
        fc = create_default_feature_config()
        assert len(fc.features) == 79

    def test_contains_all_required(self) -> None:
        fc = create_default_feature_config()
        for req in REQUIRED:
            assert req in fc.features



class TestConfigEncoderLayout:
    def test_numeric_and_boolean_indices(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)
        layout = encoder.layout

        # MINIMAL_FEATURES: living_area_sqft(0), latitude(1), longitude(2),
        # bedrooms(3), bathrooms(4), lot_size_sqft(5), year_built(6),
        # has_pool(7), has_garage(8), stories(9)
        # has_pool and has_garage are boolean, rest are numeric
        assert set(layout.numeric_indices) == {0, 1, 2, 3, 4, 5, 6, 9}
        assert set(layout.boolean_indices) == {7, 8}

    def test_lat_lon_indices(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)
        layout = encoder.layout

        assert layout.feature_names[layout.lat_index] == "latitude"
        assert layout.feature_names[layout.lon_index] == "longitude"

    def test_default_config_lat_lon_present(self) -> None:
        fc = create_default_feature_config()
        encoder = ConfigEncoder(fc)
        layout = encoder.layout
        assert layout.lat_index >= 0
        assert layout.lon_index >= 0
        assert layout.feature_names[layout.lat_index] == "latitude"
        assert layout.feature_names[layout.lon_index] == "longitude"


class TestConfigEncoderEncode:
    def test_output_shape(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        properties = [
            {"living_area_sqft": 1500, "latitude": 40.0, "longitude": -74.0,
             "bedrooms": 3, "bathrooms": 2, "lot_size_sqft": 5000,
             "year_built": 2000, "has_pool": True, "has_garage": False, "stories": 2},
            {"living_area_sqft": 2000, "latitude": 41.0, "longitude": -73.0,
             "bedrooms": 4, "bathrooms": 3, "lot_size_sqft": 8000,
             "year_built": 2010, "has_pool": False, "has_garage": True, "stories": 1},
        ]
        result = encoder.encode(properties)

        assert result.shape == (2, 10)
        assert result.dtype == np.float32

    def test_boolean_encoding(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        properties = [
            {"living_area_sqft": 1, "latitude": 1, "longitude": 1,
             "bedrooms": 1, "bathrooms": 1, "lot_size_sqft": 1,
             "year_built": 1, "has_pool": True, "has_garage": False, "stories": 1},
        ]
        result = encoder.encode(properties)

        # has_pool=index 7, has_garage=index 8
        assert result[0, 7] == 1.0
        assert result[0, 8] == 0.0

    def test_numeric_values_preserved(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        properties = [
            {"living_area_sqft": 1500.5, "latitude": 40.7, "longitude": -74.0,
             "bedrooms": 3, "bathrooms": 2.5, "lot_size_sqft": 5000,
             "year_built": 2000, "has_pool": False, "has_garage": False, "stories": 2},
        ]
        result = encoder.encode(properties)

        assert result[0, 0] == pytest.approx(1500.5, abs=0.1)
        assert result[0, 1] == pytest.approx(40.7, abs=0.01)

    def test_missing_field_defaults_to_zero(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        # Property missing "stories" field
        properties = [
            {"living_area_sqft": 1, "latitude": 1, "longitude": 1,
             "bedrooms": 1, "bathrooms": 1, "lot_size_sqft": 1,
             "year_built": 1, "has_pool": False, "has_garage": False},
        ]
        result = encoder.encode(properties)

        # stories is at index 9, should default to 0.0
        assert result[0, 9] == 0.0

    def test_none_value_defaults_to_zero(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        properties = [
            {"living_area_sqft": None, "latitude": 1, "longitude": 1,
             "bedrooms": 1, "bathrooms": 1, "lot_size_sqft": 1,
             "year_built": 1, "has_pool": False, "has_garage": False, "stories": 1},
        ]
        result = encoder.encode(properties)

        assert result[0, 0] == 0.0

    def test_feature_order_matches_config(self) -> None:
        """Features are encoded in the order specified in config."""
        # Use a specific order
        features = [
            "bedrooms", "bathrooms", "living_area_sqft", "latitude", "longitude",
            "lot_size_sqft", "year_built", "stories", "has_pool", "has_garage",
        ]
        fc = parse_feature_config(_make_raw(features=features))
        encoder = ConfigEncoder(fc)

        properties = [
            {"bedrooms": 3, "bathrooms": 2, "living_area_sqft": 1500,
             "latitude": 40.0, "longitude": -74.0, "lot_size_sqft": 5000,
             "year_built": 2000, "stories": 2, "has_pool": True, "has_garage": False},
        ]
        result = encoder.encode(properties)

        # First column should be bedrooms (3), not living_area_sqft
        assert result[0, 0] == pytest.approx(3.0)
        assert result[0, 2] == pytest.approx(1500.0)

    def test_empty_properties_list(self) -> None:
        fc = parse_feature_config(_make_raw())
        encoder = ConfigEncoder(fc)

        result = encoder.encode([])
        assert result.ndim <= 2
        assert len(result) == 0
