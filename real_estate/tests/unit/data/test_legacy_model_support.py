"""Tests for legacy model support in TabularEncoder and parse_feature_config."""

from __future__ import annotations

import numpy as np
import pytest

from real_estate.data import (
    TabularEncoder,
    FeatureConfig,
    FeatureConfigError,
    create_default_feature_config,
    parse_feature_config,
)
from real_estate.data.config_encoder import LEGACY_FEATURES, LEGACY_MAX_FEATURES

REQUIRED = ["living_area_sqft", "latitude", "longitude", "bedrooms", "bathrooms"]

def _make_raw(features: list[str] | None = None, version: str = "1.0", legacy_model: bool = False) -> dict:
    """Helper to build a raw feature_config dict."""
    data = {
        "version": version,
        "features": features if features is not None else REQUIRED + ["lot_size_sqft", "year_built", "has_pool", "has_garage", "stories"],
    }
    if legacy_model:
        data["legacy_model"] = True
    return data

class TestLegacyModelSupport:
    def test_parse_legacy_config_success(self) -> None:
        # 76 current features + 3 legacy features = 79
        default_fc = create_default_feature_config()
        features = list(default_fc.features) + list(LEGACY_FEATURES)
        
        raw = _make_raw(features=features, legacy_model=True)
        fc = parse_feature_config(raw)
        
        assert isinstance(fc, FeatureConfig)
        assert fc.legacy_model is True
        assert len(fc.features) == 79
        for f in LEGACY_FEATURES:
            assert f in fc.features

    def test_parse_legacy_config_fails_without_flag(self) -> None:
        default_fc = create_default_feature_config()
        features = list(default_fc.features) + list(LEGACY_FEATURES)
        
        # Without legacy_model=True, 79 features should be too many
        raw = _make_raw(features=features, legacy_model=False)
        with pytest.raises(FeatureConfigError, match="Too many features: 79, maximum is 76"):
            parse_feature_config(raw)

    def test_parse_legacy_config_fails_with_unknown_legacy(self) -> None:
        # 79 features but one is NOT a known legacy feature
        default_fc = create_default_feature_config()
        features = list(default_fc.features) + list(list(LEGACY_FEATURES)[:2]) + ["completely_unknown"]
        
        raw = _make_raw(features=features, legacy_model=True)
        with pytest.raises(FeatureConfigError, match=r"Unknown feature names: \['completely_unknown'\]"):
            parse_feature_config(raw)

    def test_legacy_features_zeroed_in_encode(self) -> None:
        # Create a config with legacy features
        default_fc = create_default_feature_config()
        features = list(default_fc.features) + list(LEGACY_FEATURES)
        fc = parse_feature_config(_make_raw(features=features, legacy_model=True))
        
        encoder = TabularEncoder(fc)
        
        # Input data with values for legacy features
        properties = [
            {
                "living_area_sqft": 2000.0,
                "latitude": 34.0,
                "longitude": -118.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "price_change_since_last_sale": 50000.0,
                "price_appreciation_rate": 0.05,
                "annual_appreciation_rate": 0.02,
            }
        ]
        
        encoded = encoder.encode(properties)
        
        # Find indices of legacy features
        feature_names = [f for f in fc.features if f != "property_images"]
        for legacy_f in LEGACY_FEATURES:
            idx = feature_names.index(legacy_f)
            # Should be zero regardless of input
            assert encoded[0, idx] == 0.0
            
        # Standard features should still be encoded correctly
        assert encoded[0, feature_names.index("living_area_sqft")] == 2000.0

    def test_compute_layout_identifies_legacy_as_numeric(self) -> None:
        # Legacy features are numeric, should be in numeric_indices
        features = REQUIRED + ["lot_size_sqft", "year_built", "has_pool", "has_garage", "stories"] + list(LEGACY_FEATURES)
        fc = parse_feature_config(_make_raw(features=features, legacy_model=True))
        encoder = TabularEncoder(fc)
        layout = encoder.layout
        
        feature_names = [f for f in fc.features if f != "property_images"]
        for legacy_f in LEGACY_FEATURES:
            idx = feature_names.index(legacy_f)
            assert idx in layout.numeric_indices
            assert idx not in layout.boolean_indices
