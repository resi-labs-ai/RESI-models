"""Tests for feature perturbation."""

import numpy as np

from real_estate.data import FeatureLayout
from real_estate.generalization_detector import (
    GeneralizationConfig,
    perturb_features,
    perturb_spatial,
)


def _make_layout(
    n_numeric: int = 51, n_boolean: int = 28
) -> FeatureLayout:
    """Create a FeatureLayout for testing."""
    total = n_numeric + n_boolean
    feature_names = tuple(f"num_{i}" for i in range(n_numeric)) + tuple(
        f"bool_{i}" for i in range(n_boolean)
    )
    # Put latitude at index 4, longitude at index 5 (within numeric)
    names_list = list(feature_names)
    names_list[4] = "latitude"
    names_list[5] = "longitude"
    return FeatureLayout(
        feature_names=tuple(names_list),
        numeric_indices=tuple(range(n_numeric)),
        boolean_indices=tuple(range(n_numeric, total)),
        lat_index=4,
        lon_index=5,
    )


DEFAULT_LAYOUT = _make_layout(51, 28)


class TestPerturbFeatures:
    def test_returns_copy(self) -> None:
        """Perturbation returns a new array, does not modify original."""
        features = np.ones((10, 79), dtype=np.float32)
        config = GeneralizationConfig()
        perturbed = perturb_features(features, config, DEFAULT_LAYOUT)

        assert perturbed is not features
        np.testing.assert_array_equal(features, np.ones((10, 79), dtype=np.float32))

    def test_numeric_columns_perturbed(self) -> None:
        """Numeric columns are modified."""
        features = np.ones((100, 79), dtype=np.float32)
        config = GeneralizationConfig()
        perturbed = perturb_features(features, config, DEFAULT_LAYOUT)

        # Numeric columns should differ from original
        numeric_diff = np.abs(perturbed[:, :51] - features[:, :51])
        assert numeric_diff.sum() > 0

    def test_boolean_columns_untouched(self) -> None:
        """Boolean columns are not modified."""
        features = np.ones((100, 79), dtype=np.float32)
        config = GeneralizationConfig()
        perturbed = perturb_features(features, config, DEFAULT_LAYOUT)

        np.testing.assert_array_equal(perturbed[:, 51:], features[:, 51:])

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same perturbation."""
        features = np.ones((10, 79), dtype=np.float32)
        config = GeneralizationConfig(seed=123)

        perturbed1 = perturb_features(features, config, DEFAULT_LAYOUT)
        perturbed2 = perturb_features(features, config, DEFAULT_LAYOUT)

        np.testing.assert_array_equal(perturbed1, perturbed2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different perturbations."""
        features = np.ones((10, 79), dtype=np.float32)

        perturbed1 = perturb_features(
            features, GeneralizationConfig(seed=1), DEFAULT_LAYOUT
        )
        perturbed2 = perturb_features(
            features, GeneralizationConfig(seed=2), DEFAULT_LAYOUT
        )

        assert not np.array_equal(perturbed1, perturbed2)

    def test_noise_magnitude(self) -> None:
        """Noise magnitude matches configured noise_pct."""
        features = np.ones((10000, 79), dtype=np.float32) * 100.0
        config = GeneralizationConfig(global_noise_pct=0.01)
        perturbed = perturb_features(features, config, DEFAULT_LAYOUT)

        # Relative change should be approximately ±1%
        relative_change = (perturbed[:, :51] - features[:, :51]) / features[:, :51]
        assert abs(relative_change.std() - 0.01) < 0.002

    def test_non_contiguous_numeric_indices(self) -> None:
        """Perturbation works with non-contiguous numeric indices."""
        # Layout with numeric at indices 0, 2, 4 and boolean at 1, 3
        layout = FeatureLayout(
            feature_names=(
                "living_area_sqft",
                "has_pool",
                "bedrooms",
                "has_garage",
                "latitude",
                "longitude",
                "bathrooms",
                "lot_size_sqft",
                "year_built",
                "has_basement",
            ),
            numeric_indices=(0, 2, 4, 5, 6, 7, 8),
            boolean_indices=(1, 3, 9),
            lat_index=4,
            lon_index=5,
        )
        features = np.ones((50, 10), dtype=np.float32)
        config = GeneralizationConfig(seed=42)
        perturbed = perturb_features(features, config, layout)

        # Boolean columns should be untouched
        np.testing.assert_array_equal(perturbed[:, 1], features[:, 1])
        np.testing.assert_array_equal(perturbed[:, 3], features[:, 3])
        np.testing.assert_array_equal(perturbed[:, 9], features[:, 9])

        # Numeric columns should be perturbed
        assert not np.array_equal(perturbed[:, 0], features[:, 0])
        assert not np.array_equal(perturbed[:, 2], features[:, 2])


class TestPerturbSpatial:
    def test_spatial_noise_independent_from_global(self) -> None:
        """perturb_spatial uses a different seed than perturb_features."""
        features = np.ones((100, 79), dtype=np.float32) * 40.0
        config = GeneralizationConfig(seed=42)

        global_result = perturb_features(features, config, DEFAULT_LAYOUT)
        spatial_result = perturb_spatial(features, config, DEFAULT_LAYOUT)

        # The lat column (index 4) should be modified differently by each
        assert not np.array_equal(
            global_result[:, 4] - features[:, 4],
            spatial_result[:, 4] - features[:, 4],
        )

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same spatial perturbation."""
        features = np.ones((10, 79), dtype=np.float32) * 40.0
        config = GeneralizationConfig(seed=123)

        result1 = perturb_spatial(features, config, DEFAULT_LAYOUT)
        result2 = perturb_spatial(features, config, DEFAULT_LAYOUT)

        np.testing.assert_array_equal(result1, result2)

    def test_returns_copy_when_no_lat_lon(self) -> None:
        """Returns unmodified copy when layout has no lat/lon."""
        layout = FeatureLayout(
            feature_names=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
            numeric_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            boolean_indices=(),
            lat_index=-1,
            lon_index=-1,
        )
        features = np.ones((10, 10), dtype=np.float32)
        config = GeneralizationConfig(seed=42)

        result = perturb_spatial(features, config, layout)
        np.testing.assert_array_equal(result, features)
        assert result is not features

    def test_only_modifies_lat_lon_columns(self) -> None:
        """Only latitude and longitude columns are modified."""
        features = np.ones((100, 79), dtype=np.float32) * 40.0
        config = GeneralizationConfig(seed=42)

        result = perturb_spatial(features, config, DEFAULT_LAYOUT)

        # Non-lat/lon columns should be untouched
        non_spatial = [i for i in range(79) if i not in (4, 5)]
        np.testing.assert_array_equal(result[:, non_spatial], features[:, non_spatial])

        # Lat/lon should be modified
        assert not np.array_equal(result[:, 4], features[:, 4])
        assert not np.array_equal(result[:, 5], features[:, 5])
