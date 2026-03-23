"""Tests for feature perturbation."""

import numpy as np

from real_estate.generalization_detector import GeneralizationConfig, perturb_features


class TestPerturbFeatures:
    def test_returns_copy(self) -> None:
        """Perturbation returns a new array, does not modify original."""
        features = np.ones((10, 80), dtype=np.float32)
        config = GeneralizationConfig()
        perturbed = perturb_features(features, config)

        assert perturbed is not features
        np.testing.assert_array_equal(features, np.ones((10, 80), dtype=np.float32))

    def test_numeric_columns_perturbed(self) -> None:
        """First N numeric columns are modified."""
        features = np.ones((100, 80), dtype=np.float32)
        config = GeneralizationConfig(num_numeric_features=52)
        perturbed = perturb_features(features, config)

        # Numeric columns should differ from original
        numeric_diff = np.abs(perturbed[:, :52] - features[:, :52])
        assert numeric_diff.sum() > 0

    def test_boolean_columns_untouched(self) -> None:
        """Boolean columns (after numeric) are not modified."""
        features = np.ones((100, 80), dtype=np.float32)
        config = GeneralizationConfig(num_numeric_features=52)
        perturbed = perturb_features(features, config)

        np.testing.assert_array_equal(perturbed[:, 52:], features[:, 52:])

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same perturbation."""
        features = np.ones((10, 80), dtype=np.float32)
        config = GeneralizationConfig(seed=123)

        perturbed1 = perturb_features(features, config)
        perturbed2 = perturb_features(features, config)

        np.testing.assert_array_equal(perturbed1, perturbed2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different perturbations."""
        features = np.ones((10, 80), dtype=np.float32)

        perturbed1 = perturb_features(features, GeneralizationConfig(seed=1))
        perturbed2 = perturb_features(features, GeneralizationConfig(seed=2))

        assert not np.array_equal(perturbed1, perturbed2)

    def test_noise_magnitude(self) -> None:
        """Noise magnitude matches configured noise_pct."""
        features = np.ones((10000, 80), dtype=np.float32) * 100.0
        config = GeneralizationConfig(global_noise_pct=0.01, num_numeric_features=52)
        perturbed = perturb_features(features, config)

        # Relative change should be approximately ±1%
        relative_change = (perturbed[:, :52] - features[:, :52]) / features[:, :52]
        assert abs(relative_change.std() - 0.01) < 0.002
