"""Feature perturbation for generalization testing."""

from __future__ import annotations

import numpy as np

from .models import GeneralizationConfig


def perturb_features(
    features: np.ndarray,
    config: GeneralizationConfig,
) -> np.ndarray:
    """
    Apply multiplicative noise to numeric features.

    Adds ±noise_pct Gaussian noise to the first N numeric columns,
    leaving boolean columns untouched.

    Args:
        features: Input feature matrix (N_samples x N_features).
        config: Generalization config with noise level and seed.

    Returns:
        Copy of features with noise applied to numeric columns.
    """
    rng = np.random.default_rng(config.seed)
    num_numeric = min(config.num_numeric_features, features.shape[1])

    perturbed = features.copy()
    noise = rng.normal(0, config.global_noise_pct, size=(features.shape[0], num_numeric))
    perturbed[:, :num_numeric] *= 1.0 + noise

    return perturbed


def perturb_spatial(
    features: np.ndarray,
    config: GeneralizationConfig,
) -> np.ndarray:
    """
    Apply additive Gaussian noise to lat/lon columns only.

    Uses additive (not multiplicative) noise since lat/lon are coordinates.

    Args:
        features: Input feature matrix (N_samples x N_features).
        config: Generalization config with spatial noise std and column indices.

    Returns:
        Copy of features with noise applied to lat/lon columns.
    """
    rng = np.random.default_rng(config.seed)
    n_cols = features.shape[1]

    if config.lat_index >= n_cols or config.lon_index >= n_cols:
        raise ValueError(
            f"Spatial indices (lat={config.lat_index}, lon={config.lon_index}) "
            f"out of bounds for features with {n_cols} columns"
        )

    perturbed = features.copy()
    n_samples = features.shape[0]
    perturbed[:, config.lat_index] += rng.normal(0, config.spatial_noise_std, size=n_samples)
    perturbed[:, config.lon_index] += rng.normal(0, config.spatial_noise_std, size=n_samples)

    return perturbed
