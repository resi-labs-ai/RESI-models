"""Feature perturbation for generalization testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .models import GeneralizationConfig

if TYPE_CHECKING:
    from ..data.config_encoder import FeatureLayout

# XOR mask to derive spatial perturbation seed from the base seed,
# ensuring spatial and global perturbation RNGs are independent.
_SPATIAL_SEED_MASK = 0x5947141


def perturb_features(
    features: np.ndarray,
    config: GeneralizationConfig,
    layout: FeatureLayout,
) -> np.ndarray:
    """
    Apply multiplicative noise to numeric features.

    Adds ±noise_pct Gaussian noise to numeric columns (identified by layout),
    leaving boolean columns untouched.

    Args:
        features: Input feature matrix (N_samples x N_features).
        config: Generalization config with noise level and seed.
        layout: Feature layout with numeric/boolean column indices.

    Returns:
        Copy of features with noise applied to numeric columns.
    """
    rng = np.random.default_rng(config.seed)

    numeric_indices = list(layout.numeric_indices)
    perturbed = features.copy()
    noise = rng.normal(
        0, config.global_noise_pct, size=(features.shape[0], len(numeric_indices))
    )
    perturbed[:, numeric_indices] *= 1.0 + noise

    return perturbed


def perturb_spatial(
    features: np.ndarray,
    config: GeneralizationConfig,
    layout: FeatureLayout,
) -> np.ndarray:
    """
    Apply additive Gaussian noise to lat/lon columns only.

    Uses additive (not multiplicative) noise since lat/lon are coordinates.
    Derives a different seed from config.seed to avoid correlated noise
    with perturb_features().

    Args:
        features: Input feature matrix (N_samples x N_features).
        config: Generalization config with spatial noise std.
        layout: Feature layout with lat/lon column indices.

    Returns:
        Copy of features with noise applied to lat/lon columns.
    """
    # Derive a distinct seed so spatial noise is independent from global noise
    spatial_seed = config.seed ^ _SPATIAL_SEED_MASK if config.seed is not None else None
    rng = np.random.default_rng(spatial_seed)
    n_cols = features.shape[1]

    if layout.lat_index < 0 or layout.lon_index < 0:
        # No lat/lon in this feature config — skip spatial perturbation
        return features.copy()

    if layout.lat_index >= n_cols or layout.lon_index >= n_cols:
        raise ValueError(
            f"Spatial indices (lat={layout.lat_index}, lon={layout.lon_index}) "
            f"out of bounds for features with {n_cols} columns"
        )

    perturbed = features.copy()
    n_samples = features.shape[0]
    perturbed[:, layout.lat_index] += rng.normal(
        0, config.spatial_noise_std, size=n_samples
    )
    perturbed[:, layout.lon_index] += rng.normal(
        0, config.spatial_noise_std, size=n_samples
    )

    return perturbed
