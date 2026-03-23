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
    num_numeric = config.num_numeric_features

    perturbed = features.copy()
    noise = rng.normal(0, config.global_noise_pct, size=(features.shape[0], num_numeric))
    perturbed[:, :num_numeric] *= 1.0 + noise

    return perturbed
