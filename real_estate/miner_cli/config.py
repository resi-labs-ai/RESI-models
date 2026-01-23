"""
Miner CLI configuration.

This module provides access to:
- Feature configuration (shared with validator)
- Test samples for local evaluation (miner-cli specific)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .errors import ConfigurationError

# Shared data (used by both validator and miner)
_SHARED_DATA_DIR = Path(__file__).parent.parent / "data" / "mappings"
_FEATURE_CONFIG_PATH = _SHARED_DATA_DIR / "feature_config.yaml"

# Miner-CLI specific data
_TEST_SAMPLES_PATH = Path(__file__).parent / "test_samples.json"

# Chain constraints
MAX_REPO_ID_BYTES = 51  # Maximum bytes for HF repo ID in commitment


def load_feature_config() -> dict:
    """Load feature configuration from shared YAML file."""
    if not _FEATURE_CONFIG_PATH.exists():
        raise ConfigurationError(
            f"Feature config not found: {_FEATURE_CONFIG_PATH}. "
            "Ensure the data/mappings directory is properly set up."
        )
    with open(_FEATURE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_feature_order() -> list[str]:
    """
    Get ordered list of feature names.

    This is the exact order that model inputs must follow.
    """
    config = load_feature_config()
    return config.get("feature_order", [])


def get_expected_num_features() -> int:
    """Get expected number of input features (currently 73)."""
    return len(get_feature_order())


def load_test_samples() -> list[dict]:
    """
    Load test samples from shared JSON file.

    Returns:
        List of sample dicts with 'zpid', 'actual_price', and 'features'.
    """
    if not _TEST_SAMPLES_PATH.exists():
        raise ConfigurationError(f"Test samples not found: {_TEST_SAMPLES_PATH}")
    with open(_TEST_SAMPLES_PATH) as f:
        data = json.load(f)
    return data.get("samples", [])


def get_test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Get test samples as numpy arrays ready for model inference.

    Returns:
        Tuple of (features, ground_truth) where:
        - features: (N, 73) float32 array
        - ground_truth: (N,) float32 array of actual prices
    """
    samples = load_test_samples()
    feature_order = get_feature_order()

    features_list = []
    prices_list = []

    for sample in samples:
        features = []
        for feature_name in feature_order:
            value = sample["features"].get(feature_name, 0.0)
            features.append(float(value))
        features_list.append(features)
        prices_list.append(float(sample["actual_price"]))

    return (
        np.array(features_list, dtype=np.float32),
        np.array(prices_list, dtype=np.float32),
    )
