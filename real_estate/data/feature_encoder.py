"""Feature encoder for converting property data to ONNX model input."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import (
    FeatureConfigError,
    MissingFieldError,
)
from .feature_transforms import _FEATURE_TRANSFORM_REGISTRY

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encodes property dicts into numpy arrays for ONNX model input.

    Loads feature configuration from YAML.
    Validates input fields against config during encoding.
    Outputs a flat numpy array with numeric features and computed transforms.
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize encoder with feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "mappings" / "feature_config.yaml"

        self._config_path = config_path

        self._load_config()
        self._validate_feature_transforms()

        logger.info(
            f"FeatureEncoder initialized with {self.get_feature_count()} features "
            f"from {self._config_path}"
        )

    def _load_config(self) -> None:
        """Load feature configuration from YAML."""
        try:
            with open(self._config_path) as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FeatureConfigError(
                f"Config file not found: {self._config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise FeatureConfigError(f"Invalid YAML in feature config: {e}") from e

        required_keys = {
            "numeric_fields",
            "boolean_fields",
            "feature_order",
            "feature_transforms",
        }
        missing = required_keys - self._config.keys()
        if missing:
            raise FeatureConfigError(
                f"Feature config missing required keys: {sorted(missing)}"
            )

    def _validate_feature_transforms(self) -> None:
        """Validate all feature transforms in config have registered functions."""
        missing = []
        for field in self._config["feature_transforms"]:
            if field not in _FEATURE_TRANSFORM_REGISTRY:
                missing.append(field)

        if missing:
            raise FeatureConfigError(
                f"Feature transforms in config have no registered functions: {missing}. "
                f"Available registered functions: {list(_FEATURE_TRANSFORM_REGISTRY.keys())}. "
                f"Use @feature_transform('{missing[0]}') decorator to register."
            )

    def encode(self, properties: list[dict[str, Any]]) -> np.ndarray:
        """
        Encode a batch of properties to numpy array.

        Args:
            properties: List of property dicts

        Returns:
            np.ndarray of shape (len(properties), num_features), dtype float32

        Raises:
            MissingFieldError: If a required field is missing
        """
        logger.debug(f"Encoding {len(properties)} properties")

        batch = []
        for prop in properties:
            features = self._encode_single(prop)
            batch.append(features)

        result = np.array(batch, dtype=np.float32)
        logger.debug(f"Encoded to array shape {result.shape}")

        return result

    def _encode_single(self, prop: dict[str, Any]) -> list[float]:
        """Encode a single property according to feature_order."""
        features = []
        numeric_fields = self._config["numeric_fields"]
        boolean_fields = self._config["boolean_fields"]
        feature_transforms = self._config["feature_transforms"]

        for field in self._config["feature_order"]:
            if field in numeric_fields:
                if field not in prop:
                    raise MissingFieldError(
                        f"Missing required numeric field: '{field}'"
                    )
                features.append(float(prop[field]))

            elif field in boolean_fields:
                if field not in prop:
                    raise MissingFieldError(
                        f"Missing required boolean field: '{field}'"
                    )
                value = prop[field]
                if value is None:
                    raise MissingFieldError(f"Boolean field '{field}' is None")
                features.append(1.0 if value else 0.0)

            elif field in feature_transforms:
                value = _FEATURE_TRANSFORM_REGISTRY[field](prop)
                features.append(value)

        return features

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return list(self._config["feature_order"])

    def get_feature_count(self) -> int:
        """Return total number of features in encoded output."""
        return len(self._config["feature_order"])
