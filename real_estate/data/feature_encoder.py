"""Feature encoder for converting property data to ONNX model input."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import (
    FeatureConfigError,
    MissingFieldError,
    UnknownCategoryError,
)
from .feature_transforms import _FEATURE_TRANSFORM_REGISTRY

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encodes property dicts into numpy arrays for ONNX model input.

    Loads feature configuration and mappings from YAML/JSON files.
    Validates input fields against config during encoding.
    Outputs a flat numpy array with numeric and integer-encoded categorical features.
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
        self._mappings_dir = config_path.parent

        self._load_config()
        self._validate_feature_transforms()
        self._load_mappings()

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
            "categorical_fields",
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

    def _load_mappings(self) -> None:
        """Load categorical mappings from JSON files."""
        self._mappings: dict[str, dict[str, int]] = {}

        for field, filename in self._config["categorical_fields"].items():
            mapping_path = self._mappings_dir / filename
            try:
                with open(mapping_path) as f:
                    self._mappings[field] = json.load(f)
            except FileNotFoundError as e:
                raise FeatureConfigError(
                    f"Mapping file not found for field '{field}': {mapping_path}"
                ) from e
            except json.JSONDecodeError as e:
                raise FeatureConfigError(
                    f"Invalid JSON in mapping file for field '{field}': {e}"
                ) from e

    def encode(self, properties: list[dict[str, Any]]) -> np.ndarray:
        """
        Encode a batch of properties to numpy array.

        Args:
            properties: List of property dicts

        Returns:
            np.ndarray of shape (len(properties), num_features), dtype float32

        Raises:
            MissingFieldError: If a required field is missing
            UnknownCategoryError: If a categorical value is not in the mapping
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
        feature_transforms = self._config["feature_transforms"]
        categorical_fields = self._config["categorical_fields"]

        for field in self._config["feature_order"]:
            if field in numeric_fields:
                if field not in prop:
                    raise MissingFieldError(
                        f"Missing required numeric field: '{field}'"
                    )
                features.append(float(prop[field]))

            elif field in feature_transforms:
                value = _FEATURE_TRANSFORM_REGISTRY[field](prop)
                features.append(value)

            elif field in categorical_fields:
                if field not in prop:
                    raise MissingFieldError(
                        f"Missing required categorical field: '{field}'"
                    )
                value = self._encode_categorical(field, prop[field])
                features.append(float(value))

        return features

    def _encode_categorical(self, field: str, value: str) -> int:
        """Encode categorical value to integer."""
        mapping = self._mappings.get(field)
        if mapping is None:
            raise FeatureConfigError(f"No mapping found for categorical field: {field}")

        if value not in mapping:
            raise UnknownCategoryError(
                f"Unknown value '{value}' for field '{field}'. "
                f"Valid values: {list(mapping.keys())}"
            )

        return mapping[value]

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return list(self._config["feature_order"])

    def get_feature_count(self) -> int:
        """Return total number of features in encoded output."""
        return len(self._config["feature_order"])

    def get_categorical_mapping(self, field: str) -> dict[str, int]:
        """Return mapping for a categorical field."""
        if field not in self._mappings:
            raise FeatureConfigError(f"No mapping for field: {field}")
        return self._mappings[field].copy()
