"""Config-driven feature encoder for per-model feature selection."""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .errors import FeatureConfigError

logger = logging.getLogger(__name__)

_YAML_CONFIG_PATH = Path(__file__).parent / "mappings" / "feature_config.yaml"

REQUIRED_FEATURES = frozenset(
    {"living_area_sqft", "latitude", "longitude", "bedrooms", "bathrooms"}
)

MIN_FEATURES = 10
MAX_FEATURES = 79
SUPPORTED_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureConfig:
    """Miner's declared feature selection from feature_config.json."""

    version: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class FeatureLayout:
    """Metadata about a model's feature array layout.

    Computed from a FeatureConfig by cross-referencing the YAML data contract.
    Used by perturbation functions to know which columns are numeric/boolean/spatial.
    """

    feature_names: tuple[str, ...]
    numeric_indices: tuple[int, ...]
    boolean_indices: tuple[int, ...]
    lat_index: int
    lon_index: int

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def _load_yaml_field_sets(
    yaml_path: Path | None = None,
) -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """Load numeric and boolean field sets from the YAML data contract.

    Returns:
        (numeric_fields, boolean_fields, default_feature_order)
    """
    path = yaml_path or _YAML_CONFIG_PATH
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FeatureConfigError(f"YAML config not found: {path}") from e
    except yaml.YAMLError as e:
        raise FeatureConfigError(f"Invalid YAML: {e}") from e

    numeric = frozenset(config.get("numeric_fields", []))
    boolean = frozenset(config.get("boolean_fields", []))
    order = tuple(config.get("feature_order", []))

    if not numeric and not boolean:
        raise FeatureConfigError("YAML config has no numeric_fields or boolean_fields")

    return numeric, boolean, order


@functools.lru_cache(maxsize=1)
def _get_field_sets() -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """Get cached field sets, loading from YAML on first call (thread-safe)."""
    return _load_yaml_field_sets()


def parse_feature_config(data: dict) -> FeatureConfig:
    """Parse and validate a raw JSON dict into a FeatureConfig.

    Args:
        data: Parsed JSON dict from feature_config.json.

    Returns:
        Validated FeatureConfig.

    Raises:
        FeatureConfigError: On any validation failure.
    """
    if not isinstance(data, dict):
        raise FeatureConfigError("feature_config.json must be a JSON object")

    version = data.get("version")
    if version != SUPPORTED_VERSION:
        raise FeatureConfigError(
            f"Unsupported version: {version!r}, expected {SUPPORTED_VERSION!r}"
        )

    features = data.get("features")
    if not isinstance(features, list):
        raise FeatureConfigError("'features' must be a list of strings")

    if not all(isinstance(f, str) for f in features):
        raise FeatureConfigError("All feature names must be strings")

    # Check duplicates
    if len(features) != len(set(features)):
        dupes = [f for f in features if features.count(f) > 1]
        raise FeatureConfigError(f"Duplicate feature names: {sorted(set(dupes))}")

    # Check count bounds
    if len(features) < MIN_FEATURES:
        raise FeatureConfigError(
            f"Too few features: {len(features)}, minimum is {MIN_FEATURES}"
        )
    if len(features) > MAX_FEATURES:
        raise FeatureConfigError(
            f"Too many features: {len(features)}, maximum is {MAX_FEATURES}"
        )

    # Check all names exist in data contract
    numeric_fields, boolean_fields, _ = _get_field_sets()
    all_known = numeric_fields | boolean_fields
    unknown = set(features) - all_known
    if unknown:
        raise FeatureConfigError(f"Unknown feature names: {sorted(unknown)}")

    # Check required features
    missing_required = REQUIRED_FEATURES - set(features)
    if missing_required:
        raise FeatureConfigError(
            f"Missing required features: {sorted(missing_required)}"
        )

    return FeatureConfig(version=version, features=tuple(features))


def load_feature_config(path: Path) -> FeatureConfig:
    """Load and validate a feature_config.json file.

    Args:
        path: Path to feature_config.json.

    Returns:
        Validated FeatureConfig.

    Raises:
        FeatureConfigError: On file I/O or validation failure.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FeatureConfigError(f"feature_config.json not found: {path}") from e
    except json.JSONDecodeError as e:
        raise FeatureConfigError(f"Invalid JSON in feature_config.json: {e}") from e

    return parse_feature_config(data)


def create_default_feature_config() -> FeatureConfig:
    """Create a FeatureConfig with all features in default YAML order.

    Used as fallback for models without feature_config.json (migration period).
    Routed through parse_feature_config for full validation.
    """
    _, _, default_order = _get_field_sets()
    return parse_feature_config({
        "version": SUPPORTED_VERSION,
        "features": list(default_order),
    })


class ConfigEncoder:
    """Encodes properties using a miner's feature_config.json.

    Each model may have a different feature config, so one ConfigEncoder
    is created per model.
    """

    def __init__(self, feature_config: FeatureConfig):
        """Initialize encoder with a validated feature config.

        Args:
            feature_config: Validated FeatureConfig from parse_feature_config().
        """
        self._feature_names = feature_config.features
        self._layout = self._compute_layout(feature_config)

    @staticmethod
    def _compute_layout(config: FeatureConfig) -> FeatureLayout:
        """Compute feature layout from config and YAML data contract."""
        numeric_fields, boolean_fields, _ = _get_field_sets()

        numeric_indices: list[int] = []
        boolean_indices: list[int] = []
        lat_index = -1
        lon_index = -1

        for i, name in enumerate(config.features):
            if name in numeric_fields:
                numeric_indices.append(i)
            elif name in boolean_fields:
                boolean_indices.append(i)

            if name == "latitude":
                lat_index = i
            elif name == "longitude":
                lon_index = i

        return FeatureLayout(
            feature_names=config.features,
            numeric_indices=tuple(numeric_indices),
            boolean_indices=tuple(boolean_indices),
            lat_index=lat_index,
            lon_index=lon_index,
        )

    def encode(self, properties: list[dict]) -> np.ndarray:
        """Encode properties using only the listed features, in listed order.

        Args:
            properties: List of property dicts.

        Returns:
            np.ndarray of shape (N, len(features)), dtype float32.
        """
        numeric_fields, boolean_fields, _ = _get_field_sets()
        rows: list[list[float]] = []

        for prop in properties:
            row: list[float] = []
            for name in self._feature_names:
                value = prop.get(name, 0.0)
                if name in boolean_fields:
                    row.append(1.0 if value else 0.0)
                else:
                    row.append(float(value) if value is not None else 0.0)
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    @property
    def layout(self) -> FeatureLayout:
        """Get the feature layout for perturbation functions."""
        return self._layout
