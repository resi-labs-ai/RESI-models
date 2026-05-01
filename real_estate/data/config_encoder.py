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


IMAGES_FEATURE_NAME = "property_images"
"""Reserved feature name. Including this in the features list opts the model into
receiving property image tensors alongside numeric features."""


@dataclass(frozen=True)
class ImageBlockConfig:
    """Image input parameters for models that opt into property_images.

    Auto-populated from v1 constants when a miner includes 'property_images'
    in their feature list. Not configured by miners directly.
    """

    dim: tuple[int, int, int]
    """Image tensor dimensions as (C, H, W)."""

    max_images_per_property: int
    """Maximum images per property. Validator pads with zeros + mask."""


IMAGE_BLOCK_V1_DIM = (3, 224, 224)
MAX_IMAGES_PER_PROPERTY_V1 = 10


@dataclass(frozen=True)
class FeatureConfig:
    """Miner's declared feature selection from feature_config.json."""

    version: str
    features: tuple[str, ...]
    image_block: ImageBlockConfig | None = None


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

    # Separate property_images from numeric/boolean features
    has_images = IMAGES_FEATURE_NAME in features
    numeric_boolean_features = [f for f in features if f != IMAGES_FEATURE_NAME]

    # Count bounds apply to numeric/boolean features only
    if len(numeric_boolean_features) < MIN_FEATURES:
        raise FeatureConfigError(
            f"Too few features: {len(numeric_boolean_features)}, minimum is {MIN_FEATURES}"
        )
    if len(numeric_boolean_features) > MAX_FEATURES:
        raise FeatureConfigError(
            f"Too many features: {len(numeric_boolean_features)}, maximum is {MAX_FEATURES}"
        )

    # Check all numeric/boolean names exist in data contract
    numeric_fields, boolean_fields, _ = _get_field_sets()
    all_known = numeric_fields | boolean_fields
    unknown = set(numeric_boolean_features) - all_known
    if unknown:
        raise FeatureConfigError(f"Unknown feature names: {sorted(unknown)}")

    # Check required features
    missing_required = REQUIRED_FEATURES - set(features)
    if missing_required:
        raise FeatureConfigError(
            f"Missing required features: {sorted(missing_required)}"
        )

    # Auto-populate image_block when property_images is in the feature list
    image_block = None
    if has_images:
        image_block = ImageBlockConfig(
            dim=IMAGE_BLOCK_V1_DIM,
            max_images_per_property=MAX_IMAGES_PER_PROPERTY_V1,
        )

    return FeatureConfig(version=version, features=tuple(features), image_block=image_block)


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
    return parse_feature_config(
        {
            "version": SUPPORTED_VERSION,
            "features": list(default_order),
        }
    )


class TabularEncoder:
    """Encodes property dicts into numeric/boolean feature arrays.

    Each model may have a different feature config, so one TabularEncoder
    is created per model. Image features are handled separately.
    """

    def __init__(self, feature_config: FeatureConfig):
        """Initialize encoder with a validated feature config.

        Args:
            feature_config: Validated FeatureConfig from parse_feature_config().
        """
        # Numeric/boolean features only — property_images is handled separately
        self._feature_names = tuple(
            f for f in feature_config.features if f != IMAGES_FEATURE_NAME
        )
        self._layout = self._compute_layout(self._feature_names)

    @staticmethod
    def _compute_layout(feature_names: tuple[str, ...]) -> FeatureLayout:
        """Compute feature layout from feature names and YAML data contract."""
        numeric_fields, boolean_fields, _ = _get_field_sets()

        numeric_indices: list[int] = []
        boolean_indices: list[int] = []
        lat_index = -1
        lon_index = -1

        for i, name in enumerate(feature_names):
            if name in numeric_fields:
                numeric_indices.append(i)
            elif name in boolean_fields:
                boolean_indices.append(i)

            if name == "latitude":
                lat_index = i
            elif name == "longitude":
                lon_index = i

        return FeatureLayout(
            feature_names=feature_names,
            numeric_indices=tuple(numeric_indices),
            boolean_indices=tuple(boolean_indices),
            lat_index=lat_index,
            lon_index=lon_index,
        )

    def encode(self, properties: list[dict]) -> np.ndarray:
        """Encode properties using only the numeric/boolean features, in listed order.

        Args:
            properties: List of property dicts.

        Returns:
            np.ndarray of shape (N, num_numeric_boolean_features), dtype float32.
        """
        _, boolean_fields, _ = _get_field_sets()
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
