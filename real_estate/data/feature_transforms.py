"""Feature transform registry and implementations.

Each transform function is registered via @feature_transform decorator.
The name must match a key in feature_config.yaml's feature_transforms section.

These are derived features computed from raw property data.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .errors import InvalidTransformValueError, MissingTransformFieldError


def _default_clock() -> datetime:
    """Default clock returning current UTC time."""
    return datetime.now(UTC)


# Clock function for testing - defaults to UTC now
_get_now: Callable[[], datetime] = _default_clock

# Registry for feature transform functions
# Each function takes prop_dict and returns float
_FEATURE_TRANSFORM_REGISTRY: dict[str, Callable[[dict[str, Any]], float]] = {}


def feature_transform(name: str):
    """
    Decorator to register a feature transform function.

    The `name` argument MUST match an entry in `feature_transforms` list of
    feature_config.yaml. This binding is validated at encoder initialization.

    Args:
        name: The field name as it appears in feature_config.yaml

    Usage:
        @feature_transform("days_since_last_sale")
        def compute_days_since_last_sale(prop_dict: dict) -> float:
            ...

    The function receives:
        - prop_dict: Property data as dictionary

    Returns:
        float: The computed value
    """

    def decorator(func: Callable[[dict[str, Any]], float]):
        _FEATURE_TRANSFORM_REGISTRY[name] = func
        return func

    return decorator


def get_registered_feature_transforms() -> list[str]:
    """Return list of all registered feature transform names."""
    return list(_FEATURE_TRANSFORM_REGISTRY.keys())


# --- Clock utilities for testing ---


def set_clock(clock_fn: Callable[[], datetime]) -> None:
    """Set custom clock function for testing."""
    global _get_now
    _get_now = clock_fn


def reset_clock() -> None:
    """Reset clock to default UTC now."""
    global _get_now
    _get_now = _default_clock


# --- Helper functions ---


def _bool_to_float(value: Any) -> float:
    """
    Convert boolean-like value to float.

    Returns:
        -1.0 for None/missing
        0.0 for False
        1.0 for True
    """
    if value is None:
        return -1.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    # Handle string representations
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "yes", "1"):
            return 1.0
        if lower in ("false", "no", "0"):
            return 0.0
        return -1.0
    # Handle numeric (0 = False, non-zero = True)
    if isinstance(value, (int, float)):
        return 1.0 if value else 0.0
    return -1.0


def _make_bool_transform(field_name: str) -> Callable[[dict[str, Any]], float]:
    """
    Factory to create boolean transform functions.

    Args:
        field_name: The field to read from prop_dict

    Returns:
        Transform function that returns -1.0 (null), 0.0 (false), or 1.0 (true)
    """

    def transform(prop_dict: dict[str, Any]) -> float:
        value = prop_dict.get(field_name)
        return _bool_to_float(value)

    return transform


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe division that returns -1.0 on divide-by-zero.

    Args:
        numerator: The numerator value
        denominator: The denominator value

    Returns:
        Result of division, or -1.0 if denominator is zero
    """
    if denominator == 0:
        return -1.0
    return numerator / denominator


# --- Transform implementations ---


# ============================================================================
# DATE-BASED TRANSFORMS
# ============================================================================


@feature_transform("days_since_last_sale")
def _compute_days_since_last_sale(prop_dict: dict[str, Any]) -> float:
    """Calculate days between last sale and now."""
    if "last_sale_date" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'last_sale_date' for days_since_last_sale transform"
        )

    last_sale_date = prop_dict["last_sale_date"]

    if last_sale_date is None:
        raise MissingTransformFieldError(
            "Field 'last_sale_date' is None for days_since_last_sale transform"
        )

    try:
        last_sale = datetime.fromisoformat(last_sale_date)
    except (ValueError, TypeError) as e:
        raise InvalidTransformValueError(
            f"Cannot parse '{last_sale_date}' as ISO date for days_since_last_sale transform: {e}"
        ) from e

    if last_sale.tzinfo is None:
        raise InvalidTransformValueError(
            f"Datetime '{last_sale_date}' must include timezone for days_since_last_sale transform"
        )

    now = _get_now()
    delta = now - last_sale
    return float(delta.days)


# ============================================================================
# COMPUTED TRANSFORMS
# ============================================================================


@feature_transform("property_age")
def _compute_property_age(prop_dict: dict[str, Any]) -> float:
    """
    Calculate property age in years (current_year - year_built).

    Returns -1.0 if year_built is missing or None.
    """
    year_built = prop_dict.get("year_built")
    if year_built is None:
        return -1.0

    try:
        year_built_int = int(year_built)
    except (ValueError, TypeError):
        return -1.0

    current_year = _get_now().year
    age = current_year - year_built_int

    # Sanity check: age should be non-negative
    if age < 0:
        return -1.0

    return float(age)


@feature_transform("beds_per_bath")
def _compute_beds_per_bath(prop_dict: dict[str, Any]) -> float:
    """
    Calculate bedrooms per bathroom ratio.

    Returns -1.0 if either field is missing/None or bathrooms is 0.
    """
    bedrooms = prop_dict.get("bedrooms")
    bathrooms = prop_dict.get("bathrooms")

    if bedrooms is None or bathrooms is None:
        return -1.0

    try:
        beds = float(bedrooms)
        baths = float(bathrooms)
    except (ValueError, TypeError):
        return -1.0

    return _safe_divide(beds, baths)


@feature_transform("lot_to_living_ratio")
def _compute_lot_to_living_ratio(prop_dict: dict[str, Any]) -> float:
    """
    Calculate lot size to living area ratio.

    Returns -1.0 if either field is missing/None or living_area is 0.
    """
    lot_size = prop_dict.get("lot_size_sqft")
    living_area = prop_dict.get("living_area_sqft")

    if lot_size is None or living_area is None:
        return -1.0

    try:
        lot = float(lot_size)
        living = float(living_area)
    except (ValueError, TypeError):
        return -1.0

    return _safe_divide(lot, living)


# ============================================================================
# BOOLEAN TRANSFORMS
# ============================================================================

# List of all boolean fields to register
_BOOL_FIELDS = [
    "has_pool",
    "has_any_pool_or_spa",
    "has_garage",
    "is_new_construction",
    "has_spa",
    "has_central_air",
    "has_open_parking",
    "has_natural_gas",
    "has_fireplace",
    "has_forced_air_heating",
    "has_tile_floors",
    "has_attached_garage",
    "has_home_warranty",
    "has_heating",
    "has_view",
    "has_cooling",
    "has_hardwood_floors",
    "is_senior_community",
    "has_waterfront_view",
    "has_basement",
]

# Register all boolean transforms using the factory
for _field in _BOOL_FIELDS:
    _FEATURE_TRANSFORM_REGISTRY[_field] = _make_bool_transform(_field)


# ============================================================================
# PROPERTY TYPE ONE-HOT ENCODING
# ============================================================================

# Property type category mapping based on PropertyTypeEncoder
# Maps (property_type, property_sub_type) tuples to category names
_PROPERTY_TYPE_CATEGORIES = {
    "single_family": [
        ("Residential", "Single Family Residence"),
        ("Residential", "Cabin"),
        ("Residential", None),
        ("Residential Lease", "Single Family Residence"),
    ],
    "condo_townhouse": [
        ("Residential", "Condominium"),
        ("Residential", "Townhouse"),
        ("Residential", "Stock Cooperative"),
        ("Residential", "Own Your Own"),
        ("Residential Lease", "Condominium"),
        ("Residential Lease", "Townhouse"),
        ("Residential Lease", "Apartment"),
    ],
    "multi_family": [
        ("Residential", "Duplex"),
        ("Residential", "Triplex"),
        ("Residential", "Quadruplex"),
        ("Residential", "Multi Family"),
        ("Residential Income", None),
        ("Residential Income", "Duplex"),
        ("Residential Income", "Triplex"),
        ("Residential Income", "Quadruplex"),
        ("Residential Income", "Multi Family"),
    ],
    "manufactured": [
        ("Mobile Home", None),
        ("Manufactured In Park", None),
        ("Residential", "Manufactured Home"),
        ("Residential", "Mobile Home"),
        ("Residential", "Manufactured On Land"),
    ],
    "land": [
        ("Land", None),
        ("Land", "Unimproved Land"),
        ("Farm", "Unimproved Land"),
    ],
    "farm_ranch": [
        ("Farm", None),
        ("Farm", "Agriculture"),
        ("Farm", "Ranch"),
        ("Farm", "Farm"),
    ],
    "commercial": [
        ("Commercial Sale", None),
        ("Commercial Sale", "Office"),
        ("Commercial Sale", "Retail"),
        ("Commercial Sale", "Industrial"),
        ("Commercial Sale", "Warehouse"),
        ("Commercial Sale", "Mixed Use"),
        ("Commercial Sale", "Hotel/Motel"),
        ("Commercial Lease", None),
        ("Commercial Lease", "Office"),
        ("Commercial Lease", "Retail"),
        ("Commercial Lease", "Industrial"),
        ("Commercial Lease", "Warehouse"),
        ("Commercial Lease", "Mixed Use"),
        ("Commercial Lease", "Hotel/Motel"),
    ],
    "other": [
        ("Business Opportunity", None),
        ("Residential", "Boat Slip"),
        ("Residential", "Timeshare"),
        ("Residential", "Deeded Parking"),
        ("Residential Lease", "Boat Slip"),
    ],
}

# Build reverse lookup for fast encoding
_PROPERTY_TYPE_LOOKUP: dict[tuple[str | None, str | None], str] = {}
for _category, _mappings in _PROPERTY_TYPE_CATEGORIES.items():
    for _prop_type, _prop_subtype in _mappings:
        _PROPERTY_TYPE_LOOKUP[(_prop_type, _prop_subtype)] = _category

# List of all property type categories (for one-hot encoding)
PROPERTY_TYPE_CATEGORY_ORDER = [
    "single_family",
    "condo_townhouse",
    "multi_family",
    "manufactured",
    "land",
    "farm_ranch",
    "commercial",
    "other",
]


def _normalize_string(value: Any) -> str | None:
    """Normalize string value (handle None, strip whitespace)."""
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value).strip() if value else None
    return value.strip() if value else None


def _get_property_category(prop_dict: dict[str, Any]) -> str:
    """
    Determine property category from property_type and property_sub_type.

    Args:
        prop_dict: Property data dictionary

    Returns:
        Category name (one of PROPERTY_TYPE_CATEGORY_ORDER)
    """
    prop_type = _normalize_string(prop_dict.get("property_type"))
    prop_subtype = _normalize_string(prop_dict.get("property_sub_type"))

    # Try exact match first (Type + SubType)
    key = (prop_type, prop_subtype)
    if key in _PROPERTY_TYPE_LOOKUP:
        return _PROPERTY_TYPE_LOOKUP[key]

    # Try with None subtype (Type only)
    key = (prop_type, None)
    if key in _PROPERTY_TYPE_LOOKUP:
        return _PROPERTY_TYPE_LOOKUP[key]

    # Default to 'other' if no match found
    return "other"


def _make_property_type_transform(category: str) -> Callable[[dict[str, Any]], float]:
    """
    Factory to create property type one-hot transform functions.

    Args:
        category: The category to check for (e.g., 'single_family')

    Returns:
        Transform function that returns 1.0 if property matches category, else 0.0
    """

    def transform(prop_dict: dict[str, Any]) -> float:
        prop_category = _get_property_category(prop_dict)
        return 1.0 if prop_category == category else 0.0

    return transform


# Register all property type one-hot transforms
for _category in PROPERTY_TYPE_CATEGORY_ORDER:
    _transform_name = f"home_type_{_category}"
    _FEATURE_TRANSFORM_REGISTRY[_transform_name] = _make_property_type_transform(
        _category
    )
