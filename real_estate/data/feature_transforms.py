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

    Raises:
        MissingTransformFieldError: If year_built is missing or None
        InvalidTransformValueError: If year_built is not parseable or in the future
    """
    if "year_built" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'year_built' for property_age transform"
        )

    year_built = prop_dict["year_built"]
    if year_built is None:
        raise MissingTransformFieldError(
            "Field 'year_built' is None for property_age transform"
        )

    try:
        year_built_int = int(year_built)
    except (ValueError, TypeError) as e:
        raise InvalidTransformValueError(
            f"Cannot parse '{year_built}' as integer for property_age transform: {e}"
        ) from e

    current_year = _get_now().year
    age = current_year - year_built_int

    # Sanity check: age should be non-negative
    if age < 0:
        raise InvalidTransformValueError(
            f"year_built '{year_built}' is in the future for property_age transform"
        )

    return float(age)


@feature_transform("beds_per_bath")
def _compute_beds_per_bath(prop_dict: dict[str, Any]) -> float:
    """
    Calculate bedrooms per bathroom ratio.

    Raises:
        MissingTransformFieldError: If bedrooms or bathrooms is missing/None
        InvalidTransformValueError: If values are not parseable or bathrooms is 0
    """
    if "bedrooms" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'bedrooms' for beds_per_bath transform"
        )
    if "bathrooms" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'bathrooms' for beds_per_bath transform"
        )

    bedrooms = prop_dict["bedrooms"]
    bathrooms = prop_dict["bathrooms"]

    if bedrooms is None:
        raise MissingTransformFieldError(
            "Field 'bedrooms' is None for beds_per_bath transform"
        )
    if bathrooms is None:
        raise MissingTransformFieldError(
            "Field 'bathrooms' is None for beds_per_bath transform"
        )

    try:
        beds = float(bedrooms)
        baths = float(bathrooms)
    except (ValueError, TypeError) as e:
        raise InvalidTransformValueError(
            f"Cannot parse bedrooms/bathrooms for beds_per_bath transform: {e}"
        ) from e

    if baths == 0:
        raise InvalidTransformValueError(
            "bathrooms is 0 for beds_per_bath transform (division by zero)"
        )

    return beds / baths


@feature_transform("lot_to_living_ratio")
def _compute_lot_to_living_ratio(prop_dict: dict[str, Any]) -> float:
    """
    Calculate lot size to living area ratio.

    Raises:
        MissingTransformFieldError: If lot_size_sqft or living_area_sqft is missing/None
        InvalidTransformValueError: If values are not parseable or living_area is 0
    """
    if "lot_size_sqft" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'lot_size_sqft' for lot_to_living_ratio transform"
        )
    if "living_area_sqft" not in prop_dict:
        raise MissingTransformFieldError(
            "Missing required field 'living_area_sqft' for lot_to_living_ratio transform"
        )

    lot_size = prop_dict["lot_size_sqft"]
    living_area = prop_dict["living_area_sqft"]

    if lot_size is None:
        raise MissingTransformFieldError(
            "Field 'lot_size_sqft' is None for lot_to_living_ratio transform"
        )
    if living_area is None:
        raise MissingTransformFieldError(
            "Field 'living_area_sqft' is None for lot_to_living_ratio transform"
        )

    try:
        lot = float(lot_size)
        living = float(living_area)
    except (ValueError, TypeError) as e:
        raise InvalidTransformValueError(
            f"Cannot parse lot_size_sqft/living_area_sqft for lot_to_living_ratio transform: {e}"
        ) from e

    if living == 0:
        raise InvalidTransformValueError(
            "living_area_sqft is 0 for lot_to_living_ratio transform (division by zero)"
        )

    return lot / living
