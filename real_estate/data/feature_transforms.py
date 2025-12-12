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


# --- Transform implementations ---


def set_clock(clock_fn: Callable[[], datetime]) -> None:
    """Set custom clock function for testing."""
    global _get_now
    _get_now = clock_fn


def reset_clock() -> None:
    """Reset clock to default UTC now."""
    global _get_now
    _get_now = _default_clock


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
