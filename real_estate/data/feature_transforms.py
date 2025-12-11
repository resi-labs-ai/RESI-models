"""Feature transform registry and implementations.

Each transform function is registered via @feature_transform decorator.
The name must match a key in feature_config.yaml's feature_transforms section.

These are derived features computed from raw property data.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _default_clock() -> datetime:
    """Default clock returning current UTC time."""
    return datetime.now(UTC)


# Clock function for testing - defaults to UTC now
_get_now: Callable[[], datetime] = _default_clock

# Registry for feature transform functions
# Each function takes (prop_dict, config) and returns float
_FEATURE_TRANSFORM_REGISTRY: dict[str, Callable[[dict[str, Any], dict[str, Any]], float]] = {}


def feature_transform(name: str):
    """
    Decorator to register a feature transform function.

    The `name` argument MUST match a key in `feature_transforms` section of
    feature_config.yaml. This binding is validated at encoder initialization.

    Args:
        name: The field name as it appears in feature_config.yaml

    Usage:
        @feature_transform("days_since_last_sale")
        def compute_days_since_last_sale(prop_dict: dict, config: dict) -> float:
            ...

    The function receives:
        - prop_dict: Property data as dictionary
        - config: The field's config from feature_config.yaml (e.g. source, default)

    Returns:
        float: The computed value
    """

    def decorator(func: Callable[[dict[str, Any], dict[str, Any]], float]):
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
def _compute_days_since_last_sale(prop_dict: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate days between last sale and now."""
    last_sale_date = prop_dict.get(config.get("source", "last_sale_date"))
    default = config.get("default", 0)

    if not last_sale_date:
        logger.debug(f"No last_sale_date found, using default: {default}")
        return float(default)

    try:
        last_sale = datetime.fromisoformat(last_sale_date)
        # Ensure timezone-aware comparison
        if last_sale.tzinfo is None:
            last_sale = last_sale.replace(tzinfo=UTC)
        now = _get_now()
        delta = now - last_sale
        return float(delta.days)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse last_sale_date '{last_sale_date}': {e}, using default: {default}")
        return float(default)
