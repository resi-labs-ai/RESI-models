"""Miscellaneous utility functions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import lru_cache, update_wrapper
from math import floor
from typing import TYPE_CHECKING, Any

# Import only for type checking (mypy) to avoid circular import at runtime
if TYPE_CHECKING:
    from real_estate.validator import Validator

logger = logging.getLogger(__name__)


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate
    a new hash value at regular time intervals specified by `seconds`.

    Args:
        seconds: The number of seconds after which a new hash value will be generated.

    Yields:
        A hash value that represents the current time interval.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1) -> Callable:
    """
    Decorator that creates a cache of the most recently used function calls
    with a time-to-live (TTL) feature.

    Args:
        maxsize: Maximum size of the cache. Defaults to 128.
        typed: If True, arguments of different types will be cached separately.
        ttl: The time-to-live for each cache entry in seconds. Defaults to -1 (permanent).

    Returns:
        A decorator that can be applied to functions to cache their return values.
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash: int, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


# Last successfully fetched block (fallback on transient errors)
_last_known_block: int | None = None


# 12 seconds updating block (matches Bittensor block time)
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(validator: Validator) -> int:
    """
    Get the current block number with TTL caching.

    Retrieves the block number via subtensor websocket connection.
    Cached for 12 seconds to avoid excessive RPC calls while staying
    reasonably current with chain state.

    Falls back to last known block on transient WebSocket errors.

    Args:
        validator: The validator instance (must have subtensor initialized).

    Returns:
        The current block number.
    """
    global _last_known_block  # noqa: PLW0603
    try:
        result: int = validator.subtensor.get_current_block()
        _last_known_block = result
        return result
    except Exception as e:
        if _last_known_block is not None:
            logger.warning(
                f"subtensor.get_current_block() failed ({e}), "
                f"using last known block {_last_known_block}"
            )
            return _last_known_block
        logger.error(f"subtensor.get_current_block() failed with no fallback: {e}")
        raise
