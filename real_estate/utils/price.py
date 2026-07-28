"""Utilities for fetching TAO and Alpha (Zipcode) prices."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)

# Cache for prices to avoid excessive API/RPC calls
_price_cache: dict[str, float] = {}


async def get_tao_price_usd() -> float:
    """Fetch TAO price from MEXC or CoinGecko."""
    for url, key in [
        ("https://api.mexc.com/api/v3/ticker/price?symbol=TAOUSDT", "price"),
        (
            "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd",
            "bittensor",
        ),
    ]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                res.raise_for_status()
                data = res.json()
                price = float(data[key] if key != "bittensor" else data[key]["usd"])
                if price <= 0:
                    logger.debug(f"Invalid price {price} from {url}")
                    continue
                _price_cache["tao_usd"] = price
                return price
        except Exception as e:
            logger.debug(f"Failed to fetch TAO price from {url}: {e}")

    return _price_cache.get("tao_usd", 0.0)


# A dTAO subnet's alpha price is a pool ratio strictly below 1 TAO. A value at or
# above 1.0 is not a real price — this function used to hard-code 1.0 on failure,
# which valued alpha ~166x too high and made the burn cap over-burn and starve
# miners. Treat anything outside (0, 1) as "unknown" and return 0.0 so callers can
# refuse to burn on it.
_MAX_PLAUSIBLE_ALPHA_TAO = 1.0


def _plausible_alpha(p: float) -> bool:
    return 0.0 < p < _MAX_PLAUSIBLE_ALPHA_TAO


def _scale_int(result: object) -> int:
    """Pull the integer out of a substrate query result (ScaleType / list / scalar)."""
    v = getattr(result, "value", result)
    if isinstance(v, (list, tuple)):
        v = v[0]
    return int(v)


async def get_alpha_price_tao(subtensor: bt.subtensor, netuid: int) -> float:
    """Alpha price in TAO, from the AMM pool reserves in storage: SubnetTAO /
    SubnetAlphaIn. Both are in rao, so the units cancel and the ratio is TAO per
    alpha directly. We read storage rather than the SwapRuntimeApi runtime call
    (`get_subnet_price`) because that runtime call fails on a chain-runtime/SDK
    version mismatch, whereas storage reads keep working.

    Returns 0.0 when no trustworthy price is available (query failed or the value
    is implausible). Callers MUST treat 0.0 as 'unknown' and NOT burn on it: a
    bogus non-zero price silently over-burns and starves miners.
    """
    try:
        tao_rao = _scale_int(subtensor.query_subtensor("SubnetTAO", params=[netuid]))
        alpha_rao = _scale_int(
            subtensor.query_subtensor("SubnetAlphaIn", params=[netuid])
        )
        if alpha_rao > 0:
            p = tao_rao / alpha_rao
            if _plausible_alpha(p):
                _price_cache[f"alpha_{netuid}"] = p
                return p
    except Exception as e:
        logger.warning(f"Alpha price fetch failed: {e}")
    # No trustworthy live value — use the last good cached price, else 0.0 (unknown).
    return _price_cache.get(f"alpha_{netuid}", 0.0)
