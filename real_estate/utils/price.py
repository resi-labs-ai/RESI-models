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


async def get_alpha_price_tao(subtensor: bt.subtensor, netuid: int) -> float:
    """Fetch Alpha price in TAO from subtensor."""
    try:
        # Standard SDK method for dTAO price
        if hasattr(subtensor, "get_subnet_price"):
            price = subtensor.get_subnet_price(netuid)
            if price:
                p = float(price.tao)
                if p > 0:
                    _price_cache[f"alpha_{netuid}"] = p
                    return p

        # Fallback to info
        info = subtensor.get_subnet_info(netuid)
        price = float(getattr(info, "price", 1.0))
        _price_cache[f"alpha_{netuid}"] = price
        return price
    except Exception as e:
        logger.debug(f"Alpha price fetch failed: {e}")
        return _price_cache.get(f"alpha_{netuid}", 1.0)
