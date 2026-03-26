"""Subtensor connection resilience utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)


def patch_subtensor_reconnect(subtensor: bt.subtensor) -> None:
    """Monkey-patch subtensor to auto-reconnect on stale WebSocket.

    The bittensor substrate client keeps a persistent WebSocket connection.
    If the remote node closes it (timeout, restart, etc.), subsequent calls
    fail with BrokenPipeError or similar. This patch checks the socket state
    before each RPC call and reconnects if necessary.
    """
    try:
        from websockets.client import OPEN as WS_OPEN
    except ImportError:
        logger.warning("websockets not installed, skipping WS reconnect patch")
        return

    substrate = subtensor.substrate
    orig_connect = substrate.connect

    def _reconnecting_connect(*args: Any, **kwargs: Any) -> Any:
        current_ws = getattr(substrate, "ws", None)
        if current_ws is not None and current_ws.state == WS_OPEN:
            return current_ws

        logger.warning("Subtensor WebSocket not OPEN — reconnecting")
        try:
            new_ws = orig_connect(*args, **kwargs)
        except Exception:
            logger.error("Failed to reconnect subtensor WebSocket", exc_info=True)
            raise
        setattr(substrate, "ws", new_ws)
        return new_ws

    substrate.connect = _reconnecting_connect
    logger.info("Subtensor WebSocket auto-reconnect enabled")
