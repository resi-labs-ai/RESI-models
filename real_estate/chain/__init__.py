"""Chain interaction layer via Pylon."""

from .errors import (
    AuthenticationError,
    ChainConnectionError,
    ChainError,
    CommitmentError,
    MetagraphError,
    WeightSettingError,
)
from .models import (
    ChainModelMetadata,
    Commitment,
    ExtrinsicCall,
    ExtrinsicData,
    Metagraph,
    Neuron,
)


def __getattr__(name: str):
    """Lazy import ChainClient and PylonConfig to avoid pylon dependency at import time."""
    if name in ("ChainClient", "PylonConfig"):
        from .client import ChainClient, PylonConfig

        return ChainClient if name == "ChainClient" else PylonConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client (lazy loaded)
    "ChainClient",
    "PylonConfig",
    # Errors
    "AuthenticationError",
    "ChainConnectionError",
    "ChainError",
    "CommitmentError",
    "MetagraphError",
    "WeightSettingError",
    # Models
    "ChainModelMetadata",
    "Commitment",
    "ExtrinsicCall",
    "ExtrinsicData",
    "Metagraph",
    "Neuron",
]
