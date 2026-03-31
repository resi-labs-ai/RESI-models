"""
Decentralized randomness module for shared deterministic seeds.

Validators submit timelocked commitments to the chain, which are
auto-revealed after a timelock period. All validators then harvest
the revealed values and combine them into a shared seed.

Usage:
    from real_estate.randomness import (
        DecentralizedSeedProvider,
        RandomnessConfig,
        SeedResult,
    )

    provider = DecentralizedSeedProvider(subtensor, wallet, netuid)
    provider.commit()
    # ... wait for reveal ...
    result = provider.harvest(validator_hotkeys, min_reveal_block, committed_hotkeys)
    seed = result.seed  # Shared deterministic seed
"""

from .models import RandomnessConfig, SeedResult
from .provider import DecentralizedSeedProvider, combine_reveals

__all__ = [
    "DecentralizedSeedProvider",
    "RandomnessConfig",
    "SeedResult",
    "combine_reveals",
]
