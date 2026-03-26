"""Data models for decentralized randomness module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RandomnessConfig:
    """Configuration for decentralized randomness seed generation."""

    cycle_window_hours: float = 4.0
    """Time window (hours) before evaluation that bounds the randomness cycle.

    Used for: wall-clock fallback commit timing when epoch queries fail,
    freshness filter (reject reveals older than this), and snapshot TTL
    (discard committed-hotkeys snapshots older than this).
    """

    blocks_until_reveal: int = 360
    """Drand timelock duration in blocks (~72 min at 12s/block = 1 epoch)."""

    seed_modulus: int = 2**32
    """Seed range [0, modulus). Matches numpy default_rng seed range."""

    reveal_buffer_seconds: int = 300
    """Extra wait after expected reveal time for chain propagation (seconds)."""

    block_time_seconds: int = 12
    """Expected block production interval (seconds)."""

    def __post_init__(self) -> None:
        if self.seed_modulus <= 0:
            raise ValueError(f"seed_modulus must be positive, got {self.seed_modulus}")
        if self.blocks_until_reveal <= 0:
            raise ValueError(
                f"blocks_until_reveal must be positive, got {self.blocks_until_reveal}"
            )
        if self.block_time_seconds <= 0:
            raise ValueError(
                f"block_time_seconds must be positive, got {self.block_time_seconds}"
            )
        if self.cycle_window_hours <= 0:
            raise ValueError(
                f"cycle_window_hours must be positive, got {self.cycle_window_hours}"
            )
        if self.reveal_buffer_seconds < 0:
            raise ValueError(
                f"reveal_buffer_seconds must be non-negative, got {self.reveal_buffer_seconds}"
            )


@dataclass(frozen=True)
class SeedResult:
    """Result of combining revealed validator commitments into a shared seed."""

    seed: int
    """Combined deterministic seed."""

    num_reveals: int
    """How many validators contributed reveals."""

    validator_hotkeys: frozenset[str]
    """Which validators contributed reveals."""
