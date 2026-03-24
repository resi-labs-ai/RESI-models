"""Data models for decentralized randomness module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RandomnessConfig:
    """Configuration for decentralized randomness seed generation."""

    commit_hours_before_eval: float = 4.0
    """When to commit random value (hours before scheduled evaluation)."""

    blocks_until_reveal: int = 360
    """Drand timelock duration in blocks (~72 min at 12s/block = 1 epoch)."""

    seed_modulus: int = 2**32
    """Seed range [0, modulus). Matches numpy default_rng seed range."""


@dataclass(frozen=True)
class SeedResult:
    """Result of combining revealed validator commitments into a shared seed."""

    seed: int
    """Combined deterministic seed."""

    num_reveals: int
    """How many validators contributed reveals."""

    validator_hotkeys: frozenset[str]
    """Which validators contributed reveals."""
