"""Data models for model management module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedModelMetadata:
    """
    Minimal metadata stored alongside cached model file.

    Only stores what's needed for cache invalidation and disk tracking.
    Full commitment data comes from chain via Pylon (already cached there).
    """

    hash: str  # SHA-1 prefix (8 chars) - compared with chain commitment
    size_bytes: int  # For disk space tracking

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hash": self.hash,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CachedModelMetadata:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            hash=data["hash"],
            size_bytes=data["size_bytes"],
        )


@dataclass(frozen=True)
class CachedModel:
    """
    A cached model with its metadata.

    Returned from cache lookup operations.
    """

    path: Path
    metadata: CachedModelMetadata


@dataclass(frozen=True)
class ExtrinsicRecord:
    """
    Parsed extrinsic_record.json from HuggingFace repo.

    This file must be created by the miner after submitting their model
    commitment on-chain. Contains hotkey and extrinsic ID for verification.
    """

    hotkey: str
    extrinsic: str  # Format: "{block}-{index}"

    @property
    def block_number(self) -> int:
        """Extract block number from extrinsic ID."""
        return int(self.extrinsic.split("-")[0])

    @property
    def extrinsic_index(self) -> int:
        """
        Extract extrinsic index from extrinsic ID.

        Handles both decimal and hex formats (0x prefix).
        """
        idx_str = self.extrinsic.split("-")[1]
        base = 16 if idx_str.lower().startswith("0x") else 10
        return int(idx_str, base)

    @classmethod
    def from_dict(cls, data: dict) -> ExtrinsicRecord:
        """Create from dictionary (JSON from HF repo)."""
        return cls(
            hotkey=data["hotkey"],
            extrinsic=data["extrinsic"],
        )


@dataclass
class DownloadResult:
    """
    Result of a model download attempt.

    Used by scheduler to track download outcomes.
    """

    hotkey: str
    success: bool
    path: Path | None = None
    error: Exception | None = None

    @property
    def error_message(self) -> str | None:
        """Get error message if failed."""
        if self.error:
            return f"{type(self.error).__name__}: {self.error}"
        return None
