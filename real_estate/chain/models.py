"""Type-safe chain data models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ChainModelMetadata:
    """
    Model metadata from chain commitment.

    Immutable to prevent accidental modification.
    """

    hotkey: str
    hf_repo_id: str
    model_hash: str  # SHA-256 hash (64 chars)
    block_number: int
    timestamp: int  # Unix timestamp

    def is_committed_before(self, block: int) -> bool:
        """Check if model was committed before a specific block."""
        return self.block_number < block

    @classmethod
    def from_commitment_data(
        cls, hotkey: str, data: dict[str, Any]
    ) -> ChainModelMetadata:
        """
        Parse commitment data dict into ChainModelMetadata.

        Expected format (compact JSON):
        {
            "h": "abc123...def456",  # model hash (64 chars SHA-256)
            "r": "user/model",       # HF repo
            "v": "1.0.0",            # version (optional)
            "t": 1700000000          # timestamp
        }
        """
        return cls(
            hotkey=hotkey,
            hf_repo_id=data.get("r", ""),
            model_hash=data.get("h", ""),
            block_number=data.get("b", 0),
            timestamp=data.get("t", 0),
        )

    @classmethod
    def from_hex(
        cls, hotkey: str, hex_data: str, block_number: int = 0
    ) -> ChainModelMetadata:
        """
        Parse hex-encoded commitment data.

        Args:
            hotkey: Miner's hotkey address
            hex_data: Hex-encoded JSON commitment data
            block_number: Block number where commitment was made
        """
        # Decode hex to bytes, then to string, then parse JSON
        hex_str = hex_data[2:] if hex_data.startswith("0x") else hex_data
        raw_bytes = bytes.fromhex(hex_str)
        json_str = raw_bytes.decode("utf-8")
        data = json.loads(json_str)

        return cls(
            hotkey=hotkey,
            hf_repo_id=data.get("r", ""),
            model_hash=data.get("h", ""),
            block_number=block_number or data.get("b", 0),
            timestamp=data.get("t", 0),
        )


@dataclass(frozen=True)
class Commitment:
    """Raw chain commitment data from Pylon."""

    hotkey: str
    data: str  # Hex-encoded commitment data
    block: int

    def decode(self) -> dict[str, Any]:
        """Decode hex data to dictionary."""
        hex_data = self.data[2:] if self.data.startswith("0x") else self.data
        raw_bytes = bytes.fromhex(hex_data)
        json_str = raw_bytes.decode("utf-8")
        result: dict[str, Any] = json.loads(json_str)
        return result

    def to_metadata(self) -> ChainModelMetadata:
        """Convert to ChainModelMetadata."""
        return ChainModelMetadata.from_hex(self.hotkey, self.data, self.block)


@dataclass(frozen=True)
class Neuron:
    """
    Neuron data from metagraph.

    Represents a registered participant on the subnet.
    """

    uid: int
    hotkey: str
    coldkey: str
    stake: float
    trust: float
    consensus: float
    incentive: float
    dividends: float
    emission: float
    is_active: bool

    @classmethod
    def from_pylon_response(cls, uid: int, data: dict[str, Any]) -> Neuron:
        """Parse Pylon neuron response into Neuron object."""
        return cls(
            uid=uid,
            hotkey=data.get("hotkey", ""),
            coldkey=data.get("coldkey", ""),
            stake=float(data.get("stake", 0)),
            trust=float(data.get("trust", 0)),
            consensus=float(data.get("consensus", 0)),
            incentive=float(data.get("incentive", 0)),
            dividends=float(data.get("dividends", 0)),
            emission=float(data.get("emission", 0)),
            is_active=data.get("active", False),
        )


@dataclass(frozen=True)
class ExtrinsicCall:
    """Call data within an extrinsic."""

    call_module: str
    call_function: str
    call_args: list[dict[str, Any]]


@dataclass(frozen=True)
class ExtrinsicData:
    """
    Decoded extrinsic data from chain.

    Contains address and call details from a specific block extrinsic.
    """

    block_number: int
    extrinsic_index: int
    extrinsic_hash: str
    extrinsic_length: int
    address: str | None  # Signer address (None for unsigned extrinsics)
    call: ExtrinsicCall

    def is_commitment_extrinsic(self) -> bool:
        """Check if this extrinsic is a commitment call."""
        return (
            self.call.call_module.lower() == "commitments"
            and self.call.call_function.lower() == "set_commitment"
        )


@dataclass
class Metagraph:
    """
    Metagraph snapshot from chain.

    Contains all neurons and their current state.
    """

    block: int
    neurons: list[Neuron]
    timestamp: datetime

    @property
    def hotkeys(self) -> list[str]:
        """Get all hotkeys in the metagraph."""
        return [n.hotkey for n in self.neurons]

    @property
    def uids(self) -> list[int]:
        """Get all UIDs in the metagraph."""
        return [n.uid for n in self.neurons]

    def get_uid(self, hotkey: str) -> int | None:
        """Get UID for a hotkey."""
        for neuron in self.neurons:
            if neuron.hotkey == hotkey:
                return neuron.uid
        return None

    def get_neuron(self, hotkey: str) -> Neuron | None:
        """Get neuron by hotkey."""
        for neuron in self.neurons:
            if neuron.hotkey == hotkey:
                return neuron
        return None

    @classmethod
    def from_pylon_response(cls, data: dict[str, Any]) -> Metagraph:
        """
        Parse Pylon neurons response into Metagraph.

        Note: netuid is configured at Pylon service level, not in response.
        """
        block_data = data.get("block", {})
        neurons_data = data.get("neurons", {})

        # neurons dict is keyed by hotkey, uid is inside neuron_data
        neurons = [
            Neuron.from_pylon_response(neuron_data.get("uid", 0), neuron_data)
            for neuron_data in neurons_data.values()
        ]

        return cls(
            block=block_data.get("number", 0),
            neurons=neurons,
            timestamp=datetime.now(UTC),
        )
