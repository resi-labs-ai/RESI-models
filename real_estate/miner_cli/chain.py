"""Chain utilities for miner CLI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ExtrinsicNotFoundError

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtrinsicInfo:
    """Information about a found extrinsic."""

    block_number: int
    extrinsic_index: int

    @property
    def extrinsic_id(self) -> str:
        """Return extrinsic ID in block-index format."""
        return f"{self.block_number}-{self.extrinsic_index}"

    def to_record_dict(self, hotkey: str) -> dict:
        """
        Build extrinsic_record.json content.

        This matches the format expected by validators in ModelVerifier.
        Only includes the two required fields: extrinsic and hotkey.
        """
        return {
            "extrinsic": self.extrinsic_id,
            "hotkey": hotkey,
        }


def scan_for_commitment_extrinsic(
    subtensor: bt.subtensor,
    signer_hotkey: str,
    start_block: int,
    max_blocks: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> ExtrinsicInfo:
    """
    Scan blocks to find the commitment extrinsic.

    After submitting a commitment, it lands in one of the next few blocks.
    This function scans forward from start_block to find it.

    Args:
        subtensor: Bittensor subtensor instance.
        signer_hotkey: SS58 address of the signer to match.
        start_block: Block number to start scanning from.
        max_blocks: Maximum number of blocks to scan forward.
        on_progress: Optional callback(current_block, blocks_scanned) for progress.

    Returns:
        ExtrinsicInfo with block number and index.

    Raises:
        ExtrinsicNotFoundError: If extrinsic not found within scan range.
    """
    logger.debug(
        f"Scanning for commitment extrinsic from block {start_block}, "
        f"max {max_blocks} blocks"
    )

    for offset in range(max_blocks):
        block_num = start_block + offset

        if on_progress:
            on_progress(block_num, offset + 1)

        result = _check_block_for_commitment(subtensor, block_num, signer_hotkey)

        if result is not None:
            logger.debug(f"Found commitment at {result.extrinsic_id}")
            return result

    raise ExtrinsicNotFoundError(
        f"Commitment extrinsic not found in {max_blocks} blocks "
        f"starting from {start_block}. You can look it up manually on a block explorer."
    )


def _check_block_for_commitment(
    subtensor: bt.subtensor,
    block_num: int,
    signer_hotkey: str,
) -> ExtrinsicInfo | None:
    """
    Check a single block for a commitment extrinsic from the signer.

    Returns:
        ExtrinsicInfo if found, None otherwise.
    """
    try:
        block_hash = subtensor.get_block_hash(block_num)
        if not block_hash:
            return None

        block = subtensor.substrate.get_block(block_hash)
        if not block or "extrinsics" not in block:
            return None

        for idx, extrinsic in enumerate(block["extrinsics"]):
            if _is_matching_commitment(extrinsic, signer_hotkey):
                return ExtrinsicInfo(block_number=block_num, extrinsic_index=idx)

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        # Block state may be pruned on archive-less nodes
        if "State discarded" in error_str or "StateDiscardedError" in error_type:
            logger.debug(f"Block {block_num} state discarded (pruned)")
        else:
            logger.debug(f"Error scanning block {block_num}: {e}")

    return None


def _is_matching_commitment(extrinsic, signer_hotkey: str) -> bool:
    """Check if extrinsic is a commitment from the expected signer."""
    try:
        if not hasattr(extrinsic, "value") or not extrinsic.value:
            return False

        if not extrinsic.signed:
            return False

        ext_value = extrinsic.value
        call = ext_value.get("call")
        if not call:
            return False

        # Check it's a commitment call
        if call.get("call_module") != "Commitments":
            return False
        if call.get("call_function") != "set_commitment":
            return False

        # Check signer matches
        return ext_value.get("address") == signer_hotkey

    except Exception:
        return False
