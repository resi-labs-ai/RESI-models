"""Chain interaction utilities for miner CLI."""

import json

import bittensor as bt

from .config import SCAN_MAX_BLOCKS, SCAN_MAX_EXTRINSICS_PER_BLOCK


def build_commitment(model_hash: str, hf_repo_id: str) -> str:
    """Build commitment JSON: {"h": hash, "r": repo_id}."""
    return json.dumps({"h": model_hash, "r": hf_repo_id}, separators=(",", ":"))


def scan_for_extrinsic_id(
    subtensor: bt.subtensor,
    signer_hotkey: str,
    start_block: int,
    max_blocks: int = SCAN_MAX_BLOCKS,
    max_per_block: int = SCAN_MAX_EXTRINSICS_PER_BLOCK,
) -> str | None:
    """
    Scan blocks for a commitment extrinsic matching the signer.

    Args:
        subtensor: Bittensor subtensor instance.
        signer_hotkey: SS58 address of the signer to match.
        start_block: Block number to start scanning from.
        max_blocks: Maximum number of blocks to scan forward.
        max_per_block: Maximum extrinsic indices to check per block.

    Returns:
        Extrinsic ID as "{block}-{index}" or None if not found.
    """
    bt.logging.info(f"Scanning {max_blocks} blocks starting from {start_block}...")

    for block_offset in range(max_blocks):
        block_num = start_block + block_offset

        try:
            block_hash = subtensor.get_block_hash(block_num)
            if not block_hash:
                continue

            block = subtensor.substrate.get_block(block_hash)
            if not block or "extrinsics" not in block:
                continue

            extrinsics = block["extrinsics"]
            for idx, extrinsic in enumerate(extrinsics[:max_per_block]):
                try:
                    if not hasattr(extrinsic, "value") or not extrinsic.value:
                        continue

                    ext_value = extrinsic.value

                    if not extrinsic.signed:
                        continue

                    call = ext_value.get("call")
                    if not call:
                        continue

                    call_module = call.get("call_module", "")
                    call_function = call.get("call_function", "")

                    if (
                        call_module == "Commitments"
                        and call_function == "set_commitment"
                    ):
                        extrinsic_address = ext_value.get("address")
                        if extrinsic_address == signer_hotkey:
                            bt.logging.success(
                                f"Found commitment at block {block_num}, extrinsic {idx}"
                            )
                            return f"{block_num}-{idx}"

                except Exception as e:
                    bt.logging.debug(
                        f"Error processing extrinsic {idx} in block {block_num}: {e}"
                    )
                    continue

        except Exception as e:
            if "State discarded" in str(e) or "StateDiscardedError" in str(
                type(e).__name__
            ):
                bt.logging.warning(f"Block {block_num} too old (state discarded)")
                break
            else:
                bt.logging.warning(f"Error scanning block {block_num}: {e}")
                continue

    bt.logging.error(f"Commitment not found in {max_blocks} blocks")
    return None
