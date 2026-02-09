"""
Model submission to Bittensor chain.

Handles:
- Model hashing (using shared verifier utility)
- Commitment building
- Chain submission via bittensor SDK
- Extrinsic scanning (optional, for extrinsic_record.json)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..models.verifier import ModelVerifier
from .chain import ExtrinsicInfo, scan_for_commitment_extrinsic
from .config import MAX_REPO_ID_BYTES
from .errors import (
    CommitmentError,
    HotkeyNotRegisteredError,
    ModelNotFoundError,
)
from .models import SubmitResult

if TYPE_CHECKING:
    import bittensor as bt

logger = logging.getLogger(__name__)


def build_commitment(model_hash: str, hf_repo_id: str) -> str:
    """
    Build commitment JSON for chain submission.

    Format: {"h": "<sha256_hash>", "r": "<hf_repo_id>"}

    Args:
        model_hash: SHA-256 hash of model file (64 chars).
        hf_repo_id: HuggingFace repository ID.

    Returns:
        Compact JSON string.
    """
    return json.dumps({"h": model_hash, "r": hf_repo_id}, separators=(",", ":"))


def submit_model(
    model_path: str | Path,
    hf_repo_id: str,
    wallet: bt.wallet,
    subtensor: bt.subtensor,
    netuid: int,
    *,
    commit_reveal: bool = False,
    blocks_until_reveal: int = 360,
) -> SubmitResult:
    """
    Submit a model commitment to the Bittensor chain.

    This function:
    1. Validates the local model file exists
    2. Validates the HF repo ID length
    3. Checks hotkey registration
    4. Computes model hash
    5. Submits commitment to chain

    Args:
        model_path: Path to local ONNX model file.
        hf_repo_id: HuggingFace repository ID where model is uploaded.
        wallet: Bittensor wallet instance.
        subtensor: Bittensor subtensor instance.
        netuid: Subnet UID.
        commit_reveal: If True, use timelock-encrypted commit-reveal.
            The commitment will be hidden until the reveal round.
        blocks_until_reveal: Blocks until commitment is revealed (default: 360).
            Only used when commit_reveal=True. 360 blocks = 1 epoch ≈ 72 min.

    Returns:
        SubmitResult with commitment details (includes reveal_round if commit_reveal).

    Note:
        The same model file must be uploaded to the HF repo at model.onnx.
        Validators will download from HF and verify the hash matches.
    """

    model_path = Path(model_path)

    # Step 1: Validate model file exists
    if not model_path.exists():
        raise ModelNotFoundError(f"Model file not found: {model_path}")

    # Step 2: Validate repo ID length (chain constraint)
    repo_bytes = len(hf_repo_id.encode("utf-8"))
    if repo_bytes > MAX_REPO_ID_BYTES:
        raise CommitmentError(
            f"HF repo ID too long: {repo_bytes} bytes exceeds {MAX_REPO_ID_BYTES} byte limit"
        )

    # Step 3: Check hotkey registration
    hotkey_ss58 = wallet.hotkey.ss58_address
    logger.debug(f"Wallet: {wallet.name}, Hotkey: {wallet.hotkey_str}")
    logger.debug(f"Hotkey address: {hotkey_ss58}")
    logger.debug(f"Network: {subtensor.network}")

    if not subtensor.is_hotkey_registered(
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        block=None,
    ):
        raise HotkeyNotRegisteredError(
            f"Hotkey {hotkey_ss58} is not registered on subnet {netuid}. "
            "Run `btcli subnets register` first."
        )

    logger.debug(f"Hotkey registered on subnet {netuid}")

    # Step 4: Compute model hash
    model_hash = ModelVerifier.compute_hash(model_path)
    logger.debug(f"Model hash: {model_hash}")

    # Step 5: Build and submit commitment
    commitment = build_commitment(model_hash, hf_repo_id)
    logger.debug(f"Commitment: {commitment}")

    # Note: This is the block at submission time, not the block where the
    # commitment is actually included (which may be 1-2 blocks later).
    current_block = subtensor.get_current_block()
    logger.debug(f"Current block: {current_block}")

    reveal_round: int | None = None

    try:
        if commit_reveal:
            logger.debug(
                f"Using commit-reveal with {blocks_until_reveal} blocks until reveal"
            )
            success, reveal_round = subtensor.set_reveal_commitment(
                wallet=wallet,
                netuid=netuid,
                data=commitment,
                blocks_until_reveal=blocks_until_reveal,
            )
            if not success:
                raise CommitmentError("set_reveal_commitment returned False")
            logger.debug(f"Commitment submitted with reveal_round={reveal_round}")
        else:
            subtensor.commit(wallet, netuid, commitment)
            logger.debug("Commitment submitted to chain")
    except Exception as e:
        raise CommitmentError(f"Failed to submit commitment: {e}") from e

    return SubmitResult(
        model_path=str(model_path),
        hf_repo_id=hf_repo_id,
        model_hash=model_hash,
        success=True,
        submitted_at_block=current_block,
        commit_reveal=commit_reveal,
        reveal_round=reveal_round,
    )


def find_commitment_extrinsic(
    subtensor: bt.subtensor,
    hotkey_ss58: str,
    start_block: int,
    max_blocks: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> ExtrinsicInfo:
    """
    Find the commitment extrinsic on chain after submission.

    This scans blocks starting from start_block to find the commitment
    extrinsic signed by the given hotkey. Use this after submit_model()
    to get the extrinsic ID for extrinsic_record.json.

    Args:
        subtensor: Bittensor subtensor instance.
        hotkey_ss58: SS58 address of the signer.
        start_block: Block number to start scanning from.
        max_blocks: Maximum blocks to scan (default: 25, ~5 min).
        on_progress: Optional callback(block_num, blocks_scanned) for progress.

    Returns:
        ExtrinsicInfo with block number and index.

    Raises:
        ExtrinsicNotFoundError: If not found within scan range.
    """
    return scan_for_commitment_extrinsic(
        subtensor=subtensor,
        signer_hotkey=hotkey_ss58,
        start_block=start_block,
        max_blocks=max_blocks,
        on_progress=on_progress,
    )
