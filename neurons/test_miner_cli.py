#!/usr/bin/env python3
"""
RESI Miner CLI - Evaluate and submit ONNX models to the RESI subnet.

Usage:
    python neurons/test_miner_cli.py --action evaluate --model_path ./model.onnx
    python neurons/test_miner_cli.py --action submit --hf_repo_id user/repo --wallet_name miner --wallet_hotkey default
"""
# Standard library
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

# Third-party
import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import huggingface_hub
import onnx

# Local imports
from scripts.compute_hash import compute_hash

# Constants
SCAN_MAX_BLOCKS = 20
SCAN_MAX_EXTRINSICS_PER_BLOCK = 100

LICENSE_NOTICE = """
License Notice:
Your HuggingFace model repository should be licensed under MIT.
"""

def build_commitment(model_hash: str, hf_repo_id: str, timestamp: int) -> dict:
    """
    Build RESI commitment dictionary.

    Args:
        model_hash: 8-char SHA-1 hash
        hf_repo_id: HuggingFace repo ID (e.g., "user/repo")
        timestamp: Unix timestamp
        
    Returns:
        dict: {"h": "...", "r": "...", "t": ...}
    """
    model_commitment = {"h": model_hash, "r": hf_repo_id, "t": timestamp}
    return model_commitment


def hex_encode_commitment(commitment: dict) -> str:
    """
    Hex-encode a commitment dictionary.
    
    Converts commitment to compact JSON, encodes as UTF-8, then to hex.
    
    Args:
        commitment: Commitment dictionary
        
    Returns:
        Hex-encoded string
    """
    commitment_json_str = json.dumps(commitment, separators=(",", ":"))
    commitment_json_bytes = commitment_json_str.encode("utf-8")
    commitment_hex = commitment_json_bytes.hex()
    return commitment_hex

def scan_for_extrinsic_id(
    subtensor: bt.subtensor,
    signer_hotkey: str,
    start_block: int,
    max_blocks: int = SCAN_MAX_BLOCKS,
    max_per_block: int = SCAN_MAX_EXTRINSICS_PER_BLOCK
) -> str | None:
    """
    Scan blocks for a commitment extrinsic matching the signer.
    
    Args:
        subtensor: Bittensor subtensor instance
        signer_hotkey: SS58 address of the signer to match
        start_block: Block number to start scanning from
        max_blocks: Maximum number of blocks to scan forward
        max_per_block: Maximum extrinsic indices to check per block
        
    Returns:
        Extrinsic ID as "{block}-{index}" or None if not found
    """
    bt.logging.info(f"Scanning {max_blocks} blocks starting from {start_block}...")
    
    for block_offset in range(max_blocks):
        block_num = start_block + block_offset
        
        try:
            block_hash = subtensor.get_block_hash(block_num)
            if not block_hash:
                continue

            block = subtensor.substrate.get_block(block_hash)
            if not block or 'extrinsics' not in block:
                continue
            
            extrinsics = block['extrinsics']
            for idx, extrinsic in enumerate(extrinsics[:max_per_block]):
                try:
                    if not hasattr(extrinsic, 'value') or not extrinsic.value:
                        continue
                    
                    ext_value = extrinsic.value
                    
                    if not extrinsic.signed:
                        continue
                    
                    call = ext_value.get('call')
                    if not call:
                        continue
                    
                    call_module = call.get('call_module', '')
                    call_function = call.get('call_function', '')
                    
                    if (call_module == 'Commitments' and 
                        call_function == 'set_commitment'):
                        
                        extrinsic_address = ext_value.get('address')
                        if extrinsic_address == signer_hotkey:
                            bt.logging.success(f"Found commitment at block {block_num}, extrinsic {idx}")
                            return f"{block_num}-{idx}"
                
                except Exception as e:
                    bt.logging.debug(f"Error processing extrinsic {idx} in block {block_num}: {e}")
                    continue
        
        except Exception as e:
            if "State discarded" in str(e) or "StateDiscardedError" in str(type(e).__name__):
                bt.logging.warning(f"Block {block_num} too old (state discarded)")
                break
            else:
                bt.logging.warning(f"Error scanning block {block_num}: {e}")
                continue
    
    bt.logging.error(f"Commitment not found in {max_blocks} blocks")
    return None

class MinerCLI:
    """
    MinerCLI class for evaluating and submitting miner models.
    """
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.wallet = bt.wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
        self.subtensor = bt.subtensor(network=config.subtensor_network)

    def is_onnx_model(self, model_path: str) -> bool:
        """Validate ONNX model format."""
        if not os.path.exists(model_path):
            bt.logging.error(f"Model file does not exist: {model_path}")
            return False
        try:
            onnx.checker.check_model(model_path)
            return True
        except onnx.checker.ValidationError as e:
            bt.logging.error(f"ONNX validation failed: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"Error checking ONNX model: {e}")
            return False

    async def submit_model(self) -> int:
        """
        Submit a model commitment to the chain.
            
        Returns:
            Exit code (0 = success, 1 = error, 2 = invalid args)
        """
        print(LICENSE_NOTICE)
        print("📤 Submitting model to chain...")    
        if not self.config.hf_repo_id:
            bt.logging.error("--hf_repo_id is required.")
            return 2

        # Validate byte lengths for chain compatibility
        if len(self.config.hf_repo_id.encode('utf-8')) > 64:
            bt.logging.error("hf_repo_id must be 64 bytes or less")
            return 2

        if len(self.config.hf_model_filename.encode('utf-8')) > 64:
            bt.logging.error("hf_model_filename must be 64 bytes or less")
            return 2
        
        # Validate wallet info/ Load wallet and subtensor
        bt.logging.info(f"Initialized Coldkey: {self.wallet.name}, Hotkey: {self.wallet.hotkey}")
        bt.logging.info(f"Hotkey Address: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"Network: {self.config.subtensor_network}")
        
        # Check registration
        if not self.subtensor.is_hotkey_registered(
            hotkey_ss58=self.wallet.hotkey.ss58_address,
            netuid=self.config.netuid,
            block=None
        ):
            bt.logging.error(f"Hotkey not registered on netuid {self.config.netuid}. Run `btcli subnets register` first")
            return 1
        
        bt.logging.success(f"Hotkey is registered on netuid {self.config.netuid}")
        
        # Download model from HuggingFace
        try:
            bt.logging.info(f"Downloading {self.config.hf_repo_id}/{self.config.hf_model_filename}...")
            model_path = hf_hub_download(repo_id=self.config.hf_repo_id, filename=self.config.hf_model_filename, token=self.config.hf_token)
            bt.logging.success(f"Successfully downloaded model to {model_path}")
        except Exception as e:
            bt.logging.error(f"Failed to download model: {e}")
            return 1
        
        if not self.is_onnx_model(model_path):
            bt.logging.error("Downloaded file is not a valid ONNX model")
            return 1

        # Compute hash
        try:
            model_hash = compute_hash(Path(model_path))
            bt.logging.info(f"Model hash: {model_hash}")
        except Exception as e:
            bt.logging.error(f"Failed to compute hash: {e}")
            return 1
        
        # Build commitment
        commitment = build_commitment(model_hash=model_hash, hf_repo_id=self.config.hf_repo_id, timestamp=int(time.time()))
        bt.logging.info(f"Commitment: {commitment}")
        
        # Hex-encode commitment
        commitment_hex = hex_encode_commitment(commitment)
        
        # Record current block
        current_block = self.subtensor.get_current_block()
        bt.logging.info(f"Current Block: {current_block}")

        # Submit to chain
        try:
            self.subtensor.commit(self.wallet, self.config.netuid, commitment_hex)
            bt.logging.success(f"Commitment submitted to chain")
        except Exception as e:
            bt.logging.error(f"Failed to submit commitment: {e}")
            return 1

        # Scan for extrinsic ID
        extrinsic_id = scan_for_extrinsic_id(
            subtensor=self.subtensor,
            signer_hotkey=self.wallet.hotkey.ss58_address,
            start_block=current_block,
            max_blocks=self.config.extrinsic_scan_blocks
        )
        
        # Print extrinsic ID for user
        if extrinsic_id:
            print("✅ SUCCESS! Model committed to chain.")
            print(f"\nExtrinsic ID: {extrinsic_id}")
            print(f"\nAdd this to your HuggingFace model card:")
            print(f"  Repository: {self.config.hf_repo_id}")
            print(f"  Extrinsic:  {extrinsic_id}")
        else:
            bt.logging.warning(
                "Commitment submitted but extrinsic ID not found. "
                "Check a block explorer manually."
            )
        return 0

    async def evaluate_model(self) -> int:
        """
        Evaluate an ONNX model locally with dummy data.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code (0 = success, 1 = error, 2 = invalid args)
        """
        print("🔍 Evaluating model locally...")
        
        # TODO:
        # 1. Check args.model_path exists
        # 2. Validate ONNX
        # 3. Run dummy inference
        # 4. Print results
        # 5. Return 0 on success, 1 on failure
        pass

    async def upload_to_hf(self) -> int:
        """
        Upload a model to Hugging Face.
        """
        pass

    async def main(self) -> int:
        """
        Main entry point.
        
        Returns:
            Exit code
        """
        
        # Dispatch to action
        if self.config.action == "evaluate":
            return await self.evaluate_model()
        elif self.config.action == "submit":
            return await self.submit_model()
        #elif self.config.action == "upload":
        #    return await self.upload_to_hf()
        else:
            print(f"ERROR: Unrecognized action: {self.config.action}")
            return 2

# MAIN & ARGUMENT PARSING
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="RESI Miner CLI - Evaluate and submit ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Action
    parser.add_argument(
        "--action",
        required=True,
        choices=["evaluate", "submit"],
        help="Action to perform"
    )
    
    # Evaluate arguments
    parser.add_argument(
        "--model_path",
        help="Local path to ONNX model (required for evaluate)"
    )
    
    # Submit arguments
    parser.add_argument(
        "--hf_repo_id",
        help='HuggingFace repo ID, e.g., "user/repo" (required for submit)'
    )
    parser.add_argument(
        "--hf_model_filename",
        default="model.onnx",
        help="Filename in HF repo (default: model.onnx)"
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)"
    )
    
    # Chain arguments
    parser.add_argument(
        "--subtensor_network",
        default="finney",
        help='Subtensor network: "finney", "test", or "local" (default: finney)'
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=1,
        help="Subnet UID (default: 1)"
    )
    
    # Wallet arguments
    parser.add_argument(
        "--wallet_name",
        default="miner",
        help="Bittensor wallet name (default: miner)"
    )
    parser.add_argument(
        "--wallet_hotkey",
        default="default",
        help="Bittensor wallet hotkey (default: default)"
    )
    
    # Extrinsic scanning
    parser.add_argument(
        "--extrinsic_scan_blocks",
        type=int,
        default=SCAN_MAX_BLOCKS,
        help=f"Max blocks to scan for extrinsic (default: {SCAN_MAX_BLOCKS})"
    )
    
    return parser.parse_args()

# SCRIPT EXECUTION
async def main() -> int:
    """CLI entry point."""
    config = parse_args()
    cli = MinerCLI(config)
    return await cli.main()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)