"""
RESI Miner CLI - Evaluate and submit ONNX models to the Real Estate Subnet.

Usage:
    miner-cli evaluate --model.path ./model.onnx
    miner-cli submit --model.path ./model.onnx --hf.repo_id user/repo \\
        --wallet.name miner --wallet.hotkey default
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ExtrinsicNotFoundError, MinerCLIError

LICENSE_NOTICE = """
License Notice:
Your HuggingFace model repository must include a LICENSE file.
Validators verify this before evaluating your model.
"""


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    from .evaluate import evaluate_model

    print(f"Evaluating model: {args.model_path}")
    print()

    result = evaluate_model(
        model_path=args.model_path,
        max_size_mb=args.max_size_mb,
    )

    if not result.success:
        print(f"ERROR: Evaluation failed: {result.error_message}", file=sys.stderr)
        return 1

    # Display results
    metrics = result.metrics
    if metrics is None:
        print("ERROR: Evaluation succeeded but metrics are missing", file=sys.stderr)
        return 1

    print("Evaluation Results:")
    print(f"  MAPE:  {metrics.mape:.2%}")
    print(f"  Score: {metrics.score:.4f}")
    print(f"  MAE:   ${metrics.mae:,.0f}")
    print(f"  RMSE:  ${metrics.rmse:,.0f}")
    print(f"  R²:    {metrics.r2:.4f}")
    print()

    if result.inference_time_ms:
        print(f"Inference time: {result.inference_time_ms:.0f}ms")
        print()

    print("✓ Model is valid and ready for submission.")

    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Execute the submit command."""
    # Import bittensor only when needed (after argparse)
    import bittensor as bt

    from .submit import find_commitment_extrinsic, submit_model

    print(LICENSE_NOTICE)
    print("Submitting model to chain...")
    print()

    # Determine netuid from network if not specified
    if args.netuid is not None:
        netuid = args.netuid
    elif args.network in ("finney", "mainnet"):
        netuid = 46
    elif args.network in ("test", "testnet"):
        netuid = 428
    else:
        # Custom endpoint - require explicit netuid
        print(
            "ERROR: Custom network endpoint requires --netuid to be specified.",
            file=sys.stderr,
        )
        return 2

    # Initialize wallet and subtensor
    wallet = bt.wallet(
        name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path
    )
    subtensor = bt.subtensor(network=args.network)
    hotkey_ss58 = wallet.hotkey.ss58_address

    try:
        result = submit_model(
            model_path=args.model_path,
            hf_repo_id=args.hf_repo_id,
            wallet=wallet,
            subtensor=subtensor,
            netuid=netuid,
            commit_reveal=args.commit_reveal,
            blocks_until_reveal=args.reveal_blocks,
        )
    except MinerCLIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not result.success:
        print(f"ERROR: Submission failed: {result.error_message}", file=sys.stderr)
        return 1

    print()
    if result.commit_reveal:
        print("✓ Model committed to chain with commit-reveal!")
    else:
        print("✓ Model committed to chain successfully!")
    print()
    print("Commitment details:")
    print(f"  Repository:         {result.hf_repo_id}")
    print(f"  Model hash:         {result.model_hash}")
    print(f"  Submitted at block: {result.submitted_at_block}")
    if result.commit_reveal and result.reveal_round:
        print(
            f"  Commit-reveal:      Yes (reveal at drand round {result.reveal_round})"
        )

    # Scan for extrinsic ID (unless skipped)
    extrinsic_info = None
    if args.skip_scan:
        print()
        print("Skipping extrinsic scan (--skip-scan).")
        print("You can find your extrinsic ID manually on a block explorer.")
    elif result.submitted_at_block is None:
        print()
        print("Warning: Block number not available, skipping extrinsic scan.")
    else:
        print()
        print(f"Scanning for extrinsic (up to {args.scan_blocks} blocks)...")

        def on_progress(block_num: int, blocks_scanned: int) -> None:
            # Overwrite line with progress
            print(
                f"\r  Scanning block {block_num} ({blocks_scanned}/{args.scan_blocks})...",
                end="",
                flush=True,
            )

        try:
            extrinsic_info = find_commitment_extrinsic(
                subtensor=subtensor,
                hotkey_ss58=hotkey_ss58,
                start_block=result.submitted_at_block,
                max_blocks=args.scan_blocks,
                on_progress=on_progress,
            )
            # Clear progress line and show success
            print(f"\r  Found extrinsic: {extrinsic_info.extrinsic_id}              ")
        except ExtrinsicNotFoundError as e:
            # Clear progress line and show warning
            print(f"\r  Warning: {e}              ", file=sys.stderr)

    print()
    print("Next steps:")
    print("  1. Ensure model.onnx is uploaded to your HuggingFace repo")
    print("  2. Ensure the model repo is having the proper license")

    if extrinsic_info:
        print("  3. Add extrinsic_record.json to your repo with this content:")
        print()
        record = extrinsic_info.to_record_dict(hotkey_ss58)
        print(json.dumps(record, indent=2))
    else:
        print("  3. Find your extrinsic ID and add extrinsic_record.json to your repo")

    print()
    print("  4. Wait for validator evaluation")

    return 0


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="miner-cli",
        description="RESI Miner CLI - Evaluate and submit ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to execute",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATE command
    # ─────────────────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an ONNX model locally",
        description="Validate model interface and run inference on test samples.",
    )

    eval_parser.add_argument(
        "--model.path",
        dest="model_path",
        required=True,
        metavar="PATH",
        help="Path to ONNX model file",
    )

    eval_parser.add_argument(
        "--max-size-mb",
        dest="max_size_mb",
        type=int,
        default=200,
        metavar="MB",
        help="Maximum model size in MB (default: 200)",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SUBMIT command
    # ─────────────────────────────────────────────────────────────────────────
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit model commitment to chain",
        description="Hash local model and commit to Bittensor chain.",
    )

    # Model arguments
    submit_parser.add_argument(
        "--model.path",
        dest="model_path",
        required=True,
        metavar="PATH",
        help="Path to local ONNX model file (will be hashed)",
    )

    submit_parser.add_argument(
        "--hf.repo_id",
        dest="hf_repo_id",
        required=True,
        metavar="USER/REPO",
        help="HuggingFace repository ID where model is uploaded",
    )

    # Wallet arguments
    submit_parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        required=True,
        metavar="NAME",
        help="Bittensor wallet name",
    )

    submit_parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        required=True,
        metavar="HOTKEY",
        help="Bittensor wallet hotkey",
    )

    submit_parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        default="~/.bittensor/wallets",
        metavar="PATH",
        help="Path to wallet directory (default: ~/.bittensor/wallets)",
    )

    # Network arguments
    submit_parser.add_argument(
        "--network",
        default="finney",
        metavar="NETWORK",
        help="Network: finney, test, or ws:// endpoint URL (default: finney)",
    )

    submit_parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        metavar="UID",
        help="Subnet UID (default: 46 for finney, 428 for test, required for custom endpoints)",
    )

    # Extrinsic scanning arguments
    submit_parser.add_argument(
        "--skip-scan",
        dest="skip_scan",
        action="store_true",
        help="Skip scanning for extrinsic ID after submission",
    )

    submit_parser.add_argument(
        "--scan-blocks",
        dest="scan_blocks",
        type=int,
        default=25,
        metavar="N",
        help="Maximum blocks to scan for extrinsic (default: 25)",
    )

    # Commit-reveal arguments
    submit_parser.add_argument(
        "--no-commit-reveal",
        dest="commit_reveal",
        action="store_false",
        default=True,
        help="Disable timelock commit-reveal (commitment visible immediately)",
    )

    submit_parser.add_argument(
        "--reveal-blocks",
        dest="reveal_blocks",
        type=int,
        default=360,
        metavar="N",
        help="Blocks until commitment reveal (default: 360, ~1 epoch/72 min)",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI entry point."""
    # Parse args BEFORE importing bittensor to prevent argparse hijacking
    config = parse_args(args)

    try:
        if config.command == "evaluate":
            return cmd_evaluate(config)
        elif config.command == "submit":
            return cmd_submit(config)
        else:
            print(f"ERROR: Unknown command: {config.command}", file=sys.stderr)
            return 2

    except MinerCLIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
