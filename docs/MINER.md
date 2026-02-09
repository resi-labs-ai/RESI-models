# Miner CLI Guide

Guide for miners on the Real Estate Price Prediction Subnet (Bittensor subnet 46).

## Overview

The miner CLI (`miner-cli`) helps you:
1. **Evaluate** your ONNX model locally before submission
2. **Submit** your model commitment to the Bittensor blockchain

**How it works:** You train a model to predict real estate prices, test it locally, upload it to HuggingFace, then submit a commitment to the chain. Validators download your model, verify it matches your commitment, and score it based on prediction accuracy.

## Model Requirements

| Requirement | Specification |
|-------------|---------------|
| Format | ONNX |
| Max size | 200 MB |
| Input shape | `(batch, 79)` float32 |
| Output shape | `(batch, 1)` or `(batch,)` float32 |

Your model must accept exactly **79 features** in the order defined in `real_estate/data/mappings/feature_config.yaml`.

## Prerequisites

- Python 3.11+
- Bittensor wallet with registered hotkey
- HuggingFace account
- TAO for subnet registration

## Installation

```bash
git clone https://github.com/resi-labs-ai/RESI-models.git
cd RESI-models

# Install with pip
pip install -e .

# Verify
miner-cli --help
```

Expected output:
```
usage: miner-cli [-h] {evaluate,submit} ...

RESI Miner CLI - Evaluate and submit ONNX models

positional arguments:
  {evaluate,submit}  Command to execute
    evaluate         Evaluate an ONNX model locally
    submit           Submit model commitment to chain
```

## Usage

### Step 1: Evaluate Your Model

```bash
miner-cli evaluate --model.path ./my_model.onnx
```

Check that metrics meet targets (MAPE < 15%, Score > 0.85).

### Step 2: Create HuggingFace Repository

1. Go to [huggingface.co](https://huggingface.co)
2. Create a new model repository
3. Upload your onnx model to the repository root

> **Note:** Your repository must be public when validators attempt to download your model.

### Step 3: Submit to Chain

```bash
miner-cli submit \
    --model.path ./my_model.onnx \
    --hf.repo_id your-username/your-repo \
    --wallet.name miner \
    --wallet.hotkey default
```

### Step 4: Complete HuggingFace Setup

After submitting, add these files to your HuggingFace repo:

1. **LICENSE** - Must use an MIT license (required for validator download)
2. **extrinsic_record.json** - Use values from submit output:

```json
{
  "extrinsic": "142858-3",
  "hotkey": "5ABC...your_hotkey_address...XYZ"
}
```

### Step 5: Wait for Validation

Validators will automatically download your model, verify the hash matches your commitment, and score it based on prediction accuracy.
If your repository is private when validation occurs, your model will not be scored.

## Commands Reference

### miner-cli evaluate

Validates your model and runs inference on test samples.

```bash
miner-cli evaluate --model.path PATH [--max-size-mb MB]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model.path` | Yes | - | Path to ONNX model file |
| `--max-size-mb` | No | 200 | Maximum model size in MB |

**What it checks:**
1. File exists and is under size limit
2. Valid ONNX format
3. Correct input shape (batch, 79)
4. Correct output shape
5. No NaN or Inf in predictions

**Example output:**
```
Evaluating model: ./my_model.onnx

Evaluation Results:
  MAPE:  8.15%
  Score: 0.9185
  MAE:   $23,450
  RMSE:  $67,890
  R²:    0.8234

Inference time: 245ms
✓ Model is valid and ready for submission.
```

**Metrics explained:**
| Metric | What it measures | Target |
|--------|------------------|--------|
| MAPE | Mean Absolute Percentage Error | < 15% |
| Score | 1 - MAPE (higher is better) | > 0.85 |
| MAE | Average dollar error | Lower is better |
| RMSE | Penalizes large errors more | Lower is better |
| R² | Variance explained (0-1) | > 0.70 |

### miner-cli submit

Submits your model commitment to the blockchain.

```bash
miner-cli submit \
    --model.path PATH \
    --hf.repo_id USER/REPO \
    --wallet.name NAME \
    --wallet.hotkey HOTKEY \
    [--network NETWORK] \
    [--netuid UID] \
    [--skip-scan] \
    [--scan-blocks N] \
    [--no-commit-reveal] \
    [--reveal-blocks N]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model.path` | Yes | - | Path to local ONNX model file |
| `--hf.repo_id` | Yes | - | HuggingFace repo (e.g., `alice/housing-model`) |
| `--wallet.name` | Yes | - | Bittensor wallet name |
| `--wallet.hotkey` | Yes | - | Wallet hotkey name |
| `--network` | No | `finney` | Network: `finney`, `test`, or `ws://` endpoint |
| `--netuid` | No | Auto | Subnet UID (46 for finney, 428 for test) |
| `--skip-scan` | No | False | Skip scanning for extrinsic ID |
| `--scan-blocks` | No | 25 | Blocks to scan for extrinsic |
| `--no-commit-reveal` | No | False | Disable commit-reveal (not recommended) |
| `--reveal-blocks` | No | 360 | Blocks until commitment reveal (~1 epoch) |

**Network options:**
| Network | Flag | Subnet UID |
|---------|------|------------|
| Mainnet | `--network finney` | 46 |
| Testnet | `--network test` | 428 |
| Custom | `--network ws://host:port` | Must specify `--netuid` |

**Commit-Reveal (enabled by default):**

Your commitment is encrypted using timelock encryption and only revealed after `--reveal-blocks` blocks (~72 minutes by default). This prevents frontrunning - competitors cannot see your model details until the reveal.

How it works:
1. Your commitment is encrypted with a drand timelock
2. The encrypted commitment is stored on-chain
3. After `reveal_round`, the chain automatically decrypts and reveals your commitment
4. Validators can then download and evaluate your model

To disable commit-reveal (not recommended): `--no-commit-reveal`

**What it does:**
1. Validates model file exists
2. Checks HuggingFace repo ID length (max 51 bytes)
3. Verifies hotkey is registered on subnet
4. Computes SHA-256 hash of model file
5. Submits commitment to chain: `{"h":"<hash>","r":"<repo_id>"}`
6. Scans for extrinsic ID

**Example output:**
```
License Notice:
Your HuggingFace model must be MIT licensed.
Validators verify this before evaluating your model.

Submitting model to chain...

✓ Model committed to chain with commit-reveal!

Commitment details:
  Repository:         alice/housing-v1
  Model hash:         a3f8c2e9d1b4f6a8...
  Submitted at block: 142857
  Commit-reveal:      Yes (reveal at drand round 25864000)

Scanning for extrinsic (up to 25 blocks)...
  Found extrinsic: 142858-3

Next steps:
  1. Ensure model.onnx is uploaded to your HuggingFace repo
  2. Add a LICENSE file to your repo
  3. Add extrinsic_record.json to your repo with this content:

{
  "extrinsic": "142858-3",
  "hotkey": "5ABC...XYZ"
}

  4. Wait for validator evaluation (~72 min after reveal)
```

## HuggingFace Repository Structure

Your repo must contain:

```
your-username/housing-model/
├── model.onnx              # Your ONNX model (required)
└── extrinsic_record.json   # Chain commitment link (required)
```

> **Note:** Select **MIT** license when creating your HuggingFace repository.

### extrinsic_record.json Format

```json
{
  "extrinsic": "<block_number>-<extrinsic_index>",
  "hotkey": "<your_hotkey_ss58_address>"
}
```

### Private Repositories

Your HuggingFace repository can be **private** before you commit to the chain. This prevents others from monitoring your repo for new model uploads. The workflow:

1. Create a **private** HuggingFace repo
2. Upload your model and prepare all files
3. Run `miner-cli submit` to commit to chain
4. Make the repo **public** so validators can download it

The CLI only hashes your local model file - it doesn't interact with HuggingFace. Validators will need access to download your model after you commit.

### What Validators Check

Validators perform these checks before scoring your model:

**Pre-download (via HuggingFace API):**
1. MIT license in model card metadata (`license: mit` in README.md)
2. model.onnx size ≤ 200MB
3. extrinsic_record.json exists and is valid
4. Extrinsic exists on chain and was signed by your hotkey

**Post-download:**
5. SHA-256 hash of downloaded file matches your commitment

**Important:** The file you provide to `miner-cli submit` is hashed locally. This exact file must be uploaded to HuggingFace - any difference will cause validation to fail.

## Troubleshooting

### Model file not found

```
ERROR: Model file not found: nonexistent.onnx
```
**Fix:** Check your file path. Use absolute path if needed.

### Invalid ONNX format

```
ERROR: Invalid ONNX format: Unable to parse proto from file...
```
**Fix:** Re-export your model. Test with `onnx.checker.check_model("model.onnx")`.

### Wrong number of features

```
ERROR: Model expects 72 features, but validator expects 79.
```
**Fix:** Retrain your model with the correct 79 input features from `feature_config.yaml`.

### Model too large

```
ERROR: Model size 250.00MB exceeds limit of 200MB
```
**Fix:** Reduce model size via quantization or pruning.

### NaN/Inf predictions

```
ERROR: Model produced 5 NaN predictions.
```
**Fix:** Check for numerical instability. Ensure training data has no extreme values.

### Hotkey not registered

```
ERROR: Hotkey 5ABC... is not registered on subnet 46.
```
**Fix:** Register your hotkey:
```bash
btcli subnets register --wallet.name miner --wallet.hotkey default --netuid 46
```

### Commitment failed

```
ERROR: Failed to submit commitment: ...
```
**Fix:** Check network connection and wallet funds.

### Extrinsic not found

```
Warning: Could not find extrinsic in scanned blocks
```
**Fix:** Increase `--scan-blocks` or find extrinsic manually on a block explorer.

### HuggingFace repo ID too long

```
ERROR: HF repo ID too long: 64 bytes exceeds 51 byte limit
```
**Fix:** Use a shorter repository name.

## Support

- GitHub Issues: [RESI-models Issues](https://github.com/resi-labs-ai/RESI-models/issues)
- Discord: [Join the Real Estate subnet channel](https://discord.com/channels/799672011265015819/1397618038894759956)
