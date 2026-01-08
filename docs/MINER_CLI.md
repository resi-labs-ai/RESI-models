# RESI Miner CLI

Command-line tool for RESI subnet miners to evaluate ONNX models locally and submit them to the Bittensor chain.

## Installation

```bash
git clone https://github.com/your-org/RESI-models.git
cd RESI-models
uv sync
```

After installation, the `miner-cli` command will be available.

## Quick Start

```bash
miner-cli evaluate ./model.onnx

miner-cli submit \
  --hf_repo_id your-username/your-model \
  --wallet_name miner \
  --wallet_hotkey default
```

## Commands

### evaluate

```bash
miner-cli evaluate ./model.onnx
miner-cli evaluate -m ./model.onnx
```

### submit

```bash
miner-cli submit \
  --hf_repo_id <username/repo> \
  --wallet_name <wallet> \
  --wallet_hotkey <hotkey>
```

| Argument | Default | Description |
|----------|---------|-------------|
| --hf_repo_id | required | HuggingFace repo |
| --wallet_name | required | Wallet name |
| --wallet_hotkey | required | Hotkey name |
| --hf_model_filename | model.onnx | Model filename |
| --network | finney | Network |
| --netuid | 46 | Subnet UID |

## Troubleshooting

- **File not found** - Use `--hf_model_filename`
- **Not registered** - Run `btcli subnets register --netuid 46`
