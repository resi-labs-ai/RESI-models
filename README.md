# Real Estate Price Prediction Subnet

A Bittensor subnet for real estate price prediction using machine learning models.

**Subnet 46** on Bittensor Mainnet

---

## Overview

This subnet incentivizes the development of accurate real estate price prediction models. Miners submit ONNX models to HuggingFace, and validators evaluate them against ground-truth sales data.

---

## Quick Start

### For Validators

See the [Validator Setup Guide](docs/VALIDATOR_SETUP.md) for setup instructions.

### For Miners

```bash
# Clone and install
git clone https://github.com/resi-labs-ai/RESI-models.git
cd RESI-models
uv sync

# Train your model and export to ONNX
# Upload to HuggingFace
# Register commitment on-chain

uv run python -m real_estate.miner.miner_cli register \
    --wallet.name miner \
    --wallet.hotkey default \
    --hf_repo your-username/your-model
```

Miner guide coming soon.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Miners      │     │   Validators    │     │    Bittensor    │
│                 │     │                 │     │      Chain      │
│  Train models   │     │  Fetch models   │     │                 │
│  Upload to HF   │────►│  Evaluate       │────►│  Set weights    │
│  Commit hash    │     │  Score & rank   │     │  Distribute TAO │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) - a fast Python package manager.

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (for running validators)

### Install

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/resi-labs-ai/RESI-models.git
cd RESI-models
uv sync

# Run tests
uv run pytest real_estate/tests/ -v
```

### Development Workflow

```bash
# Linting
uv run ruff check .
uv run ruff check --fix .

# Formatting
uv run ruff format .

# Testing
uv run pytest real_estate/tests/ -v
uv run pytest real_estate/tests/ --cov=real_estate
```

### CI/CD

GitHub Actions runs on every push and PR:

| Job | Description |
|-----|-------------|
| **Lint** | `ruff check` and `ruff format --check` |
| **Test** | `pytest` on Python 3.11, 3.12 |

---

## Project Structure

```
RESI-models/
├── real_estate/
│   ├── chain/           # Pylon client for chain interactions
│   ├── data/            # Validation dataset management
│   ├── duplicate_detector/  # Pioneer/copier detection
│   ├── evaluation/      # Docker-based model evaluation
│   ├── incentives/      # Scoring and weight distribution
│   ├── models/          # Model downloading and verification
│   ├── orchestration/   # Validation pipeline orchestration
│   ├── validator/       # Validator entry point and config
│   └── tests/           # Test suite
├── scripts/
│   └── start_validator.py  # Auto-updating validator runner
├── docs/
│   └── VALIDATOR_SETUP.md  # Validator setup guide
├── docker-compose.yml   # Pylon service configuration
├── .env.example         # Environment template
└── pyproject.toml       # Project configuration
```

---

## Documentation

- [Validator Setup Guide](docs/VALIDATOR_SETUP.md)
- Miner Guide (coming soon)

---

## Support

- GitHub Issues: https://github.com/resi-labs-ai/RESI-models/issues
- Discord: [RESI Discord](https://discord.gg/resi)

---

## License

Coming soon.
