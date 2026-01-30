# RESI - Real Estate Price Prediction Subnet

**Subnet 46** on Bittensor Mainnet | [Dashboard](https://dashboard.resilabs.ai) | [Validator Guide](docs/VALIDATOR.md) | [Miner Guide](docs/MINER.md)

---

## Overview

RESI is a Bittensor subnet that incentivizes the development of accurate real estate price prediction models. Miners compete to build the best ONNX models for predicting US residential property prices, while validators evaluate predictions against ground-truth sales data.

### Key Features

- **Daily Evaluation Cycle**: Models are evaluated daily at 18:00 UTC against real sales data
- **Never-Before-Seen Data**: Models must be committed ~28 hours before evaluation; evaluation uses last 24 hours of sales data - ensuring models are tested on data they couldn't have seen
- **Winner-Takes-All**: Best performing model receives 99% of emissions

---

## Incentive Mechanism

### Evaluation on Unseen Data

To ensure models generalize rather than memorize, RESI enforces a temporal separation:

- **Commit Cutoff**: Models must be committed on-chain **~28 hours before evaluation**
- **Fresh Data**: Evaluation uses sales data from the **last 24 hours**

This guarantees that every model is tested against data that didn't exist when the model was submitted.

### How Scoring Works

Models are scored using **MAPE (Mean Absolute Percentage Error)**:

```
Score = 1 - MAPE
```

Example: A model with 8.5% average prediction error has a score of 0.915.

### Winner Selection

RESI uses a **threshold + commit-time mechanism** to reward innovation:

1. **Find Best Score**: Identify the highest-scoring model
2. **Define Winner Set**: All models within a configurable threshold of the best score qualify
3. **Select Winner**: Within the winner set, the **earliest on-chain commit wins**

This means:
- If you match the current best model, the original pioneer keeps winning
- To become the new winner, you must **improve by more than the threshold**
- Incremental copycats cannot displace innovators

### Reward Distribution

| Category | Share | Description |
|----------|-------|-------------|
| **Winner** | 99% | Model that pioneered the best performance |
| **Non-winners** | 1% | Shared proportionally by score among valid models |
| **Copiers** | 0% | Detected duplicates receive nothing |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Miners      │     │   Validators    │     │    Bittensor    │
│                 │     │                 │     │      Chain      │
│  Train models   │     │  Fetch models   │     │                 │
│  Upload to HF   │────►│  Run inference  │────►│  Set weights    │
│  Commit hash    │     │  Score & rank   │     │  Distribute TAO │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Components

- **Miners**: Train ML models, export to ONNX, upload to HuggingFace, register on-chain
- **Validators**: Download models, run sandboxed inference, calculate scores, set weights
- **Pylon**: Chain interaction layer handling metagraph sync and weight submission

---

## Quick Start

### For Validators

See the [Validator Setup Guide](docs/VALIDATOR.md) for complete setup instructions.

### For Miners

See the [Miner Guide](docs/MINER.md) for complete setup instructions.

---

## Model Requirements

| Requirement | Specification |
|-------------|---------------|
| **Format** | ONNX (`.onnx` file) |
| **Max Size** | 200 MB |
| **License** | MIT (verified via HuggingFace metadata) |
| **Commit Age** | Must be committed ~28 hours before evaluation |
| **Input** | Property features (see documentation) |
| **Output** | Predicted price in USD |

---

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (for validators)

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

---

## Support

- **GitHub Issues**: https://github.com/resi-labs-ai/RESI-models/issues
