# Real Estate Price Prediction Subnet

[![CI](https://github.com/konrad0960/RESI-models/actions/workflows/ci.yml/badge.svg)](https://github.com/konrad0960/RESI-models/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Bittensor subnet for real estate price prediction using machine learning models.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This subnet incentivizes the development of accurate real estate price prediction models. Miners submit ONNX models to HuggingFace, and validators evaluate them against ground-truth sales data.

**Key Features:**
- Winner-takes-all incentive mechanism (99% to best model)
- Pioneer detection to prevent model copying
- Docker-isolated model evaluation
- Chain commitments via Bittensor

---

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) - a fast Python package manager written in Rust.

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

### Setup Project

```bash
# Clone the repository
git clone https://github.com/konrad0960/RESI-models.git
cd RESI-models

# Create virtual environment and install all dependencies
uv sync --dev

# Verify installation
uv run pytest tests/ -v
```

This will:
1. Create a `.venv` virtual environment
2. Install all dependencies from `uv.lock`
3. Install the project in editable mode

---

## Development Workflow

### Running Tools

All tools are run through `uv run` to use the project's virtual environment:

```bash
# Linting
uv run ruff check .          # Check for issues
uv run ruff check --fix .    # Auto-fix issues

# Formatting
uv run ruff format --check . # Check formatting
uv run ruff format .         # Auto-format code

# Type checking
uv run mypy real_estate neurons

# Testing
uv run pytest tests/ -v                    # Run all tests
uv run pytest tests/ -v -m "not slow"      # Skip slow tests
uv run pytest tests/ --cov=real_estate     # With coverage
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Update all dependencies
uv lock --upgrade
uv sync
```

### CI/CD

This project uses **GitHub Actions** for continuous integration. On every push and PR to `main`:

| Job | Description |
|-----|-------------|
| **Lint** | Runs `ruff check` and `ruff format --check` |
| **Type Check** | Runs `mypy` static analysis |
| **Test** | Runs `pytest` on Python 3.10, 3.11, 3.12 |

The CI configuration is in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Pre-commit Checklist

Before committing, ensure:

```bash
# 1. Format code
uv run ruff format .

# 2. Fix lint issues
uv run ruff check --fix .

# 3. Run tests
uv run pytest tests/ -v

# 4. (Optional) Type check
uv run mypy real_estate neurons
```

---

## Project Structure

Repo under heavy changes, this is just for reference:
```
RESI-models/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── real_estate/             # Main package (to be created)
│   ├── chain/               # Chain interaction (Pylon)
│   ├── data/                # Dataset management
│   ├── models/              # Model downloading/verification
│   ├── evaluation/          # Docker-based evaluation
│   ├── detection/           # Duplicate detection
│   └── incentives/          # Scoring and weights
├── neurons/                 # Validator/Miner entry points
├── tests/                   # Test suite
├── pyproject.toml           # Project config (deps, tools)
├── uv.lock                  # Locked dependencies
└── README.md
```

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
