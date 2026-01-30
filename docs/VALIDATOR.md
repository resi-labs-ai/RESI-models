# Validator Setup Guide

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Validator](#running-the-validator)
- [Managing the Validator](#managing-the-validator)
- [Log Management](#log-management)
- [Configuration Reference](#configuration-reference)
- [Network Configuration](#network-configuration)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Security Recommendations](#security-recommendations)
- [Support](#support)

## Overview

The main responsibilities of a RESI validator are:

- Download Models: Fetch miner ONNX models from HuggingFace
- Evaluate: Run inference against daily validation data
- Set Weights: Submit scores to the Bittensor blockchain

## Hardware Requirements

Suggested starting point - adjust based on your load and number of miners on the subnet.

- 4+ CPU cores
- 16+ GB RAM
- 50+ GB SSD (model cache for up to 256 miners)
- Stable internet connection

## Prerequisites

- **Docker** installed and running
- **Python 3.11+**
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **PM2** process manager (`npm install -g pm2`)
- **Bittensor wallet** with sufficient TAO for registration
- Registered hotkey on subnet 46

## Installation

### Clone and Install

```bash
git clone https://github.com/resi-labs-ai/RESI-models.git
cd RESI-models
uv sync
```

### Generate Pylon Token

```bash
openssl rand -base64 32
```

Save this token - you'll need it for the `.env` file.

### Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# =============================================================================
# REQUIRED - Wallet Configuration
# =============================================================================
WALLET_NAME=validator
WALLET_HOTKEY=default
BITTENSOR_WALLET_PATH=~/.bittensor/wallets

# =============================================================================
# REQUIRED - Pylon Configuration
# =============================================================================
PYLON_TOKEN=<your_generated_token_here>
PYLON_IDENTITY=validator
PYLON_URL=http://localhost:8000

# =============================================================================
# Network Configuration
# =============================================================================
SUBTENSOR_NETWORK=finney
NETUID=46
```

## Running the Validator

### Start Pylon

Pylon handles all Bittensor chain interactions. Start it with Docker Compose:

```bash
docker compose up -d
```

> **Note for Mac (Apple Silicon) users:** Pylon has no ARM64 build.
> Run with: `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d`

Verify Pylon is running:

```bash
# Check container status
docker ps

# Check logs
docker logs resi_pylon

# Test API is responding (NETUID: mainnet=46, testnet=428 , local=<localnet_netuid>)
curl http://localhost:8000/api/v1/identity/validator/subnet/<NETUID>/block/latest/neurons
```

### Start Validator

**Option A: With Auto-Updates (Recommended)**

The auto-update script checks for new versions every 5 minutes and automatically restarts:

```bash
# Load environment and start
set -a && source .env && set +a
pm2 start "uv run python scripts/start_validator.py" --name resi_autoupdater
```

Or pass arguments directly:

```bash
pm2 start "uv run python scripts/start_validator.py \
    --wallet.name validator \
    --wallet.hotkey default \
    --netuid 46 \
    --pylon.token YOUR_PYLON_TOKEN \
    --pylon.identity validator" --name resi_autoupdater
```
This creates two PM2 processes:

- resi_autoupdater - monitors for updates and manages the validator
- resi_validator - the actual validator process

**Option B: Manual Start (No Auto-Updates)**

```bash
set -a && source .env && set +a 
pm2 start "uv run python -m real_estate.validator.validator \
    --wallet.name validator \
    --wallet.hotkey default \
    --netuid 46 \
    --pylon.token YOUR_PYLON_TOKEN \
    --pylon.identity validator" --name resi_validator
```

### Verify Startup

#### Check PM2 Status

```bash
pm2 status
pm2 logs resi_validator --lines 50
pm2 logs resi_autoupdater --lines 50
```

#### Check Pylon Logs

```bash
docker logs -f resi_pylon
```

#### Expected Startup Logs

```
INFO | Config: {'netuid': 46, 'wallet_name': 'validator', ...}
INFO | Validator initialized
INFO | Starting metagraph sync...
INFO | Pre-download phase: ...
```

## Managing the Validator

### Stop Validator

**If using auto-updates:**
```bash
pm2 stop resi_autoupdater resi_validator
```

**If running manually (no auto-updates):**
```bash
pm2 stop resi_validator
```

### Restart Validator

**If using auto-updates:**
```bash
pm2 restart resi_autoupdater
```

**If running manually:**
```bash
pm2 restart resi_validator
```

### Stop Pylon

```bash
docker compose down
```

### View Logs

```bash
# Validator logs
pm2 logs resi_validator

# Auto-updater logs (if using auto-updates)
pm2 logs resi_autoupdater

# Pylon logs
docker logs -f resi_pylon
```

### Full Restart

**If using auto-updates:**
```bash
pm2 delete resi_autoupdater resi_validator
docker compose down
docker compose up -d
set -a && source .env && set +a
pm2 start "uv run python scripts/start_validator.py" --name resi_autoupdater
```

**If running manually:**
```bash
pm2 delete resi_validator
docker compose down
docker compose up -d
set -a && source .env && set +a
pm2 start "uv run python -m real_estate.validator.validator \
    --wallet.name validator \
    --wallet.hotkey default \
    --netuid 46 \
    --pylon.token YOUR_PYLON_TOKEN \
    --pylon.identity validator" --name resi_validator
```

## Log Management

By default, PM2 stores validator logs in `~/.pm2/logs/`. These logs grow indefinitely and can consume significant disk space. We recommend using `pm2-logrotate` to automatically rotate logs daily.

### Setup (Optional)

Install the pm2-logrotate module:

```bash
pm2 install pm2-logrotate
```

Configure for daily rotation:

```bash
pm2 set pm2-logrotate:max_size 100M
pm2 set pm2-logrotate:retain 30
pm2 set pm2-logrotate:compress false
pm2 set pm2-logrotate:dateFormat YYYY-MM-DD
pm2 set pm2-logrotate:rotateInterval '0 0 * * *'
```

This will:
- Rotate logs daily at midnight (or when they exceed 100MB)
- Keep 30 days of logs
- Name files with sortable dates (e.g., `resi_validator-out__2026-01-28.log`)

### View Logs

```bash
# Current logs
pm2 logs resi_validator

# List rotated log files
ls ~/.pm2/logs/

# View a specific day's logs
cat ~/.pm2/logs/resi_validator-out__2026-01-28.log

# Search for errors across all logs
grep -E "\| ERROR \||\| CRITICAL \|" ~/.pm2/logs/resi_validator-out*.log
```

### Pylon Logs

Pylon uses Docker's json-file logging driver with automatic rotation (50MB × 10 files = 500MB max). Logs persist across container restarts but are removed when the container is deleted.

```bash
# View live pylon logs
docker logs -f resi_pylon

# View recent logs (last 100 lines)
docker logs --tail 100 resi_pylon

# View logs from last 24 hours
docker logs --since 24h resi_pylon

# Search pylon logs for errors
docker logs resi_pylon 2>&1 | grep -i error

# Export logs for debugging with subnet teams
docker logs --since 24h resi_pylon > pylon_debug.log 2>&1
```

## Configuration Reference

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `WALLET_NAME` | Bittensor wallet name |
| `WALLET_HOTKEY` | Hotkey name |
| `BITTENSOR_WALLET_PATH` | Path to wallets directory |
| `PYLON_TOKEN` | Authentication token for Pylon |
| `PYLON_IDENTITY` | Identity name (must match docker-compose) |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBTENSOR_NETWORK` | `finney` | Network name or `ws://` endpoint |
| `NETUID` | `46` | Subnet UID |
| `PYLON_URL` | `http://localhost:8000` | Pylon service URL |
| `EPOCH_LENGTH` | `360` | Blocks between weight setting |
| `SCORE_THRESHOLD` | `0.005` | Score threshold for winner selection |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DISABLE_SET_WEIGHTS` | `false` | Disable weight setting (for testing) |

## Network Configuration

### Mainnet (Finney)

```bash
SUBTENSOR_NETWORK=finney
NETUID=46
```

### Testnet

```bash
SUBTENSOR_NETWORK=test
NETUID=428
```

### Custom Endpoint

```bash
SUBTENSOR_NETWORK=ws://your-node:port
```

## Troubleshooting

### Pylon Won't Start

```bash
# Check Docker is running
docker ps

# Check logs
docker logs resi_pylon

# Verify wallet path exists
ls ~/.bittensor/wallets/$WALLET_NAME

# Ensure port 8000 is free
lsof -i :8000
```

### Validator Can't Connect to Pylon

```bash
# Verify Pylon is responding
curl http://localhost:8000/api/v1/identity/validator/subnet/46/block/latest/neurons

# Check token matches
echo $PYLON_TOKEN

# Check identity matches
echo $PYLON_IDENTITY  # Should be "validator"
```

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure dependencies are installed
uv sync

# Run with uv
uv run python -m real_estate.validator.validator ...
```

### Chain Connection Errors

1. Check network connectivity
2. Verify `SUBTENSOR_NETWORK` is correct
3. Check Pylon logs for WebSocket errors: `docker logs resi_pylon`

### Wallet Errors

```bash
# List wallets
btcli wallet list

# Check registration
btcli subnet metagraph --netuid 46 --subtensor.network finney

# Verify wallet files are readable
ls -la ~/.bittensor/wallets/$WALLET_NAME/hotkeys/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Validator Machine                       │
│                                                              │
│  ┌────────────────────┐       ┌────────────────────────────┐│
│  │    PM2 managed     │       │     Docker container       ││
│  │                    │ HTTP  │                            ││
│  │  validator.py      │──────►│   Pylon 1.0.0              ││
│  │                    │ :8000 │                            ││
│  │  (auto-updates)    │       │   (chain interactions)     ││
│  └────────────────────┘       └───────────┬────────────────┘│
│                                           │                  │
└───────────────────────────────────────────┼──────────────────┘
                                            │ WebSocket
                                            ▼
                                      ┌───────────┐
                                      │ Bittensor │
                                      │   Chain   │
                                      └───────────┘
```

The validator consists of two components:

**Pylon** (Docker) - Handles all Bittensor chain interactions including metagraph sync, weight setting, and commitment queries. Runs as a separate service to isolate chain communication.

**Validator Process** (PM2) - Downloads miner models from HuggingFace, runs daily evaluation against validation data, scores predictions, and determines weight distribution.

## Security Recommendations

1. **Use strong tokens**: Generate with `openssl rand -base64 32`
2. **Pylon bound to localhost**: docker-compose binds to `127.0.0.1:8000` by default
3. **Wallets mounted read-only**: Docker volume uses `:ro` flag
4. **Keep software updated**: Auto-update script handles this automatically

## Support

- GitHub Issues: https://github.com/resi-labs-ai/RESI-models/issues