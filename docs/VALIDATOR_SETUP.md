# Validator Setup Guide

This guide explains how to set up and run a validator for the Real Estate Price Prediction Subnet (RESI).

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

## Architecture Overview

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
1. **Pylon** (Docker) - Handles all chain interactions (metagraph, weights, commitments)
2. **Validator Process** (PM2) - Evaluates miner models and sets weights

## Step 1: Clone and Install

```bash
git clone https://github.com/resi-labs-ai/RESI-models.git
cd RESI-models
uv sync
```

## Step 2: Configure Environment

### Generate Pylon Token

First, generate a secure authentication token:

```bash
openssl rand -base64 32
```

Save this token - you'll need it for the `.env` file.

### Create Environment File

Copy the example and edit:

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

## Step 3: Start Pylon

Pylon handles all Bittensor chain interactions. Start it with Docker Compose:

```bash
docker compose up -d
```

> **Note for Mac (Apple Silicon) users:** Pylon has no ARM64 build.
> Run with: `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d`

### Verify Pylon is Running

```bash
# Check container status
docker ps

# Check logs
docker logs resi_pylon

# Test API is responding
curl http://localhost:8000/api/v1/identity/validator/subnet/46/block/latest/neurons
```

## Step 4: Start the Validator

### Option A: With Auto-Updates (Recommended)

The auto-update script checks for new versions every 5 minutes and automatically restarts:

```bash
# Load environment and start
set -a && source .env && set +a
python scripts/start_validator.py
```

Or pass arguments directly:

```bash
python scripts/start_validator.py \
    --wallet.name validator \
    --wallet.hotkey default \
    --netuid 46 \
    --pylon.token YOUR_PYLON_TOKEN \
    --pylon.identity validator
```

The script will:
1. Generate PM2 ecosystem config
2. Start the validator process
3. Monitor git for updates every 5 minutes
4. Auto-restart on new versions

### Option B: Manual Start (No Auto-Updates)

```bash
uv run python -m real_estate.validator.validator \
    --wallet.name validator \
    --wallet.hotkey default \
    --netuid 46 \
    --pylon.token YOUR_PYLON_TOKEN \
    --pylon.identity validator
```

## Step 5: Verify Everything Works

### Check PM2 Status

```bash
pm2 status
pm2 logs resi_validator --lines 50
```

### Check Pylon Logs

```bash
docker logs -f resi_pylon
```

### Expected Startup Logs

```
INFO | Config: {'netuid': 46, 'wallet_name': 'validator', ...}
INFO | Validator initialized
INFO | Starting metagraph sync...
INFO | Next evaluation scheduled at 2026-01-20 02:00:00+00:00
INFO | Pre-download phase: ...
```

## Managing the Validator

### Stop Validator

```bash
pm2 stop resi_validator
```

### Restart Validator

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

# Pylon logs
docker logs -f resi_pylon
```

### Full Restart

```bash
pm2 delete resi_validator
docker compose down
docker compose up -d
set -a && source .env && set +a
python scripts/start_validator.py
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
NETUID=46
```

### Custom Endpoint

```bash
SUBTENSOR_NETWORK=ws://your-node:9944
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

## Security Recommendations

1. **Use strong tokens**: Generate with `openssl rand -base64 32`
2. **Pylon bound to localhost**: docker-compose binds to `127.0.0.1:8000` by default
3. **Wallets mounted read-only**: Docker volume uses `:ro` flag
4. **Keep software updated**: Auto-update script handles this automatically

## Support

- GitHub Issues: https://github.com/resi-labs-ai/RESI-models/issues
