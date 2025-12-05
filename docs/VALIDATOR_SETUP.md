# Validator Setup Guide

This guide explains how to set up and run a validator for the Real Estate Price Prediction Subnet.

## Prerequisites

- **Docker** installed and running
- **Python 3.11+**
- **pm2** installed (`npm install -g pm2`)
- **Bittensor wallet** with sufficient TAO for registration
- Registered hotkey on the subnet

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Validator Machine                     │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────────────┐ │
│  │   pm2 managed    │      │    Docker container      │ │
│  │                  │ HTTP │                          │ │
│  │  validator.py    │─────►│   Pylon                  │ │
│  │                  │:8000 │                          │ │
│  │  (auto-updates)  │      │  (chain interactions)    │ │
│  └──────────────────┘      └───────────┬──────────────┘ │
│                                        │                 │
└────────────────────────────────────────┼─────────────────┘
                                         │ WebSocket
                                         ▼
                                   ┌───────────┐
                                   │ Bittensor │
                                   │   Chain   │
                                   └───────────┘
```

The validator consists of two components:
1. **Pylon** (Docker) - Handles all chain interactions reliably
2. **Validator Process** (pm2) - Evaluates models and sets weights

## Step 1: Set Up Pylon

Pylon is a high-performance proxy for Bittensor chain interactions.

### Clone and Build

```bash
git clone https://github.com/resi-subnet/bittensor-pylon.git
cd bittensor-pylon
docker build -t resi-pylon .
```

### Generate Authentication Token

```bash
openssl rand -hex 32
```

Save this token - you'll need it for both Pylon and the validator.

### Configure and Run

Follow the instructions in the [Pylon README](https://github.com/resi-subnet/bittensor-pylon#running-the-rest-api-on-docker) to configure your `.env` file using the template at `pylon/service/envs/test_env.template`.

Key settings to configure:
- `PYLON_BITTENSOR_NETWORK` - Set to `finney` for mainnet
- `PYLON_BITTENSOR_WALLET_NAME` - Your validator wallet name
- `PYLON_BITTENSOR_WALLET_HOTKEY_NAME` - Your hotkey name
- `PYLON_ID_<NAME>_BITTENSOR_TOKEN` - The token you generated above

Then start Pylon using docker-compose as described in the Pylon README.

### Verify Pylon is Running

```bash
curl http://localhost:8000/schema
curl http://localhost:8000/api/v1/commitments
```

## Step 2: Start the Validator

### Clone the Repository

```bash
git clone https://github.com/your-org/resi-subnet.git
cd resi-subnet
```

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Configure Environment

Create a `.env.validator` file:

```bash
# Pylon connection
PYLON_URL=http://localhost:8000
PYLON_TOKEN=your_secret_token_here

# Wallet (for any direct chain ops)
WALLET_NAME=validator
WALLET_HOTKEY=default

# Subnet
NETUID=XX

# Optional
HF_TOKEN=your_huggingface_token
```

### Start with Auto-Updates

```bash
python scripts/start_validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --netuid XX
```

This will:
1. Start the validator process managed by pm2
2. Automatically check for updates every 5 minutes
3. Pull and restart when new versions are available

### Manual Start (without auto-updates)

```bash
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --netuid XX
```

## Step 3: Verify Everything is Working

### Check pm2 Status

```bash
pm2 status
pm2 logs resi-validator
```

### Check Pylon Health

```bash
curl http://localhost:8000/api/v1/neurons/latest
```

### Check Validator Logs

```bash
pm2 logs resi-validator --lines 100
```

## Managing the Validator

### Stop Validator

```bash
pm2 stop resi-validator
```

### Restart Validator

```bash
pm2 restart resi-validator
```

### Stop Pylon

```bash
docker stop resi-pylon
```

### Update Pylon

```bash
cd bittensor-pylon
git pull
docker build -t resi-pylon .
cd pylon/service/envs
docker-compose down
docker-compose up -d
```

### View All Logs

```bash
# Validator logs
pm2 logs resi-validator

# Pylon logs
docker logs -f resi-pylon
```

## Troubleshooting

### Pylon won't start

1. Check Docker is running: `docker ps`
2. Check logs: `docker logs resi-pylon`
3. Verify wallet path exists: `ls ~/.bittensor/wallets/<wallet_name>`
4. Ensure port 8000 is not in use: `lsof -i :8000`

### Validator can't connect to Pylon

1. Verify Pylon is running: `curl http://localhost:8000/schema`
2. Check PYLON_URL in validator config
3. Check PYLON_TOKEN matches between validator and Pylon

### Chain connection errors

1. Check network connectivity to Bittensor
2. Verify PYLON_BITTENSOR_NETWORK is correct (`finney`, `test`, or custom endpoint)
3. Check Pylon logs for WebSocket errors

### Wallet errors

1. Verify wallet exists: `btcli wallet list`
2. Check wallet is registered: `btcli subnet metagraph --netuid XX`
3. Ensure wallet files are readable by Docker (check volume mount)

## Network Configuration

### Mainnet (Finney)

```bash
PYLON_BITTENSOR_NETWORK=finney
```

### Testnet

```bash
PYLON_BITTENSOR_NETWORK=test
```

### Custom Endpoint

```bash
PYLON_BITTENSOR_NETWORK=ws://your-node:9944
```

## Security Recommendations

1. **Use strong tokens**: Generate with `openssl rand -hex 32`
2. **Don't expose Pylon publicly**: Keep port 8000 bound to localhost or use firewall
3. **Mount wallets read-only**: Use `:ro` flag in Docker volume mount
4. **Keep software updated**: Regularly pull latest code and rebuild

## Hardware Requirements

### Minimum

- 4 CPU cores
- 8GB RAM
- 50GB SSD
- Stable internet connection

### Recommended

- 8 CPU cores
- 16GB RAM
- 100GB SSD
- Low-latency internet connection

## Support

- GitHub Issues: https://github.com/your-org/resi-subnet/issues
- Discord: [link]
