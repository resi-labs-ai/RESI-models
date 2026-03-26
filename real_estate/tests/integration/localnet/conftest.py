"""Fixtures for localnet integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from real_estate.data import ValidationDataset
from real_estate.tests.fixtures.evaluation.conftest import create_test_model

# ---------------------------------------------------------------------------
# Session-scoped: check infra, skip if unavailable
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def localnet_config():
    """Read localnet connection params from env, skip if not configured."""
    pylon_token = os.environ.get("PYLON_TOKEN", "")
    if not pylon_token:
        pytest.skip("PYLON_TOKEN not set — skipping chain tests")
    return {
        "pylon_url": os.environ.get("PYLON_URL", "http://localhost:8000"),
        "pylon_token": pylon_token,
        "pylon_identity": os.environ.get("PYLON_IDENTITY", "validator"),
        "subtensor_network": os.environ.get(
            "SUBTENSOR_NETWORK", "ws://68.183.141.180:80"
        ),
        "netuid": int(os.environ.get("NETUID", "46")),
    }


@pytest.fixture(scope="session")
def subtensor_endpoint():
    """Subtensor websocket endpoint, skip if not set."""
    endpoint = os.environ.get("SUBTENSOR_NETWORK", "")
    if not endpoint:
        pytest.skip("SUBTENSOR_NETWORK not set")
    return endpoint


@pytest.fixture(scope="session")
async def chain_client(localnet_config):
    """Real ChainClient connected to Pylon. Skips if unreachable."""
    from real_estate.chain import ChainClient, PylonConfig

    config = PylonConfig(
        url=localnet_config["pylon_url"],
        token=localnet_config["pylon_token"],
        identity=localnet_config["pylon_identity"],
    )
    async with ChainClient(config) as client:
        try:
            await client.get_metagraph()  # connectivity check
        except Exception:
            pytest.skip("Pylon not reachable")
        yield client


# ---------------------------------------------------------------------------
# Module-scoped: reused across tests in same file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """5-feature config YAML (matches existing integration test pattern)."""
    config_dir = tmp_path_factory.mktemp("config")
    config_path = config_dir / "feature_config.yaml"

    config = {
        "version": "1.0.0",
        "numeric_fields": ["sqft", "beds", "baths", "lot_size", "year_built"],
        "boolean_fields": [],
        "feature_transforms": [],
        "feature_order": ["sqft", "beds", "baths", "lot_size", "year_built"],
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture(scope="module")
def test_models_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[str]]:
    """Create {hotkey}/model.onnx layout for 3 test miners (5 features)."""
    hotkeys = ["5TestHotkey1aaa", "5TestHotkey2bbb", "5TestHotkey3ccc"]
    base = tmp_path_factory.mktemp("models")
    for i, hk in enumerate(hotkeys):
        d = base / hk
        d.mkdir()
        create_test_model(n_features=5, output_path=d / "model.onnx", seed=i * 10)
    return base, hotkeys


@pytest.fixture(scope="module")
def sample_properties() -> list[dict]:
    """Create 20 sample properties with 5 numeric fields."""
    np.random.seed(42)
    properties = []
    for _ in range(20):
        properties.append(
            {
                "price": float(np.random.uniform(200000, 800000)),
                "sqft": float(np.random.uniform(1000, 3000)),
                "beds": float(np.random.randint(2, 6)),
                "baths": float(np.random.randint(1, 4)),
                "lot_size": float(np.random.uniform(5000, 20000)),
                "year_built": float(np.random.randint(1960, 2020)),
            }
        )
    return properties


@pytest.fixture(scope="module")
def validation_dataset(sample_properties: list[dict]) -> ValidationDataset:
    """20 sample properties with 5 fields."""
    return ValidationDataset(properties=sample_properties)
