import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from real_estate.validator import Validator
from real_estate.chain.models import Neuron, Metagraph
from datetime import datetime

def create_mock_neuron(uid: int, hotkey: str, validator_permit: bool = False, emission: float = 1.0) -> Neuron:
    return Neuron(
        uid=uid,
        hotkey=hotkey,
        coldkey=f"coldkey_{uid}",
        stake=100.0,
        trust=0.5,
        consensus=0.5,
        incentive=0.1,
        dividends=0.1,
        emission=emission,
        is_active=True,
        validator_permit=validator_permit,
    )

@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    config.netuid = 46
    config.burn_uid = 2
    config.subtensor_network = "mock"
    config.wallet_name = "mock"
    config.wallet_hotkey = "mock"
    config.wallet_path = "/tmp"
    config.pylon_url = "mock"
    config.pylon_token = "mock"
    config.pylon_identity = None
    config.validation_data_url = "mock"
    config.validation_data_max_retries = 3
    config.validation_data_retry_delay = 1
    config.validation_data_schedule_hour = 2
    config.validation_data_schedule_minute = 0
    config.validation_data_download_raw = False
    config.test_data_path = None
    config.randomness_enabled = False
    config.randomness_cycle_window_hours = 4.0
    config.randomness_blocks_until_reveal = 360
    config.randomness_reveal_buffer_seconds = 300
    config.randomness_block_time_seconds = 12
    config.randomness_min_quorum = 2
    config.score_threshold = 0.005
    config.docker_timeout = 3600
    config.docker_memory = "4g"
    config.docker_cpu = 1.0
    config.docker_max_concurrent = 1
    config.wandb_project = "mock"
    config.wandb_entity = "mock"
    config.wandb_api_key = "mock"
    config.wandb_off = True
    config.wandb_offline = True
    config.wandb_log_predictions = False
    config.wandb_predictions_top_n = 10
    config.disable_set_weights = False
    config.epoch_length = 100
    return config

@pytest.fixture
def validator(mock_config):
    with (
        patch("real_estate.validator.validator.check_config"),
        patch("real_estate.validator.validator.bt.subtensor"),
        patch("real_estate.validator.validator.bt.wallet") as mock_wallet,
        patch("real_estate.validator.validator.ValidationClient"),
        patch("real_estate.validator.validator.ValidationOrchestrator"),
    ):
        mock_wallet.return_value.hotkey.ss58_address = "our_hotkey"
        v = Validator(mock_config)
        v.hotkeys = ["hk0", "hk1", "hk2", "hk3"]
        return v

@pytest.mark.asyncio
async def test_apply_burn_below_limit(validator):
    # Setup: 10 Alpha/day * $200 TAO * 0.01 TAO/Alpha = $20 USD/day (Below $3000)
    # Emission = 10 / 7200 per block
    emission_per_block = 10.0 / 7200.0
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=emission_per_block),
        create_mock_neuron(1, "hk1", emission=emission_per_block),
        create_mock_neuron(2, "hk2", emission=0.0), # Burn UID
    ]
    
    weights = {"hk0": 0.5, "hk1": 0.5}
    
    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        # Set MANUAL_BURN to 0.1 for testing
        validator.MANUAL_BURN = 0.1
        adjusted = await validator._apply_burn(weights)
        
        # cap_burn should be 0.0
        # burn_amount = 1.0 - (1.0 - 0.0) * (1.0 - 0.1) = 0.1
        assert adjusted["hk2"] == pytest.approx(0.1)
        assert adjusted["hk0"] == pytest.approx(0.5 * 0.9)
        assert adjusted["hk1"] == pytest.approx(0.5 * 0.9)

@pytest.mark.asyncio
async def test_apply_burn_above_limit(validator):
    # Setup: 1,000,000 Alpha/day * $200 TAO * 0.01 TAO/Alpha = $2,000,000 USD/day (Above $3000)
    emission_per_block = 1_000_000.0 / 7200.0
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=emission_per_block),
        create_mock_neuron(1, "hk1", emission=emission_per_block),
        create_mock_neuron(2, "hk2", emission=0.0), # Burn UID
    ]
    
    weights = {"hk0": 0.5, "hk1": 0.5}
    
    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        validator.MANUAL_BURN = 0.0 # Only cap burn
        # Daily value = (2,000,000 / 7200) * 7200 * 0.01 * 200 = 2,000,000 * 2 = 4,000,000 USD?
        # Wait. emission is per block. sum(emissions) = 2 * (1M/7200)
        # total_miner_alpha_emission_daily = (2M / 7200) * 7200 = 2M Alpha
        # value = 2M * 0.01 * 200 = 4,000,000 USD.
        # cap_burn = 1.0 - (3000 / 4,000,000) = 1.0 - 0.00075 = 0.99925
        
        adjusted = await validator._apply_burn(weights)
        
        expected_burn = 1.0 - (3000.0 / 4000000.0)
        assert adjusted["hk2"] == pytest.approx(expected_burn)
        assert adjusted["hk0"] == pytest.approx(0.5 * (1.0 - expected_burn))

@pytest.mark.asyncio
async def test_apply_burn_manual_100_percent(validator):
    # Even if below limit, if MANUAL_BURN is 1.0, it should be 100% burn
    emission_per_block = 10.0 / 7200.0
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=emission_per_block),
    ]
    weights = {"hk0": 1.0}
    
    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        validator.MANUAL_BURN = 1.0
        adjusted = await validator._apply_burn(weights)
        
        assert adjusted["hk2"] == pytest.approx(1.0)
        assert adjusted.get("hk0", 0.0) == 0.0
