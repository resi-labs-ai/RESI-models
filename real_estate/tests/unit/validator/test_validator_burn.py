import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from real_estate.validator import Validator
from real_estate.chain.models import Neuron, Metagraph
from datetime import datetime, UTC

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
        # Metagraph emission is per-tempo; burn annualizes by 7200 / tempo.
        v.subtensor.tempo.return_value = 360  # -> steps_per_day = 20
        return v

@pytest.mark.parametrize(
    "day, expected_cap",
    [
        (0, 3000.0),
        (1, 2900.0),
        (15, 1500.0),
        (29, 100.0),
        (30, 0.0),
        (31, 0.0),
        (100, 0.0),
    ],
)
def test_reward_cap_tapers_by_100_per_day_and_floors_at_zero(day, expected_cap):
    # Day 0 starts at $3,000 and each subsequent day decrements by $100,
    # reaching $0 at day 30 and never going negative thereafter.
    assert Validator._reward_cap_for_day(day) == pytest.approx(expected_cap)


@pytest.mark.parametrize(
    "now, expected_day",
    [
        (datetime(2026, 6, 1, 18, 0, tzinfo=UTC), 0),    # well before anchor -> clamped
        (datetime(2026, 6, 24, 17, 59, tzinfo=UTC), 0),  # just before the day-0 boundary
        (datetime(2026, 6, 24, 18, 0, tzinfo=UTC), 0),   # anchor exactly = day 0
        (datetime(2026, 6, 25, 17, 59, tzinfo=UTC), 0),  # still within day 0
        (datetime(2026, 6, 25, 18, 0, tzinfo=UTC), 1),   # one cycle later
        (datetime(2026, 7, 24, 17, 0, tzinfo=UTC), 29),  # just before day-30 boundary
        (datetime(2026, 7, 24, 18, 0, tzinfo=UTC), 30),  # day 30
    ],
)
def test_current_taper_day_counts_cycles_from_anchor(validator, now, expected_day):
    # Anchor = Validator.TAPER_START_DATE (2026-06-24) at the hardcoded
    # TAPER_ANCHOR_HOUR_UTC (18:00 UTC). Day index increments at each 24h
    # boundary and clamps to 0 pre-anchor.
    assert validator._current_taper_day(now) == expected_day


def test_current_taper_day_ignores_operator_eval_schedule(validator):
    # Consensus property: the taper boundary is hardcoded (18:00 UTC) and must
    # NOT shift when an operator changes their eval schedule — otherwise
    # validators would taper onto different days. With the boundary at 18:00,
    # 17:59 the next day is still day 0 and 18:00 is day 1, regardless of the
    # configured schedule_hour/minute.
    validator.config.validation_data_schedule_hour = 2
    validator.config.validation_data_schedule_minute = 30
    assert validator._current_taper_day(datetime(2026, 6, 25, 17, 59, tzinfo=UTC)) == 0
    assert validator._current_taper_day(datetime(2026, 6, 25, 18, 0, tzinfo=UTC)) == 1


@pytest.mark.asyncio
async def test_zero_cap_burns_everything(validator):
    # When the tapered Reward Cap reaches $0, 100% of emission is burned:
    # the burn UID gets all the weight and miners get none.
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=5.0),
        create_mock_neuron(1, "hk1", emission=5.0),
        create_mock_neuron(2, "hk2", emission=0.0),  # Burn UID
    ]
    weights = {"hk0": 0.5, "hk1": 0.5}

    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
        patch.object(validator, "_current_reward_cap_usd", return_value=0.0),
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        adjusted = await validator._apply_burn(weights)

    assert adjusted["hk2"] == pytest.approx(1.0)
    assert adjusted.get("hk0", 0.0) == pytest.approx(0.0)
    assert adjusted.get("hk1", 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_zero_cap_burns_everything_even_when_price_feed_is_down(validator):
    # A dead price feed makes get_tao_price_usd() return 0.0. At the $0 floor
    # the burn must NOT depend on prices, otherwise an outage would re-enable
    # 100% miner payouts at the moment we want everything burned.
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=5.0),
        create_mock_neuron(1, "hk1", emission=5.0),
        create_mock_neuron(2, "hk2", emission=0.0),  # Burn UID
    ]
    weights = {"hk0": 0.5, "hk1": 0.5}

    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
        patch.object(validator, "_current_reward_cap_usd", return_value=0.0),
    ):
        mock_tao.return_value = 0.0  # price feed down
        mock_alpha.return_value = 0.0
        adjusted = await validator._apply_burn(weights)

    assert adjusted["hk2"] == pytest.approx(1.0)
    assert adjusted.get("hk0", 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_apply_burn_below_cap_does_not_burn(validator):
    # At day 0 the cap is $3,000. Miner value of $20/day is below it, so the
    # Cap Burn is 0 and weights pass through unchanged (no burn allocation).
    emission_per_tempo = 10.0 / 20.0
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=emission_per_tempo),
        create_mock_neuron(1, "hk1", emission=emission_per_tempo),
        create_mock_neuron(2, "hk2", emission=0.0),  # Burn UID
    ]
    weights = {"hk0": 0.5, "hk1": 0.5}

    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
        patch.object(validator, "_current_taper_day", return_value=0),  # day-0 cap = $3,000
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        adjusted = await validator._apply_burn(weights)

    assert adjusted["hk0"] == pytest.approx(0.5)
    assert adjusted["hk1"] == pytest.approx(0.5)
    assert adjusted.get("hk2", 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_apply_burn_above_cap_burns_excess(validator):
    # At day 0 the cap is $3,000. Miner value of $4,000,000/day is far above it,
    # so the Cap Burn keeps the distributed value at the cap.
    # 1,000,000 Alpha/day per miner; steps_per_day = 20 -> emission_per_tempo.
    emission_per_tempo = 1_000_000.0 / 20.0
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = [
        create_mock_neuron(0, "hk0", emission=emission_per_tempo),
        create_mock_neuron(1, "hk1", emission=emission_per_tempo),
        create_mock_neuron(2, "hk2", emission=0.0),  # Burn UID
    ]
    weights = {"hk0": 0.5, "hk1": 0.5}

    with (
        patch("real_estate.validator.validator.get_tao_price_usd", new_callable=AsyncMock) as mock_tao,
        patch("real_estate.validator.validator.get_alpha_price_tao", new_callable=AsyncMock) as mock_alpha,
        patch.object(validator, "_current_taper_day", return_value=0),  # day-0 cap = $3,000
    ):
        mock_tao.return_value = 200.0
        mock_alpha.return_value = 0.01
        # Daily miner value = 2M Alpha * 0.01 TAO * $200 = $4,000,000.
        adjusted = await validator._apply_burn(weights)

    # Cap Burn is the sole burn authority: burn_amount == cap_burn.
    expected_burn = 1.0 - (3000.0 / 4_000_000.0)
    assert adjusted["hk2"] == pytest.approx(expected_burn)
    assert adjusted["hk0"] == pytest.approx(0.5 * (1.0 - expected_burn))
