from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from real_estate.chain.models import Neuron
from real_estate.validator import Validator


def create_mock_neuron(
    uid: int,
    hotkey: str,
    validator_permit: bool = False,
    emission: float = 1.0,
    incentive: float = 0.1,
) -> Neuron:
    return Neuron(
        uid=uid,
        hotkey=hotkey,
        coldkey=f"coldkey_{uid}",
        stake=100.0,
        trust=0.5,
        consensus=0.5,
        incentive=incentive,
        dividends=0.1,
        emission=emission,
        is_active=True,
        validator_permit=validator_permit,
    )


@pytest.fixture
def mock_config(tmp_path) -> MagicMock:
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
    # Price-window state persists here; a temp dir keeps tests isolated + on disk.
    config.model_cache_path = tmp_path
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
        # Skip the on-disk state so each test controls it directly.
        v._price_window_loaded = True
        v._last_alpha_loaded = True
        v._last_alpha_tao = 0.0
        return v


def _set_neurons(
    validator, emission_per_tempo: float, n_miners: int = 2, winner_permit: bool = False
) -> None:
    """Two miners with incentive, plus a pure-validator burn UID (incentive=0).

    `winner_permit=True` gives the miners a validator_permit to prove they are
    still counted as miners (they earn incentive) despite holding the permit.
    """
    neurons = [
        create_mock_neuron(
            i,
            f"hk{i}",
            emission=emission_per_tempo,
            incentive=0.5,
            validator_permit=winner_permit,
        )
        for i in range(n_miners)
    ]
    # burn UID: a pure validator — has a permit, earns no incentive.
    neurons.append(
        create_mock_neuron(2, "hk2", emission=0.0, incentive=0.0, validator_permit=True)
    )
    validator.metagraph = MagicMock()
    validator.metagraph.neurons = neurons


def _price_patches():
    return (
        patch(
            "real_estate.validator.validator.get_tao_price_usd",
            new_callable=AsyncMock,
            return_value=200.0,
        ),
        patch(
            "real_estate.validator.validator.get_alpha_price_tao",
            new_callable=AsyncMock,
            return_value=0.01,
        ),
    )


@pytest.mark.asyncio
async def test_under_limit_no_cap_burn(validator):
    """A modest miner rate stays under $3k → no cap burn (manual burn only)."""
    # 2 * 0.5 * 20 = 20 alpha/day * (0.01 * 200 = $2) = $40/day — far under limit.
    _set_neurons(validator, emission_per_tempo=10.0 / 20.0)
    weights = {"hk0": 0.5, "hk1": 0.5}
    p_tao, p_alpha = _price_patches()
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 0.1
        adjusted = await validator._apply_burn(weights)

    # cap_burn = 0 → burn_amount = 1 - (1)(1-0.1) = 0.1
    assert adjusted["hk2"] == pytest.approx(0.1)
    assert adjusted["hk0"] == pytest.approx(0.5 * 0.9)


@pytest.mark.asyncio
async def test_winner_with_validator_permit_still_counted(validator):
    """A miner that gained a validator_permit is still measured as a miner.

    This is the bug the redesign fixes: filtering on `not validator_permit` would
    exclude the winner and read miner rewards as ~$0, so nothing gets burned.
    """
    # Big rate that SHOULD trip the cap: 2 * 50000 * 20 = 2,000,000 alpha/day
    # * $2 = $4,000,000/day. The miners carry a validator_permit here.
    _set_neurons(validator, emission_per_tempo=50_000.0, winner_permit=True)
    weights = {"hk0": 0.5, "hk1": 0.5}
    p_tao, p_alpha = _price_patches()
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 0.0
        adjusted = await validator._apply_burn(weights)

    # Because the permit-holding miners are still counted, the cap fires.
    assert adjusted["hk2"] > 0.99  # nearly everything burned


@pytest.mark.asyncio
async def test_partial_burn_at_overshoot(validator):
    """Rate above the limit → burn only the overshoot down to the limit."""
    # 2 * 50000 * 20 = 2,000,000 alpha/day * $2 = $4,000,000/day.
    _set_neurons(validator, emission_per_tempo=50_000.0)
    weights = {"hk0": 0.5, "hk1": 0.5}
    p_tao, p_alpha = _price_patches()
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 0.0
        adjusted = await validator._apply_burn(weights)

    expected_burn = 1.0 - Validator.REWARD_LIMIT_USD / 4_000_000.0
    assert adjusted["hk2"] == pytest.approx(expected_burn)
    assert adjusted["hk0"] == pytest.approx(0.5 * (1.0 - expected_burn))


@pytest.mark.asyncio
async def test_manual_burn_overrides_everything(validator):
    """MANUAL_BURN = 1.0 forces 100% burn regardless of rate."""
    _set_neurons(validator, emission_per_tempo=10.0 / 20.0)
    weights = {"hk0": 1.0}
    p_tao, p_alpha = _price_patches()
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 1.0
        adjusted = await validator._apply_burn(weights)

    assert adjusted["hk2"] == pytest.approx(1.0)
    assert adjusted.get("hk0", 0.0) == 0.0


@pytest.mark.asyncio
async def test_untrusted_price_skips_cap_but_keeps_manual_burn(validator):
    """A 0.0 alpha price (failed chain call) must NOT burn via the cap — burning on
    a bogus price is what starves miners — but MANUAL_BURN still applies."""
    # A rate that WOULD trip the cap hard if it were priced.
    _set_neurons(validator, emission_per_tempo=50_000.0)
    weights = {"hk0": 0.5, "hk1": 0.5}
    p_tao = patch(
        "real_estate.validator.validator.get_tao_price_usd",
        new_callable=AsyncMock,
        return_value=200.0,
    )
    p_alpha = patch(
        "real_estate.validator.validator.get_alpha_price_tao",
        new_callable=AsyncMock,
        return_value=0.0,  # chain price call failed
    )
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 0.0
        adjusted = await validator._apply_burn(weights)
    # No cap burn: weights returned unchanged, nothing routed to the burn UID.
    assert adjusted == weights

    # But a manual burn is price-independent and must still fire.
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 1.0
        adjusted = await validator._apply_burn(weights)
    assert adjusted["hk2"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_untrusted_price_holds_last_good(validator):
    """A failed fetch with a known last-good price keeps capping on it (no uncap)."""
    validator._last_alpha_tao = 0.01  # last good price in TAO
    _set_neurons(validator, emission_per_tempo=50_000.0)  # 2,000,000 α/day
    weights = {"hk0": 0.5, "hk1": 0.5}
    p_tao = patch(
        "real_estate.validator.validator.get_tao_price_usd",
        new_callable=AsyncMock,
        return_value=200.0,
    )
    p_alpha = patch(
        "real_estate.validator.validator.get_alpha_price_tao",
        new_callable=AsyncMock,
        return_value=0.0,  # fetch failed
    )
    with p_tao, p_alpha:
        validator.MANUAL_BURN = 0.0
        adjusted = await validator._apply_burn(weights)
    # 2,000,000 α/day * ($0.01*200=$2) = $4,000,000/day → cap fires on last-good.
    expected_burn = 1.0 - Validator.REWARD_LIMIT_USD / 4_000_000.0
    assert adjusted["hk2"] == pytest.approx(expected_burn)


def test_cap_burn_math(validator):
    """_cap_burn burns the overshoot and nothing when under the limit."""
    limit = Validator.REWARD_LIMIT_USD
    assert validator._cap_burn(limit * 0.5, 1.0) == 0.0  # under the limit
    assert validator._cap_burn(0.0, 5.0) == 0.0  # no emission
    # 2x the limit → burn half to land back at the limit.
    assert validator._cap_burn(limit * 2.0, 1.0) == pytest.approx(0.5)


def test_rolling_average_smooths_price(validator):
    """The window averages spot prices, damping intraday swings."""
    validator._price_window = []
    assert validator._avg_alpha_usd(2.0) == pytest.approx(2.0)
    assert validator._avg_alpha_usd(4.0) == pytest.approx(3.0)
    assert validator._avg_alpha_usd(6.0) == pytest.approx(4.0)


def test_window_capped_at_sample_limit(validator):
    """The window keeps only the last PRICE_WINDOW_SAMPLES prices."""
    validator._price_window = []
    for _ in range(Validator.PRICE_WINDOW_SAMPLES + 5):
        validator._avg_alpha_usd(1.0)
    assert len(validator._price_window) == Validator.PRICE_WINDOW_SAMPLES


def test_price_window_survives_restart(validator):
    """Samples persist to disk and reload, so a restart keeps the window."""
    validator._price_window = []
    validator._avg_alpha_usd(1.5)
    validator._avg_alpha_usd(2.5)

    # Simulate a restart: wipe in-memory state, reload from disk.
    validator._price_window = []
    validator._price_window_loaded = False
    validator._load_price_window()

    assert validator._price_window == pytest.approx([1.5, 2.5])
