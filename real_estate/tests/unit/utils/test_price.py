from unittest.mock import MagicMock, patch

import pytest

from real_estate.utils.price import _price_cache, get_alpha_price_tao, get_tao_price_usd


@pytest.fixture(autouse=True)
def clear_cache():
    _price_cache.clear()


@pytest.mark.asyncio
async def test_get_tao_price_usd_mexc_success():
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_res = MagicMock()
        mock_res.raise_for_status = MagicMock()
        mock_res.json = MagicMock(return_value={"price": "200.0"})
        mock_get.return_value = mock_res

        price = await get_tao_price_usd()
        assert price == 200.0


@pytest.mark.asyncio
async def test_get_tao_price_usd_coingecko_success():
    # Force MEXC to fail to test CoinGecko
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_res_mexc = MagicMock()
        mock_res_mexc.raise_for_status.side_effect = Exception("MEXC Fail")

        mock_res_cg = MagicMock()
        mock_res_cg.raise_for_status = MagicMock()
        mock_res_cg.json = MagicMock(return_value={"bittensor": {"usd": 205.0}})

        mock_get.side_effect = [mock_res_mexc, mock_res_cg]

        price = await get_tao_price_usd()
        assert price == 205.0


@pytest.mark.asyncio
async def test_get_tao_price_usd_all_fail_returns_cache():
    from real_estate.utils.price import _price_cache

    _price_cache["tao_usd"] = 190.0

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("All Fail")

        price = await get_tao_price_usd()
        assert price == 190.0


def _scale(value):
    """Mimic a substrate query result: .value is a single-element list."""
    m = MagicMock()
    m.value = [value]
    return m


def _reserves_subtensor(tao_rao, alpha_rao):
    st = MagicMock()
    st.query_subtensor.side_effect = lambda name, **_: _scale(
        tao_rao if name == "SubnetTAO" else alpha_rao
    )
    return st


@pytest.mark.asyncio
async def test_get_alpha_price_from_reserves():
    # Real on-chain reserves from SN46; ratio is TAO per alpha (rao cancels).
    st = _reserves_subtensor(11048307774624, 1833491530986819)
    price = await get_alpha_price_tao(st, 46)
    assert price == pytest.approx(0.006026, rel=1e-3)


@pytest.mark.asyncio
async def test_get_alpha_price_unavailable_returns_zero_not_one():
    """A failed query must return 0.0 (unknown), never 1.0 — 1.0 over-burns."""
    st = MagicMock()
    st.query_subtensor.side_effect = ValueError("SwapRuntimeApi/runtime mismatch")
    price = await get_alpha_price_tao(st, 46)
    assert price == 0.0


@pytest.mark.asyncio
async def test_get_alpha_price_failure_keeps_last_good():
    _price_cache["alpha_46"] = 0.0061  # a prior good read
    st = MagicMock()
    st.query_subtensor.side_effect = Exception("broken")
    price = await get_alpha_price_tao(st, 46)
    assert price == 0.0061


@pytest.mark.asyncio
async def test_get_alpha_price_rejects_implausible():
    """A ratio >= 1 TAO is not a real alpha price → treated as unknown (0.0)."""
    st = _reserves_subtensor(5, 1)  # ratio 5.0 TAO — implausible
    price = await get_alpha_price_tao(st, 46)
    assert price == 0.0
