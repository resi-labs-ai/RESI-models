import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from real_estate.utils.price import get_tao_price_usd, get_alpha_price_tao, _price_cache

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

@pytest.mark.asyncio
async def test_get_alpha_price_tao_success():
    mock_subtensor = MagicMock()
    mock_price = MagicMock()
    mock_price.tao = 0.012
    mock_subtensor.get_subnet_price.return_value = mock_price
    
    price = await get_alpha_price_tao(mock_subtensor, 46)
    assert price == 0.012

@pytest.mark.asyncio
async def test_get_alpha_price_tao_fallback():
    mock_subtensor = MagicMock()
    # Make get_subnet_price return None to trigger fallback
    mock_subtensor.get_subnet_price.return_value = None
    
    class MockInfo:
        def __init__(self):
            self.price = 0.015
            
    mock_info = MockInfo()
    mock_subtensor.get_subnet_info.return_value = mock_info
    
    price = await get_alpha_price_tao(mock_subtensor, 46)
    assert price == 0.015
