"""Tests for ScraperClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from real_estate.data import (
    ScraperAuthError,
    ScraperClient,
    ScraperConfig,
    ScraperRequestError,
)


@pytest.fixture
def config():
    """Create test scraper config."""
    return ScraperConfig(
        url="https://scraper.example.com",
        realm="testnet",
    )


@pytest.fixture
def mock_keypair():
    """Create mock Bittensor keypair."""
    keypair = MagicMock()
    keypair.ss58_address = "5MockHotkeyAddress"
    keypair.sign.return_value = b"mocksignature"
    return keypair


@pytest.fixture
def client(config, mock_keypair):
    """Create scraper client with mocked keypair."""
    return ScraperClient(config, mock_keypair)


class TestGetValidationData:
    """Tests for get_validation_data method."""

    async def test_successful_fetch(self, client):
        """Returns ValidationDataset on successful response."""
        mock_response = httpx.Response(
            200,
            json={"properties": [{"id": 1, "price": 500000}]},
            request=httpx.Request("POST", "https://scraper.example.com"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.get_validation_data()

        assert result.properties == [{"id": 1, "price": 500000}]

    async def test_401_raises_auth_error(self, client):
        """401 response raises ScraperAuthError."""
        mock_response = httpx.Response(401, text="Invalid signature")

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ScraperAuthError, match="Authentication failed"):
                await client.get_validation_data()

    async def test_403_raises_auth_error(self, client):
        """403 response raises ScraperAuthError."""
        mock_response = httpx.Response(403, text="Hotkey not in whitelist")

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ScraperAuthError, match="Hotkey not authorized"):
                await client.get_validation_data()

    async def test_500_raises_request_error(self, client):
        """500 response raises ScraperRequestError."""
        mock_response = httpx.Response(500, text="Internal server error")
        mock_response.request = httpx.Request("POST", "https://scraper.example.com")

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ScraperRequestError, match="Request failed"):
                await client.get_validation_data()

    async def test_connection_error_raises_request_error(self, client):
        """Connection error raises ScraperRequestError."""
        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(ScraperRequestError, match="Connection error"):
                await client.get_validation_data()

    async def test_invalid_json_raises_request_error(self, client):
        """Invalid JSON response raises ScraperRequestError."""
        mock_response = httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", "https://scraper.example.com"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ScraperRequestError, match="Invalid JSON"):
                await client.get_validation_data()

    async def test_missing_properties_key_raises_request_error(self, client):
        """Missing 'properties' key raises ScraperRequestError."""
        mock_response = httpx.Response(
            200,
            json={"data": []},
            request=httpx.Request("POST", "https://scraper.example.com"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ScraperRequestError, match="missing 'properties'"):
                await client.get_validation_data()


class TestFetchWithRetry:
    """Tests for fetch_with_retry method."""

    async def test_retries_on_scraper_request_error(self, mock_keypair):
        """Retries on ScraperRequestError until success."""
        config = ScraperConfig(
            url="https://scraper.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ScraperClient(config, mock_keypair)

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ScraperRequestError("temporary issue")
            return MagicMock(properties=[])

        client.get_validation_data = AsyncMock(side_effect=fail_then_succeed)

        result = await client.fetch_with_retry()

        assert call_count == 2  # 1 failure + 1 success
        assert hasattr(result, "properties")

    async def test_does_not_retry_on_auth_error(self, mock_keypair):
        """Does not retry on ScraperAuthError."""
        config = ScraperConfig(
            url="https://scraper.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ScraperClient(config, mock_keypair)

        client.get_validation_data = AsyncMock(side_effect=ScraperAuthError("bad key"))

        with pytest.raises(ScraperAuthError):
            await client.fetch_with_retry()

        assert client.get_validation_data.call_count == 1


class TestStartScheduled:
    """Tests for start_scheduled method."""

    def test_creates_scheduler_and_adds_job(self, client):
        """Creates scheduler with correct job configuration."""
        with patch("real_estate.data.scraper_client.AsyncIOScheduler") as mock_scheduler_cls:
            scheduler = mock_scheduler_cls.return_value
            on_fetch = MagicMock()

            result = client.start_scheduled(on_fetch)

            mock_scheduler_cls.assert_called_once_with(timezone="UTC")
            scheduler.add_job.assert_called_once()
            scheduler.start.assert_called_once()
            assert result == scheduler

            _, kwargs = scheduler.add_job.call_args
            assert kwargs["id"] == "validation_data_fetch"

    async def test_scheduled_job_calls_on_fetch(self, client):
        """Scheduled job calls on_fetch callback with data."""
        with patch("real_estate.data.scraper_client.AsyncIOScheduler") as mock_scheduler_cls:
            on_fetch = MagicMock()
            client.start_scheduled(on_fetch)

            # Get the scheduled function
            job_args, _ = mock_scheduler_cls.return_value.add_job.call_args
            scheduled_func = job_args[0]

            # Mock fetch_with_retry and run the scheduled function
            mock_data = MagicMock(properties=[{"id": 1}])
            with patch.object(
                client, "fetch_with_retry", new_callable=AsyncMock, return_value=mock_data
            ):
                await scheduled_func()

            on_fetch.assert_called_once_with(mock_data)


class TestSignRequest:
    """Tests for request signing."""

    def test_sign_request_includes_required_headers(self, client):
        """Signed request includes Hotkey, Nonce, Realm, Signature."""
        headers = client._sign_request("POST", "https://scraper.example.com/api/v1/validation-data")

        assert "Hotkey" in headers
        assert "Nonce" in headers
        assert "Realm" in headers
        assert "Signature" in headers
        assert headers["Realm"] == "testnet"

    def test_sign_request_uses_keypair_address(self, client, mock_keypair):
        """Signed request uses keypair's ss58_address."""
        headers = client._sign_request("POST", "https://example.com")

        assert headers["Hotkey"] == mock_keypair.ss58_address
