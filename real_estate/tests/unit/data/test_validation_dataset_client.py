"""Tests for ValidationDatasetClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from real_estate.data import (
    RawFileInfo,
    ValidationDataAuthError,
    ValidationDataNotFoundError,
    ValidationDataProcessingError,
    ValidationDataRateLimitError,
    ValidationDataRequestError,
    ValidationDatasetClient,
    ValidationDatasetClientConfig,
    ValidationDatasetResponse,
)


@pytest.fixture
def config():
    """Create test validation set config."""
    return ValidationDatasetClientConfig(
        url="https://dashboard.example.com",
        timeout=30.0,
        max_retries=2,
        retry_delay_seconds=0,  # No delay in tests
    )


@pytest.fixture
def mock_keypair():
    """Create mock Bittensor keypair."""
    keypair = MagicMock()
    keypair.ss58_address = "5MockValidatorHotkey"
    keypair.sign.return_value = b"mocksignature"
    return keypair


@pytest.fixture
def client(config, mock_keypair):
    """Create validation client with mocked keypair."""
    return ValidationDatasetClient(config, mock_keypair)


@pytest.fixture
def mock_validation_response():
    """Mock successful validation set API response."""
    return {
        "validatorUid": 42,
        "validationDate": "2025-12-29",
        "expiresAt": "2025-12-29T15:30:00.000Z",
        "validationSet": {
            "filename": "validation_set.json",
            "presignedUrl": "https://s3.example.com/validation_set.json",
            "fileSize": 15728640,
        },
        "rawDataFiles": [
            {
                "filename": "AL_20251229_140027.json",
                "presignedUrl": "https://s3.example.com/AL_20251229_140027.json",
                "fileSize": 524288,
            },
            {
                "filename": "AZ_20251229_140028.json",
                "presignedUrl": "https://s3.example.com/AZ_20251229_140028.json",
                "fileSize": 1048576,
            },
        ],
    }


@pytest.fixture
def mock_validation_data():
    """Mock validation set data."""
    return [
        {
            "property_id": "prop_001",
            "price": 450000,
            "beds": 3,
            "baths": 2,
        },
        {
            "property_id": "prop_002",
            "price": 325000,
            "beds": 2,
            "baths": 1.5,
        },
    ]


@pytest.fixture
def mock_processing_response():
    """Mock 'processing' status API response (200 with status: processing)."""
    return {
        "status": "processing",
        "validationDate": "2025-12-30",
        "message": "Validation set for 2025-12-30 is still being processed. Data scraping started at 2:00 PM EST and typically completes around 3:00 PM EST. Please check back soon.",
        "estimatedReadyTime": "2025-12-30T20:00:00.000Z",
        "retryAfter": 3600,
    }


@pytest.fixture
def mock_not_found_response():
    """Mock 'not_found' status API response (404)."""
    return {
        "status": "not_found",
        "validationDate": "2025-01-15",
        "message": "No validation set exists for 2025-01-15",
        "error": "NO_VALIDATION_SET",
        "code": "NO_VALIDATION_SET",
    }


class TestGetValidationUrls:
    """Tests for get_validation_urls method."""

    async def test_successful_fetch_returns_urls(
        self, client, mock_validation_response
    ):
        """Returns ValidationDatasetResponse on successful response."""
        mock_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.get_validation_urls()

        assert isinstance(result, ValidationDatasetResponse)
        assert result.validator_uid == 42
        assert result.validation_date == "2025-12-29"
        assert result.validation_set_url == "https://s3.example.com/validation_set.json"
        assert len(result.raw_files) == 2
        assert result.raw_files[0].filename == "AL_20251229_140027.json"

    async def test_successful_fetch_with_date_parameter(
        self, client, mock_validation_response
    ):
        """Includes date parameter in request."""
        mock_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set?date=2025-12-25"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.get_validation_urls(date="2025-12-25")

        assert result.validation_date == "2025-12-29"

    async def test_processing_status_raises_processing_error(
        self, client, mock_processing_response
    ):
        """200 response with status: 'processing' raises ValidationDataProcessingError."""
        mock_response = httpx.Response(
            200,
            json=mock_processing_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataProcessingError) as exc_info:
                await client.get_validation_urls()

        error = exc_info.value
        assert error.validation_date == "2025-12-30"
        assert error.estimated_ready_time == "2025-12-30T20:00:00.000Z"
        assert error.retry_after == 3600
        assert "still being processed" in str(error)

    async def test_processing_status_with_date_parameter(
        self, client, mock_processing_response
    ):
        """Processing status with date parameter includes retry information."""
        mock_response = httpx.Response(
            200,
            json=mock_processing_response,
            request=httpx.Request(
                "POST",
                "https://dashboard.example.com/api/auth/validation-set?date=2025-12-30",
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataProcessingError) as exc_info:
                await client.get_validation_urls(date="2025-12-30")

        error = exc_info.value
        assert error.validation_date == "2025-12-30"
        assert error.retry_after == 3600

    async def test_401_raises_auth_error(self, client):
        """401 response raises ValidationDataAuthError."""
        mock_response = httpx.Response(401, text="Invalid signature")

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataAuthError, match="Authentication failed"):
                await client.get_validation_urls()

    async def test_404_raises_not_found_error(self, client):
        """404 response raises ValidationDataNotFoundError."""
        mock_response = httpx.Response(404, text="No data for date")

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataNotFoundError, match="Data not found"):
                await client.get_validation_urls()

    async def test_429_raises_rate_limit_error(self, client):
        """429 response raises ValidationDataRateLimitError."""
        mock_response = httpx.Response(429, text="Rate limit exceeded")

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataRateLimitError, match="Rate limited"):
                await client.get_validation_urls()

    async def test_500_raises_request_error(self, client):
        """500 response raises ValidationDataRequestError."""
        mock_response = httpx.Response(500, text="Internal server error")
        mock_response.request = httpx.Request(
            "POST", "https://dashboard.example.com/api/auth/validation-set"
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataRequestError, match="Request failed"):
                await client.get_validation_urls()

    async def test_connection_error_raises_request_error(self, client):
        """Connection error raises ValidationDataRequestError."""
        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(ValidationDataRequestError, match="Connection error"):
                await client.get_validation_urls()

    async def test_invalid_json_raises_request_error(self, client):
        """Invalid JSON response raises ValidationDataRequestError."""
        mock_response = httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValidationDataRequestError, match="Invalid JSON"):
                await client.get_validation_urls()

    async def test_missing_validation_set_key_raises_error(self, client):
        """Missing 'validationSet' key raises ValidationDataRequestError."""
        mock_response = httpx.Response(
            200,
            json={"rawDataFiles": []},
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(
                ValidationDataRequestError, match="missing 'validationSet'"
            ):
                await client.get_validation_urls()

    async def test_missing_raw_data_files_key_raises_error(self, client):
        """Missing 'rawDataFiles' key raises ValidationDataRequestError."""
        mock_response = httpx.Response(
            200,
            json={"validationSet": {}},
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(
                ValidationDataRequestError, match="missing 'rawDataFiles'"
            ):
                await client.get_validation_urls()


class TestDownloadValidationSet:
    """Tests for download_validation_set method."""

    async def test_successful_download_and_parse(
        self, client, mock_validation_response, mock_validation_data
    ):
        """Successfully downloads and parses validation set."""
        # Mock API response
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        # Mock S3 download
        s3_response = httpx.Response(
            200,
            json=mock_validation_data,
            request=httpx.Request("GET", "https://s3.example.com/validation_set.json"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=s3_response
        ):
            result = await client.download_validation_set()

        assert len(result) == 2
        assert result.properties == mock_validation_data

    async def test_connection_error_downloading_data(
        self, client, mock_validation_response
    ):
        """Connection error during S3 download raises ValidationDataRequestError."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection failed"),
        ):
            with pytest.raises(
                ValidationDataRequestError, match="Failed to download validation set"
            ):
                await client.download_validation_set()

    async def test_invalid_json_in_downloaded_data(
        self, client, mock_validation_response
    ):
        """Invalid JSON in downloaded data raises ValidationDataRequestError."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        s3_response = httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("GET", "https://s3.example.com/validation_set.json"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=s3_response
        ):
            with pytest.raises(
                ValidationDataRequestError, match="Invalid JSON in validation set"
            ):
                await client.download_validation_set()


class TestDownloadRawFiles:
    """Tests for download_raw_files method."""

    async def test_successful_download_all_states(
        self, client, mock_validation_response
    ):
        """Successfully downloads all raw state files."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        al_data = {"state": "AL", "properties": [{"id": 1}]}
        az_data = {"state": "AZ", "properties": [{"id": 2}]}

        async def mock_get(url):
            if "AL_" in url:
                return httpx.Response(
                    200, json=al_data, request=httpx.Request("GET", url)
                )
            elif "AZ_" in url:
                return httpx.Response(
                    200, json=az_data, request=httpx.Request("GET", url)
                )
            return httpx.Response(404, request=httpx.Request("GET", url))

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, side_effect=mock_get
        ):
            result = await client.download_raw_files()

        assert len(result) == 2
        assert "AL" in result
        assert "AZ" in result
        assert result["AL"] == al_data
        assert result["AZ"] == az_data

    async def test_extracts_state_from_filename(
        self, client, mock_validation_response
    ):
        """Correctly extracts state code from filename."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        mock_data = {"data": "test"}

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=httpx.Response(
                200, json=mock_data, request=httpx.Request("GET", "https://s3.example.com/file.json")
            ),
        ):
            result = await client.download_raw_files()

        # Verify state codes are extracted correctly
        assert "AL" in result
        assert "AZ" in result

    async def test_handles_partial_download_failures(
        self, client, mock_validation_response
    ):
        """Continues downloading even if some files fail."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        az_data = {"state": "AZ", "properties": [{"id": 2}]}

        async def mock_get(url):
            if "AL_" in url:
                raise httpx.ConnectError("Connection failed")
            elif "AZ_" in url:
                return httpx.Response(
                    200, json=az_data, request=httpx.Request("GET", url)
                )
            return httpx.Response(404, request=httpx.Request("GET", url))

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, side_effect=mock_get
        ):
            result = await client.download_raw_files()

        # Should have AZ but not AL
        assert len(result) == 1
        assert "AZ" in result
        assert "AL" not in result


class TestDownloadRawFile:
    """Tests for download_raw_file method."""

    async def test_successful_download_specific_state(
        self, client, mock_validation_response
    ):
        """Successfully downloads specific state file."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        al_data = {"state": "AL", "properties": [{"id": 1}]}
        s3_response = httpx.Response(
            200, json=al_data, request=httpx.Request("GET", "https://s3.example.com/AL.json")
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=s3_response
        ):
            result = await client.download_raw_file("AL")

        assert result == al_data

    async def test_state_not_in_response_raises_error(
        self, client, mock_validation_response
    ):
        """Raises ValidationDataNotFoundError if state not in response."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ):
            with pytest.raises(
                ValidationDataNotFoundError, match="No raw file found for state TX"
            ):
                await client.download_raw_file("TX")

    async def test_connection_error_downloading(
        self, client, mock_validation_response
    ):
        """Connection error raises ValidationDataRequestError."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection failed"),
        ):
            with pytest.raises(
                ValidationDataRequestError, match="Failed to download raw file"
            ):
                await client.download_raw_file("AL")


class TestFetchWithRetry:
    """Tests for fetch_with_retry method."""

    async def test_retries_on_request_error(self, mock_keypair):
        """Retries on ValidationDataRequestError until success."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        call_count = 0

        async def fail_then_succeed(date=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationDataRequestError("temporary issue")
            return [{"id": 1}]

        client.download_validation_set = AsyncMock(side_effect=fail_then_succeed)

        result = await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert call_count == 2  # 1 failure + 1 success
        assert result[0] == [{"id": 1}]
        assert result[1] is None

    async def test_does_not_retry_on_auth_error(self, mock_keypair):
        """Does not retry on ValidationDataAuthError."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        client.download_validation_set = AsyncMock(
            side_effect=ValidationDataAuthError("bad key")
        )

        with pytest.raises(ValidationDataAuthError):
            await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert client.download_validation_set.call_count == 1

    async def test_does_not_retry_on_not_found_error(self, mock_keypair):
        """Does not retry on ValidationDataNotFoundError."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        client.download_validation_set = AsyncMock(
            side_effect=ValidationDataNotFoundError("no data")
        )

        with pytest.raises(ValidationDataNotFoundError):
            await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert client.download_validation_set.call_count == 1

    async def test_retries_on_rate_limit_error(self, mock_keypair):
        """Retries on ValidationDataRateLimitError."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        call_count = 0

        async def rate_limit_then_succeed(date=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationDataRateLimitError("rate limited")
            return [{"id": 1}]

        client.download_validation_set = AsyncMock(side_effect=rate_limit_then_succeed)

        result = await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert call_count == 2
        assert result[0] == [{"id": 1}]

    async def test_retries_on_processing_error(self, mock_keypair):
        """Retries on ValidationDataProcessingError (data still being processed)."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=3,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        call_count = 0

        async def processing_then_succeed(date=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationDataProcessingError(
                    "Validation set is still being processed",
                    validation_date="2025-12-30",
                    estimated_ready_time="2025-12-30T20:00:00.000Z",
                    retry_after=3600,
                )
            return [{"id": 1}]

        client.download_validation_set = AsyncMock(side_effect=processing_then_succeed)

        result = await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert call_count == 2  # 1 processing + 1 success
        assert result[0] == [{"id": 1}]

    async def test_download_validation_only(self, mock_keypair):
        """Downloads only validation set when download_raw=False."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=1,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        validation_data = [{"id": 1}]
        client.download_validation_set = AsyncMock(return_value=validation_data)
        client.download_raw_files = AsyncMock(return_value={"AL": {}})

        result = await client.fetch_with_retry(download_validation=True, download_raw=False)

        assert result[0] == validation_data
        assert result[1] is None
        client.download_validation_set.assert_called_once()
        client.download_raw_files.assert_not_called()

    async def test_download_raw_only(self, mock_keypair):
        """Downloads only raw files when download_validation=False."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=1,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        raw_data = {"AL": {}}
        client.download_validation_set = AsyncMock(return_value=[{"id": 1}])
        client.download_raw_files = AsyncMock(return_value=raw_data)

        result = await client.fetch_with_retry(download_validation=False, download_raw=True)

        assert result[0] is None
        assert result[1] == raw_data
        client.download_validation_set.assert_not_called()
        client.download_raw_files.assert_called_once()

    async def test_download_both(self, mock_keypair):
        """Downloads both validation set and raw files."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=1,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        validation_data = [{"id": 1}]
        raw_data = {"AL": {}}
        client.download_validation_set = AsyncMock(return_value=validation_data)
        client.download_raw_files = AsyncMock(return_value=raw_data)

        result = await client.fetch_with_retry(download_validation=True, download_raw=True)

        assert result[0] == validation_data
        assert result[1] == raw_data
        client.download_validation_set.assert_called_once()
        client.download_raw_files.assert_called_once()

    async def test_download_neither_returns_none(self, mock_keypair):
        """Returns (None, None) when both flags are False."""
        config = ValidationDatasetClientConfig(
            url="https://dashboard.example.com",
            max_retries=1,
            retry_delay_seconds=0,
        )
        client = ValidationDatasetClient(config, mock_keypair)

        client.download_validation_set = AsyncMock(return_value=[{"id": 1}])
        client.download_raw_files = AsyncMock(return_value={"AL": {}})

        result = await client.fetch_with_retry(download_validation=False, download_raw=False)

        assert result[0] is None
        assert result[1] is None
        client.download_validation_set.assert_not_called()
        client.download_raw_files.assert_not_called()


class TestSignRequest:
    """Tests for request signing."""

    def test_sign_request_constructs_correct_message(self, client):
        """Signed request constructs correct message format."""
        nonce = "1703923200.123"
        url = "https://dashboard.example.com/api/auth/validation-set"

        headers = client._sign_request("POST", url, nonce)

        # Verify returns dict with headers
        assert isinstance(headers, dict)
        assert "Hotkey" in headers
        assert "Nonce" in headers
        assert "Signature" in headers

        # Verify keypair.sign was called with correct message
        # Note: JSON uses no spaces (separators=(',', ':')) to match JavaScript's JSON.stringify
        expected_message = f'POST{url}{{"Hotkey":"5MockValidatorHotkey","Nonce":"{nonce}"}}'
        client._keypair.sign.assert_called_once()
        call_args = client._keypair.sign.call_args[0][0]
        assert call_args.decode() == expected_message

    def test_sign_request_includes_query_params_in_url(self, client):
        """Query parameters are included in signed URL."""
        nonce = "1703923200.123"
        url = "https://dashboard.example.com/api/auth/validation-set?date=2025-12-25"

        client._sign_request("POST", url, nonce)

        # Verify URL with query params was signed
        call_args = client._keypair.sign.call_args[0][0]
        assert b"?date=2025-12-25" in call_args

    def test_sign_request_headers_sorted_alphabetically(self, client):
        """Headers JSON is sorted alphabetically."""
        nonce = "1703923200.123"
        url = "https://dashboard.example.com/api/auth/validation-set"

        client._sign_request("POST", url, nonce)

        # Verify headers are sorted (Hotkey before Nonce)
        call_args = client._keypair.sign.call_args[0][0].decode()
        assert '{"Hotkey":' in call_args
        assert call_args.index("Hotkey") < call_args.index("Nonce")

    def test_sign_request_returns_headers_dict(self, client):
        """Returns dict with all auth headers."""
        nonce = "1703923200.123"
        url = "https://dashboard.example.com/api/auth/validation-set"

        headers = client._sign_request("POST", url, nonce)

        assert isinstance(headers, dict)
        assert headers["Hotkey"] == "5MockValidatorHotkey"
        assert headers["Nonce"] == nonce
        # Signature is hex-encoded
        assert isinstance(headers["Signature"], str)

    def test_sign_request_uses_keypair_address(self, client, mock_keypair):
        """Signed request uses keypair's ss58_address."""
        nonce = "1703923200.123"
        url = "https://dashboard.example.com/api/auth/validation-set"

        headers = client._sign_request("POST", url, nonce)

        assert headers["Hotkey"] == mock_keypair.ss58_address
        call_args = client._keypair.sign.call_args[0][0].decode()
        assert mock_keypair.ss58_address in call_args


class TestIntegration:
    """Integration tests with mocked responses."""

    async def test_complete_validation_workflow(
        self, client, mock_validation_response, mock_validation_data
    ):
        """Complete workflow: authenticate, get URLs, download data."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        s3_response = httpx.Response(
            200,
            json=mock_validation_data,
            request=httpx.Request("GET", "https://s3.example.com/validation_set.json"),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=s3_response
        ):
            # Get URLs
            urls = await client.get_validation_urls()
            assert urls.validator_uid == 42

            # Download validation set
            data = await client.download_validation_set()
            assert len(data) == 2
            assert data.properties[0]["property_id"] == "prop_001"

    async def test_complete_raw_files_workflow(
        self, client, mock_validation_response
    ):
        """Complete workflow for downloading raw files."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST", "https://dashboard.example.com/api/auth/validation-set"
            ),
        )

        al_data = {"state": "AL"}
        az_data = {"state": "AZ"}

        async def mock_get(url):
            if "AL_" in url:
                return httpx.Response(
                    200, json=al_data, request=httpx.Request("GET", url)
                )
            elif "AZ_" in url:
                return httpx.Response(
                    200, json=az_data, request=httpx.Request("GET", url)
                )
            return httpx.Response(404, request=httpx.Request("GET", url))

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ), patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, side_effect=mock_get
        ):
            # Download all raw files
            raw_data = await client.download_raw_files()
            assert len(raw_data) == 2
            assert raw_data["AL"]["state"] == "AL"
            assert raw_data["AZ"]["state"] == "AZ"

    async def test_date_parameter_in_url_construction(
        self, client, mock_validation_response
    ):
        """Date parameter is correctly included in signed URL."""
        api_response = httpx.Response(
            200,
            json=mock_validation_response,
            request=httpx.Request(
                "POST",
                "https://dashboard.example.com/api/auth/validation-set?date=2025-12-20",
            ),
        )

        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock, return_value=api_response
        ) as mock_request:
            await client.get_validation_urls(date="2025-12-20")

            # Verify the request was made with correct URL
            call_args = mock_request.call_args
            assert "?date=2025-12-20" in call_args[0][1]  # URL argument


class TestStartScheduled:
    """Tests for start_scheduled method."""

    def test_creates_scheduler_and_adds_job(self, client):
        """Creates scheduler with correct job configuration."""
        with patch(
            "real_estate.data.validation_dataset_client.AsyncIOScheduler"
        ) as mock_scheduler_cls:
            scheduler = mock_scheduler_cls.return_value
            on_fetch = MagicMock()

            result = client.start_scheduled(on_fetch)

            mock_scheduler_cls.assert_called_once_with(timezone="UTC")
            scheduler.add_job.assert_called_once()
            scheduler.start.assert_called_once()
            assert result == scheduler

            _, kwargs = scheduler.add_job.call_args
            assert kwargs["id"] == "validation_set_fetch"

    async def test_scheduled_job_calls_on_fetch(self, client, mock_validation_data):
        """Scheduled job calls on_fetch callback with data."""
        from real_estate.data import ValidationDataset

        with patch(
            "real_estate.data.validation_dataset_client.AsyncIOScheduler"
        ) as mock_scheduler_cls:
            on_fetch = MagicMock()
            client.start_scheduled(on_fetch)

            # Get the scheduled function
            job_args, _ = mock_scheduler_cls.return_value.add_job.call_args
            scheduled_func = job_args[0]

            # Mock fetch_with_retry and run the scheduled function
            mock_dataset = ValidationDataset(properties=mock_validation_data)
            with patch.object(
                client,
                "fetch_with_retry",
                new_callable=AsyncMock,
                return_value=(mock_dataset, None),
            ):
                await scheduled_func()

            on_fetch.assert_called_once_with(mock_dataset, None)

    async def test_scheduled_job_handles_errors(self, client):
        """Scheduled job catches and logs errors without crashing."""
        with patch(
            "real_estate.data.validation_dataset_client.AsyncIOScheduler"
        ) as mock_scheduler_cls:
            on_fetch = MagicMock()
            client.start_scheduled(on_fetch)

            # Get the scheduled function
            job_args, _ = mock_scheduler_cls.return_value.add_job.call_args
            scheduled_func = job_args[0]

            # Mock fetch_with_retry to raise an exception
            with patch.object(
                client,
                "fetch_with_retry",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ):
                # Should not raise, just log the error
                await scheduled_func()

            # on_fetch should be called with (None, None) to signal failure
            # This allows the validator to handle the failure (e.g., zero scores for burn)
            on_fetch.assert_called_once_with(None, None)

    async def test_scheduled_job_retries_exhausted_then_calls_callback(self, mock_keypair):
        """Callback only called after retries exhausted, not during retries."""
        config = ValidationDatasetClientConfig(
            url="https://api.example.com",
            max_retries=3,
            retry_delay_seconds=0,  # No delay for test speed
        )
        client = ValidationDatasetClient(config, mock_keypair)

        with patch(
            "real_estate.data.validation_dataset_client.AsyncIOScheduler"
        ) as mock_scheduler_cls:
            on_fetch = MagicMock()
            client.start_scheduled(on_fetch)

            # Get the scheduled function
            job_args, _ = mock_scheduler_cls.return_value.add_job.call_args
            scheduled_func = job_args[0]

            # Track callback calls during retries
            callback_calls_during_retries = []

            def track_download_attempts(*args, **kwargs):
                # Record callback call count at each retry attempt
                callback_calls_during_retries.append(on_fetch.call_count)
                raise ValidationDataRequestError("Server error")

            # Mock download_validation_set to always fail (triggers retries)
            with patch.object(
                client,
                "download_validation_set",
                new_callable=AsyncMock,
                side_effect=track_download_attempts,
            ) as mock_download:
                await scheduled_func()

                # Should have retried max_retries times
                assert mock_download.call_count == 3

            # Callback was NOT called during any retry attempt
            assert callback_calls_during_retries == [0, 0, 0]

            # Callback only called AFTER all retries exhausted
            on_fetch.assert_called_once_with(None, None)

