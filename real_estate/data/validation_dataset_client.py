"""Validation set client with hotkey-signed authentication."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bittensor import Keypair
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from .errors import (
    ValidationDataAuthError,
    ValidationDataNotFoundError,
    ValidationDataProcessingError,
    ValidationDataRateLimitError,
    ValidationDataRequestError,
)
from .models import ValidationDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationDatasetClientConfig:
    """Configuration for validation dataset client."""

    url: str
    endpoint: str = "/api/auth/validation-set"
    timeout: float = 60.0
    # URL expiration is 1 hour, no need to cache long-term
    max_retries: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    # Rate limit: 10 requests per minute
    schedule_hour: int = 2  # Default 2 AM UTC
    schedule_minute: int = 0
    download_raw: bool = False  # Whether to download raw files by default
    # Test mode: load from local file instead of API
    test_data_path: str = ""  # Path to local JSON file (bypasses API when set)


@dataclass
class RawFileInfo:
    """Info about a raw state data file."""

    filename: str
    presigned_url: str
    file_size: int


@dataclass
class ValidationDatasetResponse:
    """Response from validation dataset API."""

    validator_uid: int
    validation_date: str  # YYYY-MM-DD
    expires_at: str  # ISO timestamp
    validation_set_url: str
    validation_set_filename: str
    validation_set_size: int
    raw_files: list[RawFileInfo]


class ValidationDatasetClient:
    """
    Client for fetching validation data from dashboard API.

    Authentication uses hotkey signing:
    - Signs request with hotkey private key
    - API validates against registered validators
    - Returns presigned S3 URLs for data access
    """

    def __init__(self, config: ValidationDatasetClientConfig, keypair: Keypair):
        """
        Initialize validation dataset client.

        Args:
            config: Validation dataset client configuration
            keypair: Bittensor keypair for signing requests
        """
        self._config = config
        self._keypair = keypair
        self._base_url = config.url.rstrip("/")
        self._use_test_data = bool(config.test_data_path)

        if self._use_test_data:
            logger.info(f"TEST DATA: Will load data from {config.test_data_path}")

    def _sign_request(self, method: str, url: str, nonce: str) -> dict[str, str]:
        """
        Sign a request and return headers with authentication.

        Message format: {METHOD}{URL}{HEADERS_JSON}
        Headers JSON: {"Hotkey": "...", "Nonce": "..."}  (sorted keys)

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full request URL
            nonce: Timestamp nonce

        Returns:
            Headers dict with Hotkey, Nonce, and Signature
        """
        hotkey = self._keypair.ss58_address

        # Build headers for signing (sorted alphabetically)
        sign_headers = {
            "Hotkey": hotkey,
            "Nonce": nonce,
        }

        # IMPORTANT: Use separators=(',', ':') to match JavaScript's JSON.stringify (no spaces!)
        headers_str = json.dumps(sign_headers, sort_keys=True, separators=(",", ":"))
        data_to_sign = f"{method.upper()}{url}{headers_str}"

        signature = self._keypair.sign(data_to_sign.encode()).hex()

        return {
            **sign_headers,
            "Signature": signature,
        }

    def _load_test_data(self) -> ValidationDataset:
        """
        Load validation data from local JSON file (test mode).

        Expected JSON format:
        {
            "properties": [
                {"price": 500000, "living_area_sqft": 2000, ...},
                ...
            ]
        }
        OR just a list of properties directly.

        Returns:
            ValidationDataset loaded from file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValidationDataRequestError: If JSON format is invalid.
        """
        file_path = Path(self._config.test_data_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")

        logger.info(f"Loading test validation data from: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        # Handle multiple formats (same as download_validation_set)
        if isinstance(data, list):
            properties = data
        elif isinstance(data, dict) and "properties" in data:
            properties = data["properties"]
        elif isinstance(data, dict) and "records" in data:
            properties = data["records"]
        else:
            raise ValidationDataRequestError(
                "Invalid test data format: expected list or dict with 'properties'/'records' key"
            )

        if not properties:
            raise ValidationDataRequestError("Test data has empty properties list")

        dataset = ValidationDataset(properties=properties)
        logger.info(f"Loaded {len(dataset)} properties from test data")

        return dataset

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """
        Make authenticated request to validation API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response data

        Raises:
            ValidationDataAuthError: If authentication fails (401)
            ValidationDataNotFoundError: If data not found (404)
            ValidationDataProcessingError: If data still being processed (200 with status: processing)
            ValidationDataRateLimitError: If rate limited (429)
            ValidationDataRequestError: If request fails
        """
        # Build URL with query params if present
        url = f"{self._base_url}{endpoint}"
        if "params" in kwargs:
            # Construct URL with query params for signing
            params = kwargs["params"]
            if params:
                query_string = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{url}?{query_string}"

        nonce = str(time.time())
        auth_headers = self._sign_request(method, url, nonce)

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=auth_headers,
                )

                if response.status_code == 401:
                    raise ValidationDataAuthError(
                        f"Authentication failed: {response.text}"
                    )

                if response.status_code == 404:
                    raise ValidationDataNotFoundError(
                        f"Data not found: {response.text}"
                    )

                if response.status_code == 429:
                    raise ValidationDataRateLimitError(f"Rate limited: {response.text}")

                response.raise_for_status()

                try:
                    json_data = response.json()
                except ValueError as e:
                    raise ValidationDataRequestError(
                        f"Invalid JSON response: {e}"
                    ) from e

                # Check for "processing" status (200 response but not ready yet)
                if (
                    response.status_code == 200
                    and json_data.get("status") == "processing"
                ):
                    raise ValidationDataProcessingError(
                        json_data.get(
                            "message",
                            "Validation set is still being processed. Please check back later.",
                        ),
                        validation_date=json_data.get("validationDate"),
                        estimated_ready_time=json_data.get("estimatedReadyTime"),
                        retry_after=json_data.get("retryAfter"),
                    )

                return json_data

            except httpx.HTTPStatusError as e:
                raise ValidationDataRequestError(
                    f"Request failed: {e.response.status_code} - {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise ValidationDataRequestError(f"Connection error: {e}") from e

    async def get_validation_urls(
        self, date: str | None = None
    ) -> ValidationDatasetResponse:
        """
        Get presigned URLs for validation data.

        Args:
            date: Optional date in YYYY-MM-DD format (defaults to today)

        Returns:
            ValidationDatasetResponse with presigned URLs

        Raises:
            ValidationDataAuthError: Authentication failed (401)
            ValidationDataNotFoundError: No data for date (404)
            ValidationDataProcessingError: Data still being processed (200 with status: processing)
            ValidationDataRateLimitError: Rate limited (429)
            ValidationDataRequestError: Other errors
        """
        logger.info(f"Fetching validation URLs{f' for {date}' if date else ''}...")

        params = {"date": date} if date else {}
        data = await self._request("POST", self._config.endpoint, params=params)

        # Validate response structure
        if "validationSet" not in data:
            raise ValidationDataRequestError("Response missing 'validationSet' key")

        if "rawDataFiles" not in data:
            raise ValidationDataRequestError("Response missing 'rawDataFiles' key")

        # Parse response
        validation_set = data["validationSet"]
        raw_files = [
            RawFileInfo(
                filename=f["filename"],
                presigned_url=f["presignedUrl"],
                file_size=f["fileSize"],
            )
            for f in data["rawDataFiles"]
        ]

        return ValidationDatasetResponse(
            validator_uid=data["validatorUid"],
            validation_date=data["validationDate"],
            expires_at=data["expiresAt"],
            validation_set_url=validation_set["presignedUrl"],
            validation_set_filename=validation_set["filename"],
            validation_set_size=validation_set["fileSize"],
            raw_files=raw_files,
        )

    async def download_validation_set(
        self, date: str | None = None
    ) -> ValidationDataset:
        """
        Download and parse validation set.

        Args:
            date: Optional date in YYYY-MM-DD format

        Returns:
            ValidationDataset with properties

        Raises:
            ValidationDataAuthError: Authentication failed
            ValidationDataNotFoundError: No data for date
            ValidationDataRequestError: Download failed
        """
        logger.info("Downloading validation set...")

        response = await self.get_validation_urls(date)

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            try:
                download_response = await client.get(response.validation_set_url)
                download_response.raise_for_status()

                try:
                    data = download_response.json()

                    # Handle multiple formats:
                    # - Direct list of properties
                    # - Dict with 'properties' key
                    # - Dict with 'records' key (dashboard v1.0 format)
                    if isinstance(data, list):
                        properties = data
                    elif isinstance(data, dict) and "properties" in data:
                        properties = data["properties"]
                    elif isinstance(data, dict) and "records" in data:
                        properties = data["records"]
                    else:
                        raise ValidationDataRequestError(
                            f"Invalid validation set format: expected list or dict with 'properties'/'records' key, got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
                        )

                    logger.info(
                        f"Downloaded validation set: {len(properties)} properties"
                    )
                    return ValidationDataset(properties=properties)

                except ValueError as e:
                    raise ValidationDataRequestError(
                        f"Invalid JSON in validation set: {e}"
                    ) from e

            except httpx.RequestError as e:
                raise ValidationDataRequestError(
                    f"Failed to download validation set: {e}"
                ) from e

    async def download_raw_files(self, date: str | None = None) -> dict[str, dict]:
        """
        Download all raw state files.

        Args:
            date: Optional date in YYYY-MM-DD format

        Returns:
            Dictionary keyed by state code (e.g., {"AL": {...}, "AZ": {...}})

        Raises:
            ValidationDataAuthError: Authentication failed
            ValidationDataNotFoundError: No data for date
            ValidationDataRequestError: Download failed
        """
        logger.info("Downloading raw state files...")

        response = await self.get_validation_urls(date)

        raw_data = {}
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            for file_info in response.raw_files:
                # Extract state code from filename (e.g., "AL_20251229_140027.json" -> "AL")
                state = file_info.filename.split("_")[0]

                try:
                    download_response = await client.get(file_info.presigned_url)
                    download_response.raise_for_status()

                    try:
                        raw_data[state] = download_response.json()
                    except ValueError as e:
                        logger.warning(
                            f"Invalid JSON in raw file {file_info.filename}: {e}"
                        )
                        continue

                except httpx.RequestError as e:
                    logger.warning(
                        f"Failed to download raw file {file_info.filename}: {e}"
                    )
                    continue

        logger.info(f"Downloaded {len(raw_data)} raw state files")
        return raw_data

    async def download_raw_file(self, state: str, date: str | None = None) -> dict:
        """
        Download a specific raw state file.

        Primarily for testing and debugging. Production code uses
        download_raw_files() to fetch all states via fetch_with_retry().

        Args:
            state: State code (e.g., "AL", "AZ")
            date: Optional date in YYYY-MM-DD format

        Returns:
            Parsed JSON data for that state

        Raises:
            ValidationDataAuthError: Authentication failed
            ValidationDataNotFoundError: No data for date or state
            ValidationDataRequestError: Download failed
        """
        logger.info(f"Downloading raw file for state {state}...")

        response = await self.get_validation_urls(date)

        # Find the file for the requested state
        file_info = None
        for f in response.raw_files:
            if f.filename.startswith(f"{state}_"):
                file_info = f
                break

        if file_info is None:
            raise ValidationDataNotFoundError(
                f"No raw file found for state {state} on {response.validation_date}"
            )

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            try:
                download_response = await client.get(file_info.presigned_url)
                download_response.raise_for_status()

                try:
                    return download_response.json()
                except ValueError as e:
                    raise ValidationDataRequestError(
                        f"Invalid JSON in raw file: {e}"
                    ) from e

            except httpx.RequestError as e:
                raise ValidationDataRequestError(
                    f"Failed to download raw file: {e}"
                ) from e

    async def fetch_with_retry(
        self,
        date: str | None = None,
        download_validation: bool = True,
        download_raw: bool = False,
    ) -> tuple[ValidationDataset | None, dict[str, dict] | None]:
        """
        Fetch validation data with retry logic.

        Uses tenacity for retries with fixed delay between attempts.
        Retries on ValidationDataRequestError and ValidationDataRateLimitError.
        Does not retry on ValidationDataAuthError or ValidationDataNotFoundError.

        In test mode (test_data_path set), loads from local file instead.

        Args:
            date: Optional date in YYYY-MM-DD format
            download_validation: Whether to download validation set
            download_raw: Whether to download raw files

        Returns:
            Tuple of (ValidationDataset, raw_data) - None if not downloaded

        Raises:
            ValidationDataAuthError: If authentication fails (no retry)
            ValidationDataNotFoundError: If data not found (no retry)
            ValidationDataRequestError: If all retries exhausted
        """
        # Test data mode: load from local file instead of API
        if self._use_test_data:
            validation_data = self._load_test_data() if download_validation else None
            # Raw data not supported in test data mode
            return validation_data, None

        def _log_retry(retry_state):
            """Log retry attempts with clear context."""
            exc = retry_state.outcome.exception()
            wait_time = retry_state.next_action.sleep
            logger.warning(
                f"Validation data fetch failed: {exc}. "
                f"Retrying in {wait_time:.0f}s (attempt {retry_state.attempt_number}/{self._config.max_retries})"
            )

        async for attempt in AsyncRetrying(
            wait=wait_fixed(self._config.retry_delay_seconds),
            stop=stop_after_attempt(self._config.max_retries),
            retry=retry_if_exception_type(
                (
                    ValidationDataRequestError,
                    ValidationDataRateLimitError,
                    ValidationDataProcessingError,
                )
            ),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                validation_data = None
                raw_data = None

                if download_validation:
                    validation_data = await self.download_validation_set(date)

                if download_raw:
                    raw_data = await self.download_raw_files(date)

                return validation_data, raw_data

        # This should never be reached due to reraise=True
        return None, None

    def start_scheduled(
        self,
        on_fetch: Callable[
            [ValidationDataset | None, dict[str, dict] | None], None | Awaitable[None]
        ],
    ) -> AsyncIOScheduler:
        """
        Start scheduled validation data fetching using APScheduler.

        Fetches data daily at the configured time (default 6 PM UTC).

        In test mode: schedules an immediate fetch instead of waiting for cron time.

        Args:
            on_fetch: Callback called with (ValidationDataset, raw_data) after each successful fetch.
                      Can be sync or async. Either or both can be None depending on config.

        Returns:
            The scheduler instance (call scheduler.shutdown() to stop)
        """
        scheduler = AsyncIOScheduler(timezone="UTC")

        async def _scheduled_fetch():
            logger.info("Running scheduled validation set fetch...")
            try:
                validation_set, raw_data = await self.fetch_with_retry(
                    download_validation=True,
                    download_raw=self._config.download_raw,
                )

                # Call the callback
                result = on_fetch(validation_set, raw_data)
                if isinstance(result, Awaitable):
                    await result

                logger.info("Scheduled validation set fetch completed successfully")
            except Exception as e:
                logger.error(f"Scheduled validation set fetch failed: {e}")
                # Notify callback of failure so validator can handle it
                # (e.g., zero scores to trigger burn instead of rewarding stale winners)
                result = on_fetch(None, None)
                if isinstance(result, Awaitable):
                    await result

        # Schedule daily at configured time
        scheduler.add_job(
            _scheduled_fetch,
            "cron",
            hour=self._config.schedule_hour,
            minute=self._config.schedule_minute,
            id="validation_set_fetch",
        )
        logger.info(
            f"Scheduled validation set fetch: daily at {self._config.schedule_hour:02d}:{self._config.schedule_minute:02d} UTC"
        )
        if self._use_test_data:
            logger.info(f"  (using test data from: {self._config.test_data_path})")

        scheduler.start()
        return scheduler
