"""Scraper service client with hotkey-signed authentication."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bittensor import Keypair
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from .errors import ScraperAuthError, ScraperRequestError
from .models import ValidationDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScraperConfig:
    """Configuration for scraper client."""

    url: str
    realm: str = "devnet"  # devnet, testnet, mainnet
    validation_endpoint: str = "/api/v1/validation-data"
    timeout: float = 60.0
    # Schedule configuration (daily fetch)
    schedule_hour: int = 16  # 4 PM UTC
    schedule_minute: int = 0
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 600  # 10 minutes


def _create_retry_decorator(max_retries: int, delay_seconds: int):
    """Create a tenacity retry decorator with given config."""
    return retry(
        wait=wait_fixed(delay_seconds),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception_type(ScraperRequestError),
        reraise=True,
    )


class ScraperClient:
    """
    Client for fetching validation data from scraper service.

    Authentication uses hotkey signing:
    - Signs request with hotkey private key
    - Scraper validates against whitelist of validator hotkeys
    """

    def __init__(self, config: ScraperConfig, keypair: Keypair):
        """
        Initialize scraper client.

        Args:
            config: Scraper configuration
            keypair: Bittensor keypair for signing requests
        """
        self._config = config
        self._keypair = keypair
        self._base_url = config.url.rstrip("/")

        # Create retry decorator once at init
        self._retry_decorator = _create_retry_decorator(
            config.max_retries,
            config.retry_delay_seconds,
        )

    def _sign_request(self, method: str, url: str) -> dict[str, str]:
        """
        Sign a request and return headers with authentication.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full request URL

        Returns:
            Headers dict with Hotkey, Nonce, Realm, and Signature
        """
        nonce = str(time.time())
        hotkey = self._keypair.ss58_address

        # Build headers for signing (matching server expectation)
        sign_headers = {
            "Hotkey": hotkey,
            "Nonce": nonce,
            "Realm": self._config.realm,
        }

        headers_str = json.dumps(sign_headers, sort_keys=True)
        data_to_sign = f"{method.upper()}{url}{headers_str}"

        signature = self._keypair.sign(data_to_sign.encode()).hex()

        return {
            **sign_headers,
            "Signature": signature,
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """
        Make authenticated request to scraper service.

        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/api/v1/properties")
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response data

        Raises:
            ScraperAuthError: If authentication fails
            ScraperRequestError: If request fails
        """
        url = f"{self._base_url}{endpoint}"
        auth_headers = self._sign_request(method, url)

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=auth_headers,
                    **kwargs,
                )

                if response.status_code == 401:
                    raise ScraperAuthError(f"Authentication failed: {response.text}")

                if response.status_code == 403:
                    raise ScraperAuthError(f"Hotkey not authorized: {response.text}")

                response.raise_for_status()

                try:
                    return response.json()
                except ValueError as e:
                    raise ScraperRequestError(
                        f"Invalid JSON response from scraper: {e}"
                    ) from e

            except httpx.HTTPStatusError as e:
                raise ScraperRequestError(
                    f"Request failed: {e.response.status_code} - {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise ScraperRequestError(f"Connection error: {e}") from e

    async def get_validation_data(self) -> ValidationDataset:
        """
        Fetch validation dataset from scraper service.

        Returns:
            ValidationDataset with properties

        Raises:
            ScraperAuthError: If authentication fails
            ScraperRequestError: If request fails or response is invalid
        """
        logger.info("Fetching validation data from scraper...")

        data = await self._request("POST", self._config.validation_endpoint)

        if "properties" not in data:
            raise ScraperRequestError("Response missing 'properties' key")

        return ValidationDataset(properties=data["properties"])

    async def fetch_with_retry(self) -> ValidationDataset:
        """
        Fetch validation data with retry logic.

        Uses tenacity for retries with fixed delay between attempts.
        Only retries on ScraperRequestError (not auth errors).

        Returns:
            ValidationDataset on success

        Raises:
            ScraperAuthError: If authentication fails (no retry)
            ScraperRequestError: If all retries exhausted
        """
        return await self._retry_decorator(self.get_validation_data)()

    def start_scheduled(
        self,
        on_fetch: Callable[[ValidationDataset], None | Awaitable[None]],
    ) -> AsyncIOScheduler:
        """
        Start scheduled data fetching using APScheduler.

        Fetches data daily at the configured time (default 4 PM UTC).

        Args:
            on_fetch: Callback called with ValidationDataset after each successful fetch.
                      Can be sync or async.

        Returns:
            The scheduler instance (call scheduler.shutdown() to stop)
        """
        scheduler = AsyncIOScheduler(timezone="UTC")

        async def _scheduled_fetch():
            logger.info("Running scheduled validation data fetch...")
            try:
                data = await self.fetch_with_retry()
                result = on_fetch(data)
                if asyncio.iscoroutine(result):
                    await result
            except (ScraperAuthError, ScraperRequestError) as e:
                logger.error(f"Scheduled fetch failed: {e}")

        scheduler.add_job(
            _scheduled_fetch,
            CronTrigger(
                hour=self._config.schedule_hour,
                minute=self._config.schedule_minute,
                timezone="UTC",
            ),
            id="validation_data_fetch",
            name="Daily validation data fetch",
        )

        scheduler.start()
        logger.info(
            f"Scheduled fetcher started - daily at "
            f"{self._config.schedule_hour:02d}:{self._config.schedule_minute:02d} UTC"
        )

        return scheduler
