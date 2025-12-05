"""Pylon-based chain client for validator operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .errors import (
    AuthenticationError,
    ChainConnectionError,
    CommitmentError,
    WeightSettingError,
)
from .models import ChainModelMetadata, Commitment, Metagraph

logger = logging.getLogger(__name__)

# Retry decorator for GET requests: 3 attempts with exponential backoff + jitter
_retry_on_connection_error = retry(
    wait=wait_exponential_jitter(initial=0.1, jitter=0.2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ChainConnectionError),
    reraise=True,
)


@dataclass(frozen=True)
class PylonConfig:
    """Configuration for Pylon client."""

    url: str  # e.g., "http://localhost:8000"
    token: str  # Authentication token
    identity: str  # Identity name configured in pylon
    netuid: int  # Subnet UID
    timeout: float = 30.0  # Request timeout in seconds


class PylonClient:
    """
    Async HTTP client for Pylon chain interactions.

    Handles:
    - Fetching commitments (model metadata)
    - Setting weights
    - Fetching metagraph (neurons)

    All chain interactions go through Pylon's REST API.
    Each method creates its own connection - safe for long-running services.
    """

    def __init__(self, config: PylonConfig):
        """
        Initialize Pylon client.

        Args:
            config: Pylon connection configuration
        """
        self._config = config

    def _client(self) -> httpx.AsyncClient:
        """Create a new HTTP client for a request."""
        return httpx.AsyncClient(
            base_url=self._config.url,
            headers={
                "Authorization": f"Bearer {self._config.token}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout,
        )

    @_retry_on_connection_error
    async def get_all_commitments(self) -> dict[str, Commitment]:
        """
        Fetch all commitments from chain.

        Returns:
            Dictionary mapping hotkey -> Commitment

        Retries on transient connection errors (3 attempts with exponential backoff).
        """
        async with self._client() as client:
            try:
                url = f"/api/v1/subnet/{self._config.netuid}/block/latest/commitments"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                # Response format: {"block": {...}, "commitments": {"hotkey": "0xhexdata", ...}}
                block_number = data.get("block", {}).get("number", 0)
                commitments = {}
                for hotkey, hex_data in data.get("commitments", {}).items():
                    commitments[hotkey] = Commitment(
                        hotkey=hotkey,
                        data=hex_data,
                        block=block_number,
                    )

                logger.debug(f"Fetched {len(commitments)} commitments")
                return commitments

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise ChainConnectionError(f"Failed to fetch commitments: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    @_retry_on_connection_error
    async def get_commitment(self, hotkey: str) -> Commitment | None:
        """
        Fetch commitment for a specific hotkey.

        Args:
            hotkey: Miner's hotkey address

        Returns:
            Commitment if found, None otherwise

        Retries on transient connection errors (3 attempts with exponential backoff).
        """
        async with self._client() as client:
            try:
                url = f"/api/v1/subnet/{self._config.netuid}/block/latest/commitments/{hotkey}"
                response = await client.get(url)

                if response.status_code == 404:
                    logger.debug(f"No commitment found for hotkey {hotkey}")
                    return None

                response.raise_for_status()
                data = response.json()

                # Response format: {"hotkey": "...", "commitment": "0xhexdata"}
                hex_data = data.get("commitment")
                if hex_data is None:
                    return None

                return Commitment(
                    hotkey=data.get("hotkey", hotkey),
                    data=hex_data,
                    block=0,
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise ChainConnectionError(f"Failed to fetch commitment: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    async def get_model_metadata(self, hotkey: str) -> ChainModelMetadata | None:
        """
        Fetch and parse model metadata for a hotkey.

        Convenience method that fetches commitment and parses it.

        Args:
            hotkey: Miner's hotkey address

        Returns:
            Parsed model metadata if found, None otherwise
        """
        commitment = await self.get_commitment(hotkey)
        if commitment is None:
            return None

        try:
            return commitment.to_metadata()
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse commitment for {hotkey}: {e}")
            raise CommitmentError(f"Invalid commitment data for {hotkey}: {e}") from e

    async def set_commitment(self, data: str | bytes) -> bool:
        """
        Set commitment on chain.

        Args:
            data: Commitment data as hex string (with or without 0x prefix) or bytes

        Returns:
            True if successful

        Raises:
            CommitmentError: If commitment could not be set (including after pylon retries)
            AuthenticationError: If Pylon token is invalid
            ChainConnectionError: If connection to Pylon fails
        """
        # Convert bytes to hex string if needed
        if isinstance(data, bytes):
            data = "0x" + data.hex()
        elif not data.startswith("0x"):
            data = "0x" + data

        async with self._client() as client:
            try:
                url = f"/api/v1/identity/{self._config.identity}/subnet/{self._config.netuid}/commitments"
                response = await client.post(
                    url,
                    json={"commitment": data},
                )

                if response.status_code == 201:
                    logger.info("Commitment set successfully")
                    return True

                response.raise_for_status()
                return True

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                if e.response.status_code == 502:
                    # BadGatewayException from pylon - commitment failed after retries
                    detail = e.response.json().get("detail", "Unknown error")
                    raise CommitmentError(
                        f"Failed to set commitment on chain: {detail}"
                    ) from e
                raise CommitmentError(f"Failed to set commitment: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    @_retry_on_connection_error
    async def get_metagraph(self) -> Metagraph:
        """
        Fetch current metagraph (all neurons).

        Returns:
            Metagraph with all neurons

        Retries on transient connection errors (3 attempts with exponential backoff).
        """
        async with self._client() as client:
            try:
                url = f"/api/v1/subnet/{self._config.netuid}/neurons/latest"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                metagraph = Metagraph.from_pylon_response(data)
                logger.debug(
                    f"Fetched metagraph with {len(metagraph.neurons)} neurons "
                    f"at block {metagraph.block}"
                )

                return metagraph

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise ChainConnectionError(f"Failed to fetch metagraph: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    async def get_all_miners(self) -> list[str]:
        """
        Get all registered miner hotkeys.

        Returns:
            List of hotkey addresses
        """
        metagraph = await self.get_metagraph()
        return metagraph.hotkeys

    async def set_weights(
        self,
        weights: dict[str, float],
    ) -> None:
        """
        Set weights on chain.

        Args:
            weights: Dictionary mapping hotkey -> weight (should sum to 1.0)

        Raises:
            WeightSettingError: If weight setting fails.
            AuthenticationError: If Pylon token is invalid.
            ChainConnectionError: If connection to Pylon fails.

        Note:
            Weights are set by the identity configured in Pylon.
            The validator hotkey must be registered and have permission.
        """
        if not weights:
            return

        async with self._client() as client:
            try:
                url = f"/api/v1/identity/{self._config.identity}/subnet/{self._config.netuid}/weights"
                response = await client.put(
                    url,
                    json={"weights": weights},
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                if e.response.status_code == 403:
                    raise WeightSettingError(
                        "Permission denied - validator may not be registered or have stake"
                    ) from e
                raise WeightSettingError(f"Failed to set weights: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    @_retry_on_connection_error
    async def health_check(self) -> bool:
        """
        Check if Pylon is running and accessible.

        Returns:
            True if healthy, False otherwise

        Retries on transient connection errors (3 attempts with exponential backoff).
        """
        async with self._client() as client:
            try:
                response = await client.get("/schema/swagger")
                return response.status_code == 200
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Health check failed: {e}") from e
