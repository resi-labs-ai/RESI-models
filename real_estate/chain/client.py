"""Pylon-based chain client for validator operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .errors import (
    AuthenticationError,
    ChainConnectionError,
    CommitmentError,
    MetagraphError,
    WeightSettingError,
)
from .models import ChainModelMetadata, Commitment, Metagraph

logger = logging.getLogger(__name__)


@dataclass
class PylonConfig:
    """Configuration for Pylon client."""

    url: str  # e.g., "http://localhost:8000"
    token: str  # Authentication token
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

    async def get_all_commitments(self) -> dict[str, Commitment]:
        """
        Fetch all commitments from chain.

        Returns:
            Dictionary mapping hotkey -> Commitment
        """
        async with self._client() as client:
            try:
                response = await client.get("/api/v1/commitments")
                response.raise_for_status()
                data = response.json()

                commitments = {}
                for hotkey, hex_data in data.get("commitments", {}).items():
                    commitments[hotkey] = Commitment(
                        hotkey=hotkey,
                        data=hex_data,
                        block=0,  # Block not returned in bulk fetch
                    )

                logger.debug(f"Fetched {len(commitments)} commitments")
                return commitments

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise ChainConnectionError(f"Failed to fetch commitments: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    async def get_commitment(self, hotkey: str) -> Commitment | None:
        """
        Fetch commitment for a specific hotkey.

        Args:
            hotkey: Miner's hotkey address

        Returns:
            Commitment if found, None otherwise
        """
        async with self._client() as client:
            try:
                response = await client.get(f"/api/v1/commitments/{hotkey}")

                if response.status_code == 404:
                    logger.debug(f"No commitment found for hotkey {hotkey}")
                    return None

                response.raise_for_status()
                data = response.json()

                return Commitment(
                    hotkey=data.get("hotkey", hotkey),
                    data=data.get("data", ""),
                    block=data.get("block", 0),
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise CommitmentError(f"Failed to fetch commitment: {e}") from e
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

    async def set_commitment(self, data: str) -> bool:
        """
        Set commitment on chain.

        Args:
            data: Hex-encoded commitment data (must be <= 128 bytes)

        Returns:
            True if successful
        """
        async with self._client() as client:
            try:
                response = await client.post(
                    "/api/v1/commitments",
                    json={"data": data},
                )

                if response.status_code == 201:
                    logger.info("Commitment set successfully")
                    return True

                response.raise_for_status()
                return True

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise CommitmentError(f"Failed to set commitment: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    async def get_metagraph(self, netuid: int) -> Metagraph:
        """
        Fetch current metagraph (all neurons).

        Args:
            netuid: Subnet UID

        Returns:
            Metagraph with all neurons
        """
        async with self._client() as client:
            try:
                response = await client.get("/api/v1/neurons/latest")
                response.raise_for_status()
                data = response.json()

                metagraph = Metagraph.from_pylon_response(netuid, data)
                logger.debug(
                    f"Fetched metagraph with {len(metagraph.neurons)} neurons "
                    f"at block {metagraph.block}"
                )

                return metagraph

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid Pylon token") from e
                raise MetagraphError(f"Failed to fetch metagraph: {e}") from e
            except httpx.RequestError as e:
                raise ChainConnectionError(f"Connection error: {e}") from e

    async def get_all_miners(self, netuid: int) -> list[str]:
        """
        Get all registered miner hotkeys.

        Args:
            netuid: Subnet UID

        Returns:
            List of hotkey addresses
        """
        metagraph = await self.get_metagraph(netuid)
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
                response = await client.put(
                    "/api/v1/subnet/weights",
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

    async def health_check(self) -> bool:
        """
        Check if Pylon is running and accessible.

        Returns:
            True if healthy, False otherwise
        """
        async with self._client() as client:
            try:
                response = await client.get("/schema/swagger")
                return response.status_code == 200
            except Exception:
                return False
