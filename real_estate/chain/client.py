"""Pylon-based chain client for validator operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pylon_client.v1 import (
    AsyncConfig,
    AsyncPylonClient,
    CommitmentDataHex,
    Hotkey,
    PylonForbidden,
    PylonRequestException,
    PylonResponseException,
    PylonUnauthorized,
    Weight,
)
from scalecodec.utils.ss58 import ss58_encode

from .errors import (
    AuthenticationError,
    ChainConnectionError,
    CommitmentError,
    WeightSettingError,
)
from .models import (
    ChainModelMetadata,
    Commitment,
    ExtrinsicCall,
    ExtrinsicData,
    Metagraph,
    Neuron,
)

if TYPE_CHECKING:
    from pylon_client.v1 import Neuron as PylonNeuron

logger = logging.getLogger(__name__)


def _hex_to_ss58(hex_address: str | None) -> str | None:
    """Convert hex address to SS58 format.

    Args:
        hex_address: Hex string like "0x3ca7ffe1..." or None

    Returns:
        SS58 address like "5DSEds..." or None
    """
    if hex_address is None:
        return None
    hex_str = hex_address[2:] if hex_address.startswith("0x") else hex_address
    return ss58_encode(bytes.fromhex(hex_str))


@dataclass(frozen=True)
class PylonConfig:
    """Configuration for Pylon client."""

    url: str  # e.g., "http://localhost:8000"
    token: str  # Authentication token
    identity: str  # Identity name configured in pylon
    timeout: float = 30.0  # Request timeout in seconds


class ChainClient:
    """
    Chain client using official Pylon SDK.

    Handles:
    - Fetching commitments (model metadata)
    - Setting weights
    - Fetching metagraph (neurons)
    - Extrinsic verification (when Pylon supports it)

    Uses AsyncPylonClient for type-safe, automatically retried requests.
    """

    def __init__(self, config: PylonConfig):
        """
        Initialize chain client.

        Args:
            config: Pylon connection configuration
        """
        self._config = config
        self._pylon_config = AsyncConfig(
            address=config.url,
            identity_name=config.identity,
            identity_token=config.token,
        )
        self._client: AsyncPylonClient | None = None

    async def __aenter__(self) -> ChainClient:
        """Async context manager entry."""
        self._client = AsyncPylonClient(self._pylon_config)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def _ensure_client(self) -> AsyncPylonClient:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "ChainClient must be used as async context manager: "
                "async with ChainClient(config) as client: ..."
            )
        return self._client

    async def get_all_commitments(self) -> list[ChainModelMetadata]:
        """
        Fetch all commitments from chain.

        Returns:
            List of ChainModelMetadata for all miners with commitments

        Note:
            block_number is set to 0 because Pylon doesn't yet include the actual
            commit block. We get the real block from get_extrinsic() during verification
            and update it via scheduler._update_commitment_block().
        """
        client = self._ensure_client()

        try:
            response = await client.identity.get_commitments()

            result = []
            for hotkey, hex_data in response.commitments.items():
                try:
                    # TODO(pylon): Set to actual commit block once Pylon includes it
                    # in get_commitments response. For now, we get it from get_extrinsic()
                    # during verification and update via scheduler._update_commitment_block().
                    commitment = Commitment(
                        hotkey=hotkey,
                        data=hex_data,
                        block=0,
                    )
                    metadata = commitment.to_metadata()
                    result.append(metadata)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse commitment for {hotkey}: {e}")
                    continue

            logger.debug(f"Fetched {len(result)} valid commitments")
            return result

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException as e:
            raise ChainConnectionError(f"Failed to fetch commitments: {e}") from e

    async def get_commitment(self, hotkey: str) -> Commitment | None:
        """
        Fetch commitment for a specific hotkey.

        Args:
            hotkey: Miner's hotkey address

        Returns:
            Commitment if found, None otherwise
        """
        client = self._ensure_client()

        try:
            response = await client.identity.get_commitment(Hotkey(hotkey))

            if response.commitment is None:
                logger.debug(f"No commitment found for hotkey {hotkey}")
                return None

            return Commitment(
                hotkey=response.hotkey,
                data=response.commitment,
                block=response.block.number,
            )

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException as e:
            # 404 means no commitment exists - check via __cause__ (Pylon doesn't expose status code)
            if e.__cause__ and "404" in str(e.__cause__):
                logger.debug(f"No commitment found for hotkey {hotkey}")
                return None
            raise ChainConnectionError(f"Failed to fetch commitment: {e}") from e

    async def get_model_metadata(self, hotkey: str) -> ChainModelMetadata | None:
        """
        Fetch and parse model metadata for a hotkey.

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
            CommitmentError: If commitment could not be set
            AuthenticationError: If Pylon credentials are invalid
            ChainConnectionError: If connection to Pylon fails
        """
        client = self._ensure_client()

        # Convert to hex string
        if isinstance(data, bytes):
            hex_data = "0x" + data.hex()
        elif not data.startswith("0x"):
            hex_data = "0x" + data
        else:
            hex_data = data

        try:
            await client.identity.set_commitment(CommitmentDataHex(hex_data))
            logger.info("Commitment set successfully")
            return True

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonForbidden as e:
            raise CommitmentError(f"Permission denied: {e}") from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException as e:
            raise CommitmentError(f"Failed to set commitment: {e}") from e

    async def get_metagraph(self) -> Metagraph:
        """
        Fetch current metagraph (all neurons).

        Returns:
            Metagraph with all neurons
        """
        client = self._ensure_client()

        try:
            response = await client.identity.get_latest_neurons()

            neurons = [
                self._convert_neuron(hotkey, neuron)
                for hotkey, neuron in response.neurons.items()
            ]

            metagraph = Metagraph(
                block=response.block.number,
                neurons=neurons,
                timestamp=datetime.now(UTC),
            )

            logger.debug(
                f"Fetched metagraph with {len(metagraph.neurons)} neurons "
                f"at block {metagraph.block}"
            )

            return metagraph

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException as e:
            raise ChainConnectionError(f"Failed to fetch metagraph: {e}") from e

    @staticmethod
    def _convert_neuron(hotkey: str, pylon_neuron: PylonNeuron) -> Neuron:
        """Convert Pylon Neuron to our Neuron model."""
        return Neuron(
            uid=pylon_neuron.uid,
            hotkey=hotkey,
            coldkey=pylon_neuron.coldkey,
            stake=float(pylon_neuron.stake),
            trust=float(pylon_neuron.trust),
            consensus=float(pylon_neuron.consensus),
            incentive=float(pylon_neuron.incentive),
            dividends=float(pylon_neuron.dividends),
            emission=float(pylon_neuron.emission),
            is_active=pylon_neuron.active,
            validator_permit=pylon_neuron.validator_permit,
        )

    async def get_all_miners(self) -> list[str]:
        """
        Get all registered miner hotkeys.

        Returns:
            List of hotkey addresses
        """
        metagraph = await self.get_metagraph()
        return metagraph.hotkeys

    async def set_weights(self, weights: dict[str, float]) -> None:
        """
        Set weights on chain.

        Args:
            weights: Dictionary mapping hotkey -> weight (should sum to 1.0)

        Raises:
            WeightSettingError: If weight setting fails
            AuthenticationError: If Pylon credentials are invalid
            ChainConnectionError: If connection to Pylon fails
        """
        if not weights:
            return

        client = self._ensure_client()

        # Convert to Pylon types
        pylon_weights: dict[Hotkey, Weight] = {
            Hotkey(k): Weight(v) for k, v in weights.items()
        }

        try:
            await client.identity.put_weights(pylon_weights)
            logger.info(f"Weights submitted to Pylon for {len(weights)} miners")

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonForbidden as e:
            raise WeightSettingError(
                f"Permission denied - validator may not be registered or have stake: {e}"
            ) from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException as e:
            raise WeightSettingError(f"Failed to set weights: {e}") from e

    async def get_extrinsic(
        self,
        block_number: int,
        extrinsic_index: int,
    ) -> ExtrinsicData | None:
        """
        Fetch extrinsic data from chain.

        Args:
            block_number: Block number containing the extrinsic
            extrinsic_index: Index of extrinsic within the block

        Returns:
            ExtrinsicData with address and call details, or None if not found
        """
        client = self._ensure_client()

        try:
            response = await client.identity.get_extrinsic(
                block_number, extrinsic_index
            )

            return ExtrinsicData(
                block_number=response.block_number,
                extrinsic_index=response.extrinsic_index,
                extrinsic_hash=response.extrinsic_hash,
                extrinsic_length=response.extrinsic_length,
                address=_hex_to_ss58(response.address),
                call=ExtrinsicCall(
                    call_module=response.call.call_module,
                    call_function=response.call.call_function,
                    call_args=response.call.call_args,
                ),
            )

        except PylonUnauthorized as e:
            raise AuthenticationError(f"Invalid Pylon credentials: {e}") from e
        except PylonRequestException as e:
            cause = str(e.__cause__) if e.__cause__ else str(e)
            raise ChainConnectionError(
                f"Connection error: {cause or 'Pylon unreachable'}"
            ) from e
        except PylonResponseException:
            # Could be 404 if extrinsic not found
            logger.debug(
                f"Extrinsic not found: block={block_number}, index={extrinsic_index}"
            )
            return None

    async def health_check(self) -> bool:
        """
        Check if Pylon is running and accessible.

        Returns:
            True if healthy
        """
        try:
            # Try to fetch neurons as a health check
            client = self._ensure_client()
            await client.identity.get_latest_neurons()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
