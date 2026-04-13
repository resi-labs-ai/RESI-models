"""Model verification (extrinsic records, hash, license, feature config)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from ..data.config_encoder import FeatureConfig, parse_feature_config
from ..data.errors import FeatureConfigError
from .errors import (
    ExtrinsicVerificationError,
    FeatureConfigValidationError,
    HashMismatchError,
    LicenseError,
    ModelDownloadError,
    ModelTooLargeError,
)
from .models import ExtrinsicRecord

if TYPE_CHECKING:
    from ..chain.client import ChainClient

logger = logging.getLogger(__name__)

HF_RAW_URL = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
HF_API_URL = "https://huggingface.co/api/models/{repo_id}"


class ModelVerifier:
    """
    Verify model authenticity and integrity.

    Verification steps:
    1. Pre-download: Check license file in HF repo
    2. Pre-download: Check model size via HF API
    3. Pre-download: Verify extrinsic_record.json against chain
    4. Post-download: Verify file hash matches commitment
    """

    def __init__(
        self,
        chain_client: ChainClient,
        http_timeout: float = 30.0,
        hf_token: str | None = None,
    ):
        """
        Initialize verifier.

        Args:
            chain_client: Client for chain queries
            http_timeout: Timeout for HF API requests
            hf_token: Optional HuggingFace API token for authenticated requests
        """
        self._chain = chain_client
        self._http_timeout = http_timeout
        self._hf_token = hf_token

    @property
    def _http_headers(self) -> dict[str, str]:
        """HTTP headers for HuggingFace API requests."""
        if self._hf_token:
            return {"Authorization": f"Bearer {self._hf_token}"}
        return {}

    async def check_license(self, hf_repo_id: str) -> None:
        """
        Check HuggingFace repo has MIT license in metadata.

        Uses HF API to check model card metadata for MIT license.
        This is a case-insensitive check for "mit" in the license field.

        Args:
            hf_repo_id: HuggingFace repository ID (user/repo)

        Raises:
            LicenseError: If license missing, not MIT, or API error
        """
        url = HF_API_URL.format(repo_id=hf_repo_id)

        async with httpx.AsyncClient(
            timeout=self._http_timeout,
            follow_redirects=True,
            headers=self._http_headers,
        ) as client:
            try:
                response = await client.get(url)

                if response.status_code == 404:
                    raise LicenseError(f"Repository {hf_repo_id} not found")

                response.raise_for_status()
                model_info = response.json()

                # Check license in model card metadata
                card_data = model_info.get("cardData") or {}
                license_value = card_data.get("license") or ""

                if "mit" in license_value.lower():
                    logger.debug(f"MIT license verified for {hf_repo_id}")
                    return

                raise LicenseError(
                    f"MIT license required for {hf_repo_id}, found: {license_value or 'none'}"
                )

            except httpx.HTTPError as e:
                raise LicenseError(
                    f"Failed to check license for {hf_repo_id}: {e}"
                ) from e

    async def _fetch_repo_tree(self, hf_repo_id: str) -> list[dict]:
        """Fetch the file listing from a HuggingFace repository.

        Args:
            hf_repo_id: HuggingFace repository ID (user/repo)

        Returns:
            List of file metadata dicts from the HF API.

        Raises:
            ModelDownloadError: If API call fails.
        """
        api_url = f"https://huggingface.co/api/models/{hf_repo_id}/tree/main"

        async with httpx.AsyncClient(
            timeout=self._http_timeout,
            follow_redirects=True,
            headers=self._http_headers,
        ) as client:
            try:
                response = await client.get(api_url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise ModelDownloadError(
                    f"Failed to fetch file list from {hf_repo_id}: {e}"
                ) from e

    async def find_repo_files(
        self, hf_repo_id: str, feature_config_max_size: int = 50 * 1024
    ) -> tuple[str, int, tuple[str, int] | None]:
        """
        Find ONNX model and optional feature_config.json in one API call.

        Args:
            hf_repo_id: HuggingFace repository ID (user/repo)
            feature_config_max_size: Maximum allowed size for feature_config.json.

        Returns:
            Tuple of (onnx_filename, onnx_size, feature_config_info_or_none).
            feature_config_info is (filename, size) or None if not found/too large.

        Raises:
            ModelDownloadError: If no .onnx file found or multiple found.
        """
        files = await self._fetch_repo_tree(hf_repo_id)

        # Find .onnx files in root only (not subdirectories)
        onnx_files = [
            f
            for f in files
            if f.get("path", "").endswith(".onnx") and "/" not in f.get("path", "")
        ]

        if not onnx_files:
            raise ModelDownloadError(
                f"No .onnx file found in {hf_repo_id} repository root"
            )

        if len(onnx_files) > 1:
            names = [f.get("path") for f in onnx_files]
            raise ModelDownloadError(
                f"Multiple .onnx files found in {hf_repo_id}: {names}. "
                f"Repository must contain exactly one .onnx file in root."
            )

        onnx_file = onnx_files[0]
        onnx_filename = onnx_file.get("path", "")
        onnx_size = onnx_file.get("size", 0)
        logger.debug(f"Found {onnx_filename}: {onnx_size} bytes")

        # Find feature_config.json (optional)
        feature_config_info = None
        config_files = [f for f in files if f.get("path") == "feature_config.json"]
        if config_files:
            config_size = config_files[0].get("size", 0)
            if config_size <= feature_config_max_size:
                feature_config_info = ("feature_config.json", config_size)
                logger.debug(f"Found feature_config.json: {config_size} bytes")
            else:
                logger.warning(
                    f"feature_config.json too large: {config_size} bytes "
                    f"(max {feature_config_max_size} bytes), ignoring"
                )

        return onnx_filename, onnx_size, feature_config_info

    async def find_onnx_file(self, hf_repo_id: str) -> tuple[str, int]:
        """
        Find the ONNX model file in a HuggingFace repository.

        Args:
            hf_repo_id: HuggingFace repository ID (user/repo)

        Returns:
            Tuple of (filename, size in bytes)

        Raises:
            ModelDownloadError: If no .onnx file found, multiple found, or API error
        """
        onnx_filename, onnx_size, _ = await self.find_repo_files(hf_repo_id)
        return onnx_filename, onnx_size

    async def find_feature_config(
        self, hf_repo_id: str, max_size_bytes: int = 50 * 1024
    ) -> tuple[str, int]:
        """
        Find feature_config.json in a HuggingFace repository.

        Args:
            hf_repo_id: HuggingFace repository ID (user/repo)
            max_size_bytes: Maximum allowed size (default 50KB)

        Returns:
            Tuple of (filename, size in bytes)

        Raises:
            ModelDownloadError: If not found, too large, or API error
        """
        _, _, config_info = await self.find_repo_files(
            hf_repo_id, feature_config_max_size=max_size_bytes
        )
        if config_info is None:
            raise ModelDownloadError(
                f"feature_config.json not found or too large in {hf_repo_id}"
            )
        return config_info

    def validate_feature_config(self, config_data: dict) -> FeatureConfig:
        """
        Validate raw JSON dict from feature_config.json.

        Args:
            config_data: Parsed JSON dict.

        Returns:
            Validated FeatureConfig.

        Raises:
            FeatureConfigValidationError: On any validation failure.
        """
        try:
            return parse_feature_config(config_data)
        except FeatureConfigError as e:
            raise FeatureConfigValidationError(
                f"Invalid feature_config.json: {e}"
            ) from e

    def check_model_size(self, size: int, max_size_bytes: int, filename: str) -> None:
        """
        Validate model size against limit.

        Args:
            size: Model size in bytes
            max_size_bytes: Maximum allowed size
            filename: Model filename (for error message)

        Raises:
            ModelTooLargeError: If size exceeds limit
        """
        if size > max_size_bytes:
            raise ModelTooLargeError(
                f"Model {filename} size {size} bytes exceeds limit {max_size_bytes} bytes"
            )

    async def verify_extrinsic_record(
        self,
        hotkey: str,
        hf_repo_id: str,
        expected_hash: str,
    ) -> int:
        """
        Verify extrinsic_record.json before download.

        Steps:
        1. Download extrinsic_record.json from HF repo
        2. Parse extrinsic ID (block-index format)
        3. Query chain for extrinsic
        4. Verify signer matches hotkey
        5. Verify commitment hash matches

        Args:
            hotkey: Expected miner hotkey
            hf_repo_id: HuggingFace repository ID
            expected_hash: Hash from chain commitment

        Returns:
            Commit block number (from Pylon chain query, not miner-provided)

        Raises:
            ExtrinsicVerificationError: If any check fails
        """
        # 1. Fetch extrinsic_record.json
        url = HF_RAW_URL.format(repo_id=hf_repo_id, filename="extrinsic_record.json")

        async with httpx.AsyncClient(
            timeout=self._http_timeout,
            follow_redirects=True,
            headers=self._http_headers,
        ) as client:
            try:
                response = await client.get(url)

                if response.status_code == 404:
                    raise ExtrinsicVerificationError(
                        f"extrinsic_record.json not found in {hf_repo_id}"
                    )

                response.raise_for_status()
                record_data = response.json()

            except httpx.HTTPError as e:
                raise ExtrinsicVerificationError(
                    f"Failed to fetch extrinsic_record.json from {hf_repo_id}: {e}"
                ) from e
            except json.JSONDecodeError as e:
                raise ExtrinsicVerificationError(
                    f"Invalid JSON in extrinsic_record.json: {e}"
                ) from e

        # 2. Parse record
        try:
            record = ExtrinsicRecord.from_dict(record_data)
        except KeyError as e:
            raise ExtrinsicVerificationError(
                f"Missing required field in extrinsic_record.json: {e}"
            ) from e

        # 3. Verify hotkey in record matches expected
        if record.hotkey != hotkey:
            raise ExtrinsicVerificationError(
                f"Hotkey mismatch: record has {record.hotkey}, expected {hotkey}"
            )

        # 4. Verify extrinsic on chain
        try:
            extrinsic_data = await self._chain.get_extrinsic(
                block_number=record.block_number,
                extrinsic_index=record.extrinsic_index,
            )
        except Exception as e:
            raise ExtrinsicVerificationError(
                f"Failed to fetch extrinsic {record.extrinsic} from chain: {e}"
            ) from e

        if extrinsic_data is None:
            raise ExtrinsicVerificationError(
                f"Extrinsic {record.extrinsic} not found on chain"
            )

        # 5. Verify signer matches hotkey
        if extrinsic_data.address != hotkey:
            raise ExtrinsicVerificationError(
                f"Extrinsic signer {extrinsic_data.address} != expected hotkey {hotkey}"
            )

        # 6. Verify it's a commitment extrinsic
        if not extrinsic_data.is_commitment_extrinsic():
            raise ExtrinsicVerificationError(
                f"Extrinsic {record.extrinsic} is not a commitment call"
            )

        # 7. Extract and verify commitment hash from call_args
        chain_hash = self._extract_hash_from_call_args(extrinsic_data.call.call_args)
        if chain_hash and chain_hash != expected_hash:
            raise ExtrinsicVerificationError(
                f"Chain hash {chain_hash} != expected {expected_hash}"
            )

        logger.debug(f"Extrinsic record verified for {hotkey}")

        # Return commit block from Pylon (trusted source), not from miner's record
        return extrinsic_data.block_number

    @staticmethod
    def _extract_hash_from_call_args(call_args: list[dict]) -> str | None:
        """
        Extract model hash from commitment extrinsic call_args.

        Structure: [{'name': 'info', 'value': {'fields': [{'Raw65': '0x...'}]}}]
        The hex decodes to JSON: {"h": "abc12345", "r": "user/repo", ...}
        """
        for arg in call_args:
            if arg.get("name") == "info":
                fields = arg.get("value", {}).get("fields", [])
                if fields:
                    # Get first field (Raw65 contains hex data)
                    field = fields[0]
                    hex_data = None
                    for key, value in field.items():
                        if key.startswith("Raw"):
                            hex_data = value
                            break
                    if hex_data:
                        try:
                            hex_str = (
                                hex_data[2:] if hex_data.startswith("0x") else hex_data
                            )
                            decoded = bytes.fromhex(hex_str).decode("utf-8")
                            data = json.loads(decoded)
                            return data.get("h")
                        except (ValueError, json.JSONDecodeError) as e:
                            logger.warning(f"Failed to decode commitment data: {e}")
        return None

    def verify_hash(self, file_path: Path, expected_hash: str) -> None:
        """
        Verify downloaded file hash.

        Args:
            file_path: Path to model file
            expected_hash: Expected hash from commitment

        Raises:
            HashMismatchError: If hash doesn't match
        """
        computed_hash = self.compute_hash(file_path)

        if computed_hash != expected_hash:
            raise HashMismatchError(
                f"Hash mismatch for {file_path.name}: "
                f"computed {computed_hash}, expected {expected_hash}"
            )

        logger.debug(f"Hash verified for {file_path}")

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """
        Compute SHA-256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            64-character hex hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
