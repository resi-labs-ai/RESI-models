"""Unit tests for ModelVerifier."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from real_estate.models import (
    ExtrinsicVerificationError,
    HashMismatchError,
    LicenseError,
    ModelTooLargeError,
    ModelVerifier,
)


class TestCheckLicense:
    """Tests for ModelVerifier.check_license method."""

    @pytest.mark.asyncio
    async def test_passes_when_license_matches(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Passes when LICENSE file content matches required license."""
        verifier = ModelVerifier(mock_chain_client, required_license="Test License")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Test License"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_license_missing(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError when LICENSE file is not found."""
        verifier = ModelVerifier(mock_chain_client, required_license="Test License")

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(LicenseError, match="LICENSE file not found"):
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_license_differs(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError when LICENSE content doesn't match."""
        verifier = ModelVerifier(mock_chain_client, required_license="Expected License")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Different License Content"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(LicenseError, match="Invalid license"):
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_on_http_failure(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError on HTTP failure."""
        verifier = ModelVerifier(mock_chain_client, required_license="Test License")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            with pytest.raises(LicenseError, match="Failed to fetch LICENSE"):
                await verifier.check_license("user/repo")


class TestCheckSize:
    """Tests for ModelVerifier.check_size method."""

    @pytest.mark.asyncio
    async def test_returns_size_when_under_limit(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Returns size when model is under limit."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "model.onnx", "size": 50_000_000},
            {"path": "README.md", "size": 1000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            size = await verifier.check_size("user/repo", max_size_bytes=100_000_000)

        assert size == 50_000_000

    @pytest.mark.asyncio
    async def test_raises_error_when_over_limit(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ModelTooLargeError when model exceeds limit."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "model.onnx", "size": 500_000_000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ModelTooLargeError, match="exceeds limit"):
                await verifier.check_size("user/repo", max_size_bytes=100_000_000)

    @pytest.mark.asyncio
    async def test_returns_zero_when_model_not_found(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Returns 0 when model.onnx not in file tree."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "README.md", "size": 1000},
            {"path": "other_file.txt", "size": 500},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            size = await verifier.check_size("user/repo", max_size_bytes=100_000_000)

        assert size == 0

    @pytest.mark.asyncio
    async def test_returns_zero_on_http_error(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Returns 0 on HTTP error (graceful degradation)."""
        verifier = ModelVerifier(mock_chain_client)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPError("API error")
            )
            size = await verifier.check_size("user/repo", max_size_bytes=100_000_000)

        assert size == 0


class TestVerifyExtrinsicRecord:
    """Tests for ModelVerifier.verify_extrinsic_record method."""

    @pytest.mark.asyncio
    async def test_passes_when_all_checks_pass(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Passes when extrinsic record is valid."""
        verifier = ModelVerifier(mock_chain_client)

        # Mock HTTP response for extrinsic_record.json
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "extrinsic": "12345-0",
            "hotkey": "5TestHotkey",
        }
        mock_response.raise_for_status = MagicMock()

        # Mock chain client response
        mock_extrinsic = MagicMock()
        mock_extrinsic.address = "5TestHotkey"
        mock_extrinsic.is_commitment_extrinsic.return_value = True
        mock_extrinsic.call.call_args = []  # No call args to parse
        mock_chain_client.get_extrinsic = AsyncMock(return_value=mock_extrinsic)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await verifier.verify_extrinsic_record(
                hotkey="5TestHotkey",
                hf_repo_id="user/repo",
                expected_hash="abc12345",
            )

    @pytest.mark.asyncio
    async def test_raises_error_when_record_not_found(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ExtrinsicVerificationError when record file is missing."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(
                ExtrinsicVerificationError, match="extrinsic_record.json not found"
            ):
                await verifier.verify_extrinsic_record(
                    hotkey="5TestHotkey",
                    hf_repo_id="user/repo",
                    expected_hash="abc12345",
                )

    @pytest.mark.asyncio
    async def test_raises_error_when_hotkey_mismatch(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ExtrinsicVerificationError when hotkey doesn't match."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "extrinsic": "12345-0",
            "hotkey": "5DifferentHotkey",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ExtrinsicVerificationError, match="Hotkey mismatch"):
                await verifier.verify_extrinsic_record(
                    hotkey="5ExpectedHotkey",
                    hf_repo_id="user/repo",
                    expected_hash="abc12345",
                )

    @pytest.mark.asyncio
    async def test_raises_error_when_extrinsic_not_on_chain(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ExtrinsicVerificationError when extrinsic not found on chain."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "extrinsic": "12345-0",
            "hotkey": "5TestHotkey",
        }
        mock_response.raise_for_status = MagicMock()

        # Chain client returns None (extrinsic not found)
        mock_chain_client.get_extrinsic = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ExtrinsicVerificationError, match="not found on chain"):
                await verifier.verify_extrinsic_record(
                    hotkey="5TestHotkey",
                    hf_repo_id="user/repo",
                    expected_hash="abc12345",
                )

    @pytest.mark.asyncio
    async def test_raises_error_when_signer_mismatch(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ExtrinsicVerificationError when extrinsic signer != hotkey."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "extrinsic": "12345-0",
            "hotkey": "5TestHotkey",
        }
        mock_response.raise_for_status = MagicMock()

        # Extrinsic signed by different address
        mock_extrinsic = MagicMock()
        mock_extrinsic.address = "5WrongSigner"
        mock_chain_client.get_extrinsic = AsyncMock(return_value=mock_extrinsic)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ExtrinsicVerificationError, match="signer"):
                await verifier.verify_extrinsic_record(
                    hotkey="5TestHotkey",
                    hf_repo_id="user/repo",
                    expected_hash="abc12345",
                )

    @pytest.mark.asyncio
    async def test_raises_error_when_not_commitment_extrinsic(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ExtrinsicVerificationError when extrinsic is not a commitment call."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "extrinsic": "12345-0",
            "hotkey": "5TestHotkey",
        }
        mock_response.raise_for_status = MagicMock()

        # Extrinsic is valid but not a commitment call
        mock_extrinsic = MagicMock()
        mock_extrinsic.address = "5TestHotkey"
        mock_extrinsic.is_commitment_extrinsic.return_value = False  # Not a commitment
        mock_chain_client.get_extrinsic = AsyncMock(return_value=mock_extrinsic)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(
                ExtrinsicVerificationError, match="not a commitment call"
            ):
                await verifier.verify_extrinsic_record(
                    hotkey="5TestHotkey",
                    hf_repo_id="user/repo",
                    expected_hash="abc12345",
                )


class TestVerifyHash:
    """Tests for ModelVerifier.verify_hash method."""

    def test_passes_when_hash_matches(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Passes when computed hash matches expected."""
        verifier = ModelVerifier(mock_chain_client)

        # Create a file and compute its hash
        test_file = tmp_path / "test.onnx"
        test_file.write_bytes(b"test model content")

        # Compute expected hash
        expected_hash = verifier.compute_hash(test_file)

        # Should not raise
        verifier.verify_hash(test_file, expected_hash)

    def test_raises_error_when_hash_mismatch(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Raises HashMismatchError when hash doesn't match."""
        verifier = ModelVerifier(mock_chain_client)

        test_file = tmp_path / "test.onnx"
        test_file.write_bytes(b"test model content")

        with pytest.raises(HashMismatchError, match="Hash mismatch"):
            verifier.verify_hash(test_file, "wronghash")


class TestComputeHash:
    """Tests for ModelVerifier.compute_hash method."""

    def test_computes_correct_hash(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Computes consistent hash for same content."""
        verifier = ModelVerifier(mock_chain_client)

        test_file = tmp_path / "test.onnx"
        test_file.write_bytes(b"known content")

        hash1 = verifier.compute_hash(test_file)
        hash2 = verifier.compute_hash(test_file)

        assert hash1 == hash2

    def test_returns_64_character_sha256_hex(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Returns 64-character hexadecimal SHA-256 hash."""
        verifier = ModelVerifier(mock_chain_client)

        test_file = tmp_path / "test.onnx"
        test_file.write_bytes(b"test content")

        result = verifier.compute_hash(test_file)

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_content_produces_different_hash(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Different file content produces different hash."""
        verifier = ModelVerifier(mock_chain_client)

        file1 = tmp_path / "file1.onnx"
        file1.write_bytes(b"content A")

        file2 = tmp_path / "file2.onnx"
        file2.write_bytes(b"content B")

        hash1 = verifier.compute_hash(file1)
        hash2 = verifier.compute_hash(file2)

        assert hash1 != hash2


class TestExtractHashFromCallArgs:
    """Tests for _extract_hash_from_call_args static method."""

    def test_extracts_hash_from_valid_call_args(self) -> None:
        """Extracts hash from properly formatted call_args."""
        import json

        # Simulate the commitment data structure
        commitment_data = {"h": "abc12345", "r": "user/repo", "v": "1.0"}
        hex_encoded = "0x" + json.dumps(commitment_data).encode().hex()

        call_args = [
            {
                "name": "info",
                "value": {"fields": [{"Raw65": hex_encoded}]},
            }
        ]

        result = ModelVerifier._extract_hash_from_call_args(call_args)
        assert result == "abc12345"

    def test_returns_none_when_no_info_field(self) -> None:
        """Returns None when 'info' field is missing."""
        call_args = [{"name": "other", "value": {}}]

        result = ModelVerifier._extract_hash_from_call_args(call_args)
        assert result is None

    def test_returns_none_on_invalid_hex(self) -> None:
        """Returns None when hex data is invalid."""
        call_args = [
            {
                "name": "info",
                "value": {"fields": [{"Raw65": "not valid hex"}]},
            }
        ]

        result = ModelVerifier._extract_hash_from_call_args(call_args)
        assert result is None
