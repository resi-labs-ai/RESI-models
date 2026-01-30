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
    ModelDownloadError,
    ModelTooLargeError,
    ModelVerifier,
)


class TestCheckLicense:
    """Tests for ModelVerifier.check_license method."""

    @pytest.mark.asyncio
    async def test_passes_when_mit_license(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Passes when HF model metadata has MIT license."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cardData": {"license": "mit"}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_passes_with_mit_license_variations(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Passes with various MIT license formats (case-insensitive)."""
        verifier = ModelVerifier(mock_chain_client)

        for license_value in ["MIT", "MIT License", "The MIT License (MIT)", "mit"]:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"cardData": {"license": license_value}}
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_repo_not_found(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError when repository is not found."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(LicenseError, match="not found"):
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_not_mit_license(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError when license is not MIT."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cardData": {"license": "apache-2.0"}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(LicenseError, match="MIT license required"):
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_no_license(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError when no license in metadata."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cardData": {}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(LicenseError, match="MIT license required"):
                await verifier.check_license("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_on_http_failure(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises LicenseError on HTTP failure."""
        verifier = ModelVerifier(mock_chain_client)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            with pytest.raises(LicenseError, match="Failed to check license"):
                await verifier.check_license("user/repo")


class TestFindOnnxFile:
    """Tests for ModelVerifier.find_onnx_file method."""

    @pytest.mark.asyncio
    async def test_returns_filename_and_size(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Returns (filename, size) tuple for found .onnx file."""
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
            filename, size = await verifier.find_onnx_file("user/repo")

        assert filename == "model.onnx"
        assert size == 50_000_000

    @pytest.mark.asyncio
    async def test_finds_any_onnx_filename(self, mock_chain_client: MagicMock) -> None:
        """Finds .onnx file with any name."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "my_custom_model.onnx", "size": 50_000_000},
            {"path": "README.md", "size": 1000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            filename, size = await verifier.find_onnx_file("user/repo")

        assert filename == "my_custom_model.onnx"
        assert size == 50_000_000

    @pytest.mark.asyncio
    async def test_raises_error_when_no_onnx_found(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ModelDownloadError when no .onnx file in repo."""
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
            with pytest.raises(ModelDownloadError, match="No .onnx file found"):
                await verifier.find_onnx_file("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_when_multiple_onnx_found(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ModelDownloadError when multiple .onnx files in repo."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "model.onnx", "size": 50_000_000},
            {"path": "model_v2.onnx", "size": 60_000_000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ModelDownloadError, match="Multiple .onnx files"):
                await verifier.find_onnx_file("user/repo")

    @pytest.mark.asyncio
    async def test_ignores_onnx_in_subdirectories(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Only finds .onnx files in root, not subdirectories."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "subdir/model.onnx", "size": 50_000_000},  # in subdir, ignored
            {"path": "README.md", "size": 1000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ModelDownloadError, match="No .onnx file found"):
                await verifier.find_onnx_file("user/repo")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            "price_predictor.onnx",
            "model-v2.onnx",
            "RealEstate_Model_2024.onnx",
            "a.onnx",
            "model123.onnx",
        ],
    )
    async def test_accepts_various_valid_onnx_filenames(
        self, mock_chain_client: MagicMock, filename: str
    ) -> None:
        """Accepts various valid .onnx filenames."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": filename, "size": 50_000_000},
            {"path": "README.md", "size": 1000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            returned_filename, size = await verifier.find_onnx_file("user/repo")

        assert returned_filename == filename
        assert size == 50_000_000

    @pytest.mark.asyncio
    async def test_ignores_non_onnx_files(self, mock_chain_client: MagicMock) -> None:
        """Ignores files that don't end with .onnx."""
        verifier = ModelVerifier(mock_chain_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "model.onnx.backup", "size": 50_000_000},  # not .onnx
            {"path": "model_onnx", "size": 50_000_000},  # not .onnx
            {"path": "README.md", "size": 1000},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(ModelDownloadError, match="No .onnx file found"):
                await verifier.find_onnx_file("user/repo")

    @pytest.mark.asyncio
    async def test_raises_error_on_http_error(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Raises ModelDownloadError on HTTP error."""
        verifier = ModelVerifier(mock_chain_client)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPError("API error")
            )
            with pytest.raises(ModelDownloadError, match="Failed to fetch file list"):
                await verifier.find_onnx_file("user/repo")


class TestCheckModelSize:
    """Tests for ModelVerifier.check_model_size method."""

    def test_passes_when_under_limit(self, mock_chain_client: MagicMock) -> None:
        """Does not raise when model is under limit."""
        verifier = ModelVerifier(mock_chain_client)
        # Should not raise
        verifier.check_model_size(50_000_000, 100_000_000, "model.onnx")

    def test_raises_error_when_over_limit(self, mock_chain_client: MagicMock) -> None:
        """Raises ModelTooLargeError when model exceeds limit."""
        verifier = ModelVerifier(mock_chain_client)

        with pytest.raises(ModelTooLargeError, match="exceeds limit"):
            verifier.check_model_size(500_000_000, 100_000_000, "model.onnx")

    def test_raises_error_at_exact_limit(self, mock_chain_client: MagicMock) -> None:
        """Raises ModelTooLargeError when model equals limit."""
        verifier = ModelVerifier(mock_chain_client)

        with pytest.raises(ModelTooLargeError, match="exceeds limit"):
            verifier.check_model_size(100_000_001, 100_000_000, "model.onnx")

    def test_includes_filename_in_error(self, mock_chain_client: MagicMock) -> None:
        """Error message includes filename."""
        verifier = ModelVerifier(mock_chain_client)

        with pytest.raises(ModelTooLargeError, match="my_model.onnx"):
            verifier.check_model_size(500_000_000, 100_000_000, "my_model.onnx")


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
        mock_extrinsic.block_number = 12345  # Block from Pylon
        mock_extrinsic.is_commitment_extrinsic.return_value = True
        mock_extrinsic.call.call_args = []  # No call args to parse
        mock_chain_client.get_extrinsic = AsyncMock(return_value=mock_extrinsic)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            commit_block = await verifier.verify_extrinsic_record(
                hotkey="5TestHotkey",
                hf_repo_id="user/repo",
                expected_hash="abc12345",
            )

        # Returns commit block from Pylon (trusted source)
        assert commit_block == 12345

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
