"""Unit tests verifying HF token propagates through the dependency chain."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from real_estate.models import (
    DownloadConfig,
    ModelCache,
    ModelDownloader,
    ModelVerifier,
)
from real_estate.models.factory import create_model_scheduler


HF_TOKEN = "hf_test_token_abc123"


class TestDownloaderTokenPropagation:
    """Verify HF token reaches hf_hub_download call."""

    def test_downloader_stores_token(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Token passed to constructor is stored on the instance."""
        downloader = ModelDownloader(
            download_config, mock_cache, mock_verifier, hf_token=HF_TOKEN
        )
        assert downloader._hf_token == HF_TOKEN

    def test_downloader_stores_none_when_no_token(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """No token means _hf_token is None."""
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)
        assert downloader._hf_token is None

    @pytest.mark.asyncio
    async def test_token_passed_to_hf_hub_download(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Token is forwarded to hf_hub_download as the 'token' kwarg."""
        downloader = ModelDownloader(
            download_config, mock_cache, mock_verifier, hf_token=HF_TOKEN
        )

        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake-onnx")

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            return_value=str(fake_model),
        ) as mock_hf_download:
            await downloader._download_with_retry("testuser/repo", "model.onnx")

            mock_hf_download.assert_called_once()
            _, kwargs = mock_hf_download.call_args
            assert kwargs["token"] == HF_TOKEN

    @pytest.mark.asyncio
    async def test_none_token_passed_when_not_configured(
        self,
        download_config: DownloadConfig,
        mock_cache: MagicMock,
        mock_verifier: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When no token configured, None is passed to hf_hub_download."""
        downloader = ModelDownloader(download_config, mock_cache, mock_verifier)

        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake-onnx")

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            return_value=str(fake_model),
        ) as mock_hf_download:
            await downloader._download_with_retry("testuser/repo", "model.onnx")

            _, kwargs = mock_hf_download.call_args
            assert kwargs["token"] is None


class TestVerifierTokenPropagation:
    """Verify HF token reaches httpx headers in verifier."""

    def test_verifier_stores_token(self, mock_chain_client: MagicMock) -> None:
        """Token passed to constructor is stored."""
        verifier = ModelVerifier(mock_chain_client, hf_token=HF_TOKEN)
        assert verifier._hf_token == HF_TOKEN

    def test_http_headers_include_bearer_when_token_set(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Authorization header is set when token is provided."""
        verifier = ModelVerifier(mock_chain_client, hf_token=HF_TOKEN)
        headers = verifier._http_headers
        assert headers == {"Authorization": f"Bearer {HF_TOKEN}"}

    def test_http_headers_empty_when_no_token(
        self, mock_chain_client: MagicMock
    ) -> None:
        """No headers when token is not provided."""
        verifier = ModelVerifier(mock_chain_client)
        assert verifier._http_headers == {}

    @pytest.mark.asyncio
    async def test_check_license_uses_token_header(
        self, mock_chain_client: MagicMock
    ) -> None:
        """check_license passes the Authorization header to httpx."""
        from real_estate.models.exclusive_license import (
            EXCLUSIVE_LICENSE_HASH,
            EXCLUSIVE_LICENSE_LINK,
            EXCLUSIVE_LICENSE_TEXT,
        )

        verifier = ModelVerifier(mock_chain_client, hf_token=HF_TOKEN)

        with patch("real_estate.models.verifier.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            # First call: _fetch_model_info (HF API)
            mock_api_response = MagicMock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = {
                "cardData": {
                    "license": "other",
                    "license_name": "resi-exclusive",
                    "license_link": EXCLUSIVE_LICENSE_LINK,
                }
            }
            # Second call: LICENSE file fetch
            mock_license_response = MagicMock()
            mock_license_response.status_code = 200
            mock_license_response.text = EXCLUSIVE_LICENSE_TEXT
            mock_client.get = AsyncMock(
                side_effect=[mock_api_response, mock_license_response]
            )
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await verifier.check_license("testuser/repo")
            assert result == "exclusive"

            # Verify token was passed to httpx client (first call)
            assert mock_client_cls.call_count >= 1
            first_call_kwargs = mock_client_cls.call_args_list[0]
            assert first_call_kwargs.kwargs.get("headers") == {
                "Authorization": f"Bearer {HF_TOKEN}"
            }


class TestEmptyStringTokenHandling:
    """Verify empty-string token is treated as no token (falsy)."""

    def test_verifier_empty_string_produces_no_auth_header(
        self, mock_chain_client: MagicMock
    ) -> None:
        """Empty string token must not produce an Authorization header.

        This matches the validator.py `config.hf_token or None` conversion
        at line 665 — but the verifier should also be safe if it somehow
        receives an empty string directly.
        """
        verifier = ModelVerifier(mock_chain_client, hf_token="")
        assert verifier._http_headers == {}

    def test_factory_with_empty_string_token_passes_empty_to_components(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """If factory receives empty string (bypassing validator conversion),
        verifier still produces no auth headers due to falsy check."""
        scheduler = create_model_scheduler(
            chain_client=mock_chain_client,
            cache_dir=tmp_path / "cache",
            hf_token="",
        )

        verifier = scheduler._downloader._verifier
        assert verifier._http_headers == {}


class TestFactoryTokenPropagation:
    """Verify create_model_scheduler passes token to downloader and verifier."""

    def test_factory_passes_token_to_components(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """create_model_scheduler wires hf_token into both verifier and downloader."""
        scheduler = create_model_scheduler(
            chain_client=mock_chain_client,
            cache_dir=tmp_path / "cache",
            hf_token=HF_TOKEN,
        )

        # Check the downloader got the token
        downloader = scheduler._downloader
        assert downloader._hf_token == HF_TOKEN

        # Check the verifier got the token
        verifier = downloader._verifier
        assert verifier._hf_token == HF_TOKEN

    def test_factory_none_token_when_not_provided(
        self, mock_chain_client: MagicMock, tmp_path: Path
    ) -> None:
        """Without hf_token arg, both components get None."""
        scheduler = create_model_scheduler(
            chain_client=mock_chain_client,
            cache_dir=tmp_path / "cache",
        )

        downloader = scheduler._downloader
        assert downloader._hf_token is None

        verifier = downloader._verifier
        assert verifier._hf_token is None
