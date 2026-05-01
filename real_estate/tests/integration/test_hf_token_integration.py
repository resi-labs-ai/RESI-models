"""Integration tests verifying HF token is actually sent in real HTTP requests.

These tests call the real ModelVerifier.check_license() and ModelDownloader
code paths against HuggingFace, intercepting at the HTTP layer to prove the
Authorization header is present in the actual outgoing requests.

They are read-only — no writes to any repo or chain.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from real_estate.models import (
    DownloadConfig,
    ModelCache,
    ModelDownloader,
    ModelVerifier,
)

HF_REPO = "ftjbvihzt/test_model"


@pytest.fixture
def hf_token() -> str:
    """Get HF token from environment, skip if not set."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        pytest.skip("HF_TOKEN not set — skipping live HF integration test")
    return token


class TestVerifierSendsAuthHeader:
    """Prove real ModelVerifier methods send Authorization header."""

    @pytest.mark.asyncio
    async def test_check_license_sends_bearer_token(self, hf_token: str) -> None:
        """Call the real check_license() and intercept the outgoing request."""
        captured_requests: list[httpx.Request] = []

        _OriginalClient = httpx.AsyncClient

        class CapturingClient(_OriginalClient):
            async def send(self, request, **kwargs):
                captured_requests.append(request)
                return await super().send(request, **kwargs)

        mock_chain = MagicMock()
        verifier = ModelVerifier(mock_chain, hf_token=hf_token)

        with patch("real_estate.models.verifier.httpx.AsyncClient", CapturingClient):
            await verifier.check_license(HF_REPO)

        assert len(captured_requests) > 0, "No HTTP requests were captured"

        # Find the request to HuggingFace API
        hf_request = next(
            (r for r in captured_requests if "huggingface.co" in str(r.url)), None
        )
        assert hf_request is not None, "No request to huggingface.co found"

        auth_header = hf_request.headers.get("authorization")
        assert auth_header == f"Bearer {hf_token}", (
            f"Expected 'Bearer {hf_token[:8]}...', got: {auth_header}"
        )

    @pytest.mark.asyncio
    async def test_find_onnx_file_sends_bearer_token(self, hf_token: str) -> None:
        """Call the real find_onnx_file() and intercept the outgoing request."""
        captured_requests: list[httpx.Request] = []

        _OriginalClient = httpx.AsyncClient

        class CapturingClient(_OriginalClient):
            async def send(self, request, **kwargs):
                captured_requests.append(request)
                return await super().send(request, **kwargs)

        mock_chain = MagicMock()
        verifier = ModelVerifier(mock_chain, hf_token=hf_token)

        with patch("real_estate.models.verifier.httpx.AsyncClient", CapturingClient):
            filename, size = await verifier.find_onnx_file(HF_REPO)

        assert filename.endswith(".onnx")
        assert size > 0

        hf_request = next(
            (r for r in captured_requests if "huggingface.co" in str(r.url)), None
        )
        assert hf_request is not None
        assert hf_request.headers.get("authorization") == f"Bearer {hf_token}"

    @pytest.mark.asyncio
    async def test_check_license_without_token_sends_no_auth(self) -> None:
        """Without token, no Authorization header is sent."""
        captured_requests: list[httpx.Request] = []

        _OriginalClient = httpx.AsyncClient

        class CapturingClient(_OriginalClient):
            async def send(self, request, **kwargs):
                captured_requests.append(request)
                return await super().send(request, **kwargs)

        mock_chain = MagicMock()
        verifier = ModelVerifier(mock_chain, hf_token=None)

        with patch("real_estate.models.verifier.httpx.AsyncClient", CapturingClient):
            try:
                await verifier.check_license(HF_REPO)
            except Exception:
                pass  # License check may fail — we only care about the header

        hf_request = next(
            (r for r in captured_requests if "huggingface.co" in str(r.url)), None
        )
        assert hf_request is not None
        assert "authorization" not in hf_request.headers


class TestDownloaderSendsToken:
    """Prove real ModelDownloader passes token to hf_hub_download."""

    @pytest.mark.asyncio
    async def test_hf_hub_download_receives_token(
        self, hf_token: str, tmp_path: Path
    ) -> None:
        """Call real _download_with_retry and verify token kwarg hits hf_hub_download."""
        captured_kwargs: dict = {}

        # We need the real hf_hub_download to still work, but also capture its args
        from huggingface_hub import hf_hub_download as real_hf_hub_download

        def capturing_hf_hub_download(**kwargs):
            captured_kwargs.update(kwargs)
            return real_hf_hub_download(**kwargs)

        cache = ModelCache(tmp_path / "cache")
        mock_chain = MagicMock()
        verifier = ModelVerifier(mock_chain, hf_token=hf_token)
        downloader = ModelDownloader(
            config=DownloadConfig(max_retries=1),
            cache=cache,
            verifier=verifier,
            hf_token=hf_token,
        )

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            side_effect=capturing_hf_hub_download,
        ):
            path = await downloader._download_with_retry(HF_REPO, "test_model.onnx")

        assert path.exists()
        assert path.stat().st_size > 0
        assert captured_kwargs.get("token") == hf_token, (
            f"Expected token='{hf_token[:8]}...', got: {captured_kwargs.get('token')}"
        )

    @pytest.mark.asyncio
    async def test_hf_hub_download_receives_none_without_token(
        self, tmp_path: Path
    ) -> None:
        """Without token, hf_hub_download receives token=None."""
        captured_kwargs: dict = {}

        from huggingface_hub import hf_hub_download as real_hf_hub_download

        def capturing_hf_hub_download(**kwargs):
            captured_kwargs.update(kwargs)
            return real_hf_hub_download(**kwargs)

        cache = ModelCache(tmp_path / "cache")
        mock_chain = MagicMock()
        verifier = ModelVerifier(mock_chain, hf_token=None)
        downloader = ModelDownloader(
            config=DownloadConfig(max_retries=1),
            cache=cache,
            verifier=verifier,
            hf_token=None,
        )

        with patch(
            "real_estate.models.downloader.hf_hub_download",
            side_effect=capturing_hf_hub_download,
        ):
            path = await downloader._download_with_retry(HF_REPO, "test_model.onnx")

        assert path.exists()
        assert captured_kwargs.get("token") is None
