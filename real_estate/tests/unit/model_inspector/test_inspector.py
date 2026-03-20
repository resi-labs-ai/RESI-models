"""Tests for ModelInspector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from real_estate.model_inspector import (
    InspectionConfig,
    ModelInspector,
    RejectionReason,
)


def _make_inspector_with_mock(
    inspector: ModelInspector,
    model_paths: dict[str, Path],
    inspection_results: dict[str, dict],
):
    """
    Wire up a mock Docker client that returns pre-canned inspection results.

    Args:
        inspector: The inspector to patch.
        model_paths: Mapping of hotkey -> model path.
        inspection_results: Mapping of hotkey -> container JSON output.
    """
    hotkeys = list(model_paths.keys())
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        hotkey = hotkeys[call_count]
        call_count += 1
        workspace = list(kwargs["volumes"].keys())[0]
        Path(workspace, "inspection_results.json").write_text(
            json.dumps(inspection_results[hotkey])
        )
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = b""
        return container

    mock_client = MagicMock()
    mock_client.containers.run.side_effect = fake_run
    inspector._client = mock_client


_CLEAN = {
    "lookup_pattern": False,
    "unused_initializer_ratio": 0.0,
    "price_like_values_total": 50,
    "total_params": 5_000_000,
}


class TestModelInspector:
    @pytest.fixture
    def inspector(self) -> ModelInspector:
        return ModelInspector(InspectionConfig(
            price_count_threshold=50_000,
            reject_unused_initializers=True,
        ))

    @pytest.fixture
    def model_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "model.onnx"
        path.write_bytes(b"fake")
        return path

    @pytest.mark.asyncio
    async def test_lookup_pattern_rejects(self, inspector: ModelInspector, model_path: Path) -> None:
        """LOOKUP_PATTERN in container output triggers rejection."""
        paths = {"bad": model_path}
        _make_inspector_with_mock(inspector, paths, {
            "bad": {**_CLEAN, "lookup_pattern": True},
        })

        result = await inspector.inspect_all(paths)
        assert result.is_rejected("bad")
        assert result.results[0].rejection_reason == RejectionReason.LOOKUP_PATTERN

    @pytest.mark.asyncio
    async def test_unused_initializers_rejects(self, inspector: ModelInspector, model_path: Path) -> None:
        """Any unused initializers triggers rejection."""
        paths = {"bad": model_path}
        _make_inspector_with_mock(inspector, paths, {
            "bad": {**_CLEAN, "unused_initializer_ratio": 0.89},
        })

        result = await inspector.inspect_all(paths)
        assert result.is_rejected("bad")
        assert result.results[0].rejection_reason == RejectionReason.UNUSED_INITIALIZERS

    @pytest.mark.asyncio
    async def test_price_values_rejects(self, inspector: ModelInspector, model_path: Path) -> None:
        """Excessive price-like values in weights triggers rejection."""
        paths = {"bad": model_path}
        _make_inspector_with_mock(inspector, paths, {
            "bad": {**_CLEAN, "price_like_values_total": 2_000_000},
        })

        result = await inspector.inspect_all(paths)
        assert result.is_rejected("bad")
        assert result.results[0].rejection_reason == RejectionReason.PRICES_IN_WEIGHTS

    @pytest.mark.asyncio
    async def test_clean_model_passes(self, inspector: ModelInspector, model_path: Path) -> None:
        """Model passing all checks is not rejected."""
        paths = {"good": model_path}
        _make_inspector_with_mock(inspector, paths, {"good": _CLEAN})

        result = await inspector.inspect_all(paths)
        assert not result.is_rejected("good")

    @pytest.mark.asyncio
    async def test_rejection_priority(self, inspector: ModelInspector, model_path: Path) -> None:
        """Early-exit: lookup pattern takes priority over other failures."""
        paths = {"bad": model_path}
        _make_inspector_with_mock(inspector, paths, {
            "bad": {
                "lookup_pattern": True,
                "unused_initializer_ratio": 0.89,
                "price_like_values_total": 2_000_000,
                "total_params": 47_000_000,
            },
        })

        result = await inspector.inspect_all(paths)
        assert result.results[0].rejection_reason == RejectionReason.LOOKUP_PATTERN

    @pytest.mark.asyncio
    async def test_docker_failure_rejects(self, inspector: ModelInspector, model_path: Path) -> None:
        """Docker failure causes rejection with error details."""
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = RuntimeError("Docker exploded")
        inspector._client = mock_client

        result = await inspector.inspect_all({"broken": model_path})
        assert result.is_rejected("broken")
        assert "Docker exploded" in result.results[0].error_message

    @pytest.mark.asyncio
    async def test_container_nonzero_exit_rejects(self, inspector: ModelInspector, model_path: Path) -> None:
        """Non-zero exit code causes rejection."""
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 1}
        container.logs.return_value = b"error"

        mock_client = MagicMock()
        mock_client.containers.run.return_value = container
        inspector._client = mock_client

        result = await inspector.inspect_all({"failing": model_path})
        assert result.is_rejected("failing")
        assert result.results[0].error is not None

    @pytest.mark.asyncio
    async def test_mixed_models(self, inspector: ModelInspector, tmp_path: Path) -> None:
        """Mix of clean and rejected models."""
        clean_path = tmp_path / "clean.onnx"
        clean_path.write_bytes(b"fake")
        bad_path = tmp_path / "bad.onnx"
        bad_path.write_bytes(b"fake")

        paths = {"clean": clean_path, "bad": bad_path}
        _make_inspector_with_mock(inspector, paths, {
            "clean": _CLEAN,
            "bad": {**_CLEAN, "lookup_pattern": True},
        })

        result = await inspector.inspect_all(paths)
        assert not result.is_rejected("clean")
        assert result.is_rejected("bad")
        assert len(result.rejected_hotkeys) == 1
