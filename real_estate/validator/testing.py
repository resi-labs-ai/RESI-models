"""Test-only model scheduler for local development (--test-models-dir)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from real_estate.models import DownloadResult

from real_estate.chain.models import ChainModelMetadata

logger = logging.getLogger(__name__)


class StaticModelScheduler:
    """Serves pre-existing local models, bypassing HF download and chain commitments."""

    def __init__(
        self,
        model_paths: dict[str, Path],
        metadata: dict[str, ChainModelMetadata],
    ):
        self._model_paths = model_paths
        self._known_commitments = metadata

    @property
    def known_commitments(self) -> dict[str, ChainModelMetadata]:
        return self._known_commitments

    def get_available_models(
        self, registered_hotkeys: set[str], current_block: int  # noqa: ARG002
    ) -> dict[str, Path]:
        # When test-models-dir is used, ignore registered_hotkeys filter
        # (test models have fake hotkeys not on chain)
        return dict(self._model_paths)

    async def run_pre_download(self, eval_time: datetime) -> dict[str, DownloadResult]:  # noqa: ARG002
        return {}  # Nothing to download

    async def run_catch_up(
        self, failed_hotkeys: set[str] | None = None  # noqa: ARG002
    ) -> dict[str, DownloadResult]:
        return {}


def build_static_scheduler(models_dir: Path) -> StaticModelScheduler:
    """
    Build a StaticModelScheduler from a directory of local models.

    Expected layout: {models_dir}/{hotkey}/model.onnx
    """
    model_paths: dict[str, Path] = {}
    metadata: dict[str, ChainModelMetadata] = {}

    for model_file in sorted(models_dir.glob("*/model.onnx")):
        hotkey = model_file.parent.name
        model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
        model_paths[hotkey] = model_file
        metadata[hotkey] = ChainModelMetadata(
            hotkey=hotkey,
            hf_repo_id=f"local/{hotkey}",
            model_hash=model_hash,
            block_number=1,
        )

    logger.info(
        f"Static scheduler loaded {len(model_paths)} models from {models_dir}"
    )
    return StaticModelScheduler(model_paths, metadata)
