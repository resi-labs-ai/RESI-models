"""Model inspector — static ONNX analysis in Docker containers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    InspectionBatchResult,
    InspectionConfig,
    ModelInspectionResult,
    RejectionReason,
)

if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container

logger = logging.getLogger(__name__)

# Path to inspection script (copied into container)
_INSPECTION_SCRIPT_PATH = Path(__file__).parent / "inspection_script.py"


class ModelInspector:
    """
    Pre-flight ONNX model inspection in Docker containers.

    Performs static analysis (no inference, no input data) to detect
    memorization indicators before running expensive evaluation.

    Checks:
    - LOOKUP_PATTERN: TopK/ArgMin/ArgMax + Gather ops (KNN lookup)
    - Any unused initializers (dead weight padding)
    - Price-like values in weights > threshold (memorized listings)

    Any check failing → model rejected, not evaluated, gets 0 weight.
    """

    def __init__(self, config: InspectionConfig):
        self._config = config
        self._client: docker.DockerClient | None = None

    @property
    def config(self) -> InspectionConfig:
        return self._config

    def _get_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            import docker

            self._client = docker.from_env()
            self._client.ping()
            self._ensure_image()
        return self._client

    def _ensure_image(self) -> None:
        """Ensure Docker image is available locally."""
        try:
            self._client.images.get(self._config.image)
        except Exception as e:
            raise RuntimeError(
                f"Docker image '{self._config.image}' not found. "
                f"Build it with: docker build -t {self._config.image} real_estate/evaluation/"
            ) from e

    async def inspect_all(self, model_paths: dict[str, Path]) -> InspectionBatchResult:
        """
        Inspect all models for memorization indicators.

        Runs inspections concurrently using asyncio.to_thread.

        Args:
            model_paths: Mapping of hotkey -> path to ONNX model file.

        Returns:
            InspectionBatchResult with rejected hotkeys and per-model results.
        """
        logger.info(f"Inspecting {len(model_paths)} models...")

        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def inspect_with_semaphore(
            hotkey: str, model_path: Path
        ) -> ModelInspectionResult:
            async with semaphore:
                return await asyncio.to_thread(
                    self._inspect_single_safe, hotkey, model_path
                )

        tasks = [
            inspect_with_semaphore(hotkey, model_path)
            for hotkey, model_path in model_paths.items()
        ]

        results = list(await asyncio.gather(*tasks))

        for result in results:
            if result.is_rejected:
                logger.warning(
                    f"Model rejected: {result.hotkey} — {result.rejection_reason.value}"
                )

        rejected_count = sum(1 for r in results if r.is_rejected)
        logger.info(
            f"Inspection complete: {len(results)} inspected, "
            f"{rejected_count} rejected"
        )

        return InspectionBatchResult(results=tuple(results))

    def _inspect_single_safe(self, hotkey: str, model_path: Path) -> ModelInspectionResult:
        """Inspect a single model, catching exceptions into a rejection result."""
        try:
            return self._inspect_single(hotkey, model_path)
        except Exception as e:
            logger.exception(f"Inspection failed for {hotkey}, rejecting")
            return ModelInspectionResult(
                hotkey=hotkey,
                has_lookup_pattern=False,
                has_unused_initializers=False,
                price_like_values=0,
                total_params=0,
                rejection_reason=RejectionReason.INSPECTION_FAILED,
                error=e,
            )

    def _inspect_single(self, hotkey: str, model_path: Path) -> ModelInspectionResult:
        """
        Inspect a single model in a Docker container.

        Copies model + inspection script to temp dir, runs container,
        reads back inspection_results.json, applies rejection rules.
        """
        client = self._get_client()

        with tempfile.TemporaryDirectory(prefix="onnx_inspect_") as workspace:
            workspace_path = Path(workspace)

            # Copy model and inspection script to workspace
            shutil.copy(model_path, workspace_path / "model.onnx")
            shutil.copy(_INSPECTION_SCRIPT_PATH, workspace_path / "inspection_container_script.py")

            volumes = {
                str(workspace_path): {"bind": "/workspace", "mode": "rw"},
            }

            container: Container | None = None
            try:
                container = client.containers.run(
                    self._config.image,
                    command=["python", "/workspace/inspection_container_script.py"],
                    volumes=volumes,
                    mem_limit=self._config.memory_limit,
                    nano_cpus=int(self._config.cpu_limit * 1e9),
                    network_disabled=True,
                    read_only=True,
                    pids_limit=50,
                    detach=True,
                    remove=False,
                )

                result = container.wait(timeout=self._config.timeout_seconds)
                exit_code = result.get("StatusCode", -1)
                logs = container.logs().decode("utf-8", errors="replace")

                if exit_code != 0:
                    logger.warning(
                        f"Inspection container failed for {hotkey} "
                        f"(exit={exit_code}): {logs[-500:]}"
                    )
                    raise RuntimeError(f"Inspection container exited with code {exit_code}")

                # Read results
                results_path = workspace_path / "inspection_results.json"
                if not results_path.exists():
                    raise RuntimeError("inspection_results.json not created")

                with open(results_path) as f:
                    data = json.load(f)

                return self._apply_rejection_rules(hotkey, data)

            finally:
                if container:
                    with contextlib.suppress(Exception):
                        container.remove(force=True)

    def _apply_rejection_rules(
        self, hotkey: str, data: dict
    ) -> ModelInspectionResult:
        """Apply rejection rules to inspection data."""
        has_lookup_pattern = bool(data.get("lookup_pattern", False))
        has_unused_initializers = float(data.get("unused_initializer_ratio", 0.0)) > 0
        price_count = int(data.get("price_like_values_total", 0))
        total_params = int(data.get("total_params", 0))

        # Pipeline: early exit on first failure
        rejection_reason: RejectionReason | None = None
        if has_lookup_pattern:
            rejection_reason = RejectionReason.LOOKUP_PATTERN
        elif self._config.reject_unused_initializers and has_unused_initializers:
            rejection_reason = RejectionReason.UNUSED_INITIALIZERS
        elif price_count > self._config.price_count_threshold:
            rejection_reason = RejectionReason.PRICES_IN_WEIGHTS

        return ModelInspectionResult(
            hotkey=hotkey,
            has_lookup_pattern=has_lookup_pattern,
            has_unused_initializers=has_unused_initializers,
            price_like_values=price_count,
            total_params=total_params,
            rejection_reason=rejection_reason,
        )
