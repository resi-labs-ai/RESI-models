"""
Docker-based ONNX model execution.

Runs ONNX models in isolated Docker containers for security.
Models cannot access network, have limited CPU/memory, and are killed on timeout.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .errors import (
    DockerExecutionError,
    DockerImageError,
    DockerNotAvailableError,
    InferenceTimeoutError,
    InvalidPredictionError,
)

if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DockerConfig:
    """Configuration for Docker-based model execution."""

    image: str = "resi-onnx-runner:latest"
    """Docker image to use. Must have Python, numpy, and onnxruntime."""

    memory_limit: str = "2g"
    """Memory limit for container (e.g., '2g', '512m')."""

    cpu_limit: float = 1.0
    """CPU limit (1.0 = 1 core, 0.5 = half core)."""

    timeout_seconds: int = 300
    """Maximum time for inference (5 minutes default)."""

    network_disabled: bool = True
    """Disable network access for security."""

    read_only_rootfs: bool = True
    """Make root filesystem read-only for security."""

    pids_limit: int = 50
    """Maximum number of processes in container."""


@dataclass
class InferenceResult:
    """Result of running inference in Docker."""

    predictions: np.ndarray
    inference_time_ms: float
    container_logs: str = ""


# Path to inference script (copied into container)
_INFERENCE_SCRIPT_PATH = Path(__file__).parent / "inference_script.py"


class DockerRunner:
    """
    Execute ONNX models in isolated Docker containers.

    Security features:
    - No network access
    - Memory/CPU limits
    - Read-only filesystem (except /workspace)
    - Process limits
    - Timeout enforcement

    Usage:
        runner = DockerRunner(DockerConfig())
        result = runner.run_inference(model_path, input_data)
        predictions = result.predictions
    """

    def __init__(self, config: DockerConfig):
        """
        Initialize Docker runner.

        Args:
            config: Docker configuration.

        Raises:
            DockerNotAvailableError: If Docker daemon is not available.
        """
        self._config = config
        self._client: docker.DockerClient | None = None
        self._image_ready = False

    def _get_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
                # Test connection
                self._client.ping()
            except ImportError as e:
                raise DockerNotAvailableError(
                    "docker package not installed. Install with: pip install docker"
                ) from e
            except Exception as e:
                raise DockerNotAvailableError(
                    f"Cannot connect to Docker daemon: {e}"
                ) from e

        return self._client

    def _ensure_image(self) -> None:
        """Ensure Docker image is available locally."""
        if self._image_ready:
            return

        client = self._get_client()

        try:
            client.images.get(self._config.image)
            logger.debug(f"Docker image {self._config.image} available")
        except Exception as e:
            raise DockerImageError(
                f"Docker image '{self._config.image}' not found. "
                f"Build it with: docker build -t {self._config.image} real_estate/evaluation/"
            ) from e

        self._image_ready = True

    def run_inference(
        self,
        model_path: Path,
        input_data: np.ndarray,
    ) -> InferenceResult:
        """
        Run inference on ONNX model in Docker container.

        Args:
            model_path: Path to .onnx model file
            input_data: Input features as numpy array (N x F)

        Returns:
            InferenceResult with predictions and timing

        Raises:
            DockerNotAvailableError: If Docker is not available
            DockerImageError: If image cannot be pulled
            DockerExecutionError: If container execution fails
            InferenceTimeoutError: If inference exceeds timeout
            InvalidPredictionError: If model produces invalid output
        """
        self._ensure_image()
        client = self._get_client()

        # Create temporary workspace
        with tempfile.TemporaryDirectory(prefix="onnx_eval_") as workspace:
            workspace_path = Path(workspace)

            # Copy model to workspace
            import shutil

            model_dest = workspace_path / "model.onnx"
            shutil.copy(model_path, model_dest)

            # Save input data
            input_path = workspace_path / "input.npy"
            np.save(input_path, input_data.astype(np.float32))

            # Copy inference script
            script_path = workspace_path / "run_inference.py"
            shutil.copy(_INFERENCE_SCRIPT_PATH, script_path)

            # Prepare container config
            volumes = {
                str(workspace_path): {"bind": "/workspace", "mode": "rw"},
            }

            # Run container
            start_time = time.time()
            container: Container | None = None

            try:
                container = client.containers.run(
                    self._config.image,
                    command=["python", "/workspace/run_inference.py"],
                    volumes=volumes,
                    mem_limit=self._config.memory_limit,
                    nano_cpus=int(self._config.cpu_limit * 1e9),
                    network_disabled=self._config.network_disabled,
                    read_only=self._config.read_only_rootfs,
                    pids_limit=self._config.pids_limit,
                    detach=True,
                    remove=False,  # We'll remove manually after getting logs
                )

                # Wait for completion with timeout
                result = container.wait(timeout=self._config.timeout_seconds)
                exit_code = result.get("StatusCode", -1)
                logs = container.logs().decode("utf-8", errors="replace")

                elapsed_ms = (time.time() - start_time) * 1000

                if exit_code != 0:
                    # Exit code 6 = output validation failed (NaN/Inf)
                    if exit_code == 6:
                        # Extract just the error line from container logs
                        error_detail = "NaN or Inf detected"
                        for line in logs.splitlines():
                            if "[ERROR]" in line:
                                error_detail = line.replace("[ERROR]", "").strip()
                                break
                        raise InvalidPredictionError(
                            f"Model produced invalid predictions: {error_detail}"
                        )
                    raise DockerExecutionError(
                        f"Container exited with code {exit_code}",
                        exit_code=exit_code,
                        logs=logs,
                    )

                # Load predictions
                output_path = workspace_path / "output.npy"
                if not output_path.exists():
                    raise DockerExecutionError(
                        "Output file not created by container",
                        exit_code=exit_code,
                        logs=logs,
                    )

                predictions = np.load(output_path)

                # Validate predictions
                predictions = self._validate_predictions(predictions, len(input_data))

                return InferenceResult(
                    predictions=predictions,
                    inference_time_ms=elapsed_ms,
                    container_logs=logs,
                )

            except Exception as e:
                if isinstance(
                    e,
                    (
                        DockerExecutionError,
                        InferenceTimeoutError,
                        InvalidPredictionError,
                    ),
                ):
                    raise

                # Check if timeout
                if "timeout" in str(e).lower() or "read timed out" in str(e).lower():
                    raise InferenceTimeoutError(
                        f"Inference timed out after {self._config.timeout_seconds}s",
                        exit_code=None,
                        logs=container.logs().decode() if container else "",
                    ) from e

                raise DockerExecutionError(f"Container execution failed: {e}") from e

            finally:
                # Clean up container
                if container:
                    with contextlib.suppress(Exception):
                        container.remove(force=True)

    def _validate_predictions(
        self,
        predictions: np.ndarray,
        expected_length: int,
    ) -> np.ndarray:
        """Validate and normalize predictions."""
        # Flatten if needed
        predictions = predictions.flatten()

        # Check length
        if len(predictions) != expected_length:
            raise InvalidPredictionError(
                f"Prediction count mismatch: got {len(predictions)}, expected {expected_length}"
            )

        # Check for NaN/Inf (these break metrics calculations)
        if np.any(np.isnan(predictions)):
            nan_count = np.sum(np.isnan(predictions))
            raise InvalidPredictionError(f"Model produced {nan_count} NaN predictions")

        if np.any(np.isinf(predictions)):
            inf_count = np.sum(np.isinf(predictions))
            raise InvalidPredictionError(f"Model produced {inf_count} Inf predictions")

        # Note: We allow negative predictions - metrics will naturally penalize them heavily

        return predictions
