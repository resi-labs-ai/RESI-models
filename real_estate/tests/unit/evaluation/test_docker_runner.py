"""Unit tests for Docker runner with mocked Docker client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from real_estate.evaluation.docker_runner import (
    _INFERENCE_SCRIPT_PATH,
    DockerConfig,
    DockerRunner,
    InferenceResult,
)
from real_estate.evaluation.errors import (
    DockerExecutionError,
    DockerImageError,
    DockerNotAvailableError,
    InferenceTimeoutError,
    InvalidPredictionError,
)


@pytest.fixture
def mock_docker_module():
    """Create a mock docker module and inject it into sys.modules."""
    mock_docker = MagicMock()
    mock_client = MagicMock()
    mock_docker.from_env.return_value = mock_client

    with patch.dict(sys.modules, {"docker": mock_docker}):
        yield mock_docker, mock_client


class TestDockerRunnerErrorHandling:
    """Tests for DockerRunner error handling."""

    def test_raises_when_docker_daemon_unavailable(self) -> None:
        """Raises DockerNotAvailableError when daemon not running."""
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = Exception("Connection refused")

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with pytest.raises(DockerNotAvailableError, match="Cannot connect"):
                runner._get_client()

    def test_raises_when_image_not_found(self, mock_docker_module) -> None:
        """Raises DockerImageError with build instructions if image not found."""
        mock_docker, mock_client = mock_docker_module
        mock_client.images.get.side_effect = Exception("Not found")

        runner = DockerRunner(DockerConfig())

        with pytest.raises(DockerImageError, match="not found.*Build it with"):
            runner._ensure_image()


class TestDockerRunnerValidatePredictions:
    """Tests for DockerRunner._validate_predictions method."""

    def test_valid_predictions(self) -> None:
        """Valid predictions pass validation."""
        runner = DockerRunner(DockerConfig())
        predictions = np.array([[100000], [200000], [300000]])

        result = runner._validate_predictions(predictions, expected_length=3)

        assert result.shape == (3,)
        assert np.array_equal(result, [100000, 200000, 300000])

    def test_length_mismatch_raises(self) -> None:
        """Raises InvalidPredictionError on length mismatch."""
        runner = DockerRunner(DockerConfig())
        predictions = np.array([100000, 200000])

        with pytest.raises(InvalidPredictionError, match="count mismatch"):
            runner._validate_predictions(predictions, expected_length=5)

    def test_nan_values_raise(self) -> None:
        """Raises InvalidPredictionError on NaN values."""
        runner = DockerRunner(DockerConfig())
        predictions = np.array([100000, np.nan, 300000])

        with pytest.raises(InvalidPredictionError, match="NaN"):
            runner._validate_predictions(predictions, expected_length=3)

    def test_inf_values_raise(self) -> None:
        """Raises InvalidPredictionError on Inf values."""
        runner = DockerRunner(DockerConfig())
        predictions = np.array([100000, np.inf, 300000])

        with pytest.raises(InvalidPredictionError, match="Inf"):
            runner._validate_predictions(predictions, expected_length=3)

    def test_negative_values_allowed(self) -> None:
        """Negative values are allowed (metrics will penalize them)."""
        runner = DockerRunner(DockerConfig())
        predictions = np.array([100000, -50000, 300000])

        result = runner._validate_predictions(predictions, expected_length=3)

        assert np.array_equal(result, [100000, -50000, 300000])


class TestDockerRunnerRunInference:
    """Tests for DockerRunner.run_inference method."""

    def test_successful_inference(self, tmp_path: Path) -> None:
        """Successful inference returns predictions."""
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"[SUCCESS] Generated 3 predictions"
        mock_client.containers.run.return_value = mock_container

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy model")

        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                workspace = tmp_path / "workspace"
                workspace.mkdir()
                mock_tmpdir.return_value.__enter__.return_value = str(workspace)

                # Simulate container writing output
                output_path = workspace / "output.npy"
                np.save(output_path, np.array([100000, 200000, 300000]))

                result = runner.run_inference(model_path, input_data)

        assert isinstance(result, InferenceResult)
        assert len(result.predictions) == 3
        assert result.inference_time_ms > 0
        mock_container.remove.assert_called_once_with(force=True)

    def test_container_nonzero_exit(self, tmp_path: Path) -> None:
        """Raises DockerExecutionError on non-zero exit."""
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 4}
        mock_container.logs.return_value = b"[ERROR] Inference failed"
        mock_client.containers.run.return_value = mock_container

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy")

        input_data = np.array([[1.0, 2.0]])

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                workspace = tmp_path / "workspace"
                workspace.mkdir()
                mock_tmpdir.return_value.__enter__.return_value = str(workspace)

                with pytest.raises(DockerExecutionError) as exc_info:
                    runner.run_inference(model_path, input_data)

        assert exc_info.value.exit_code == 4
        assert "[ERROR]" in exc_info.value.logs

    def test_container_timeout(self, tmp_path: Path) -> None:
        """Raises InferenceTimeoutError on timeout."""
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.wait.side_effect = Exception("read timed out")
        mock_container.logs.return_value = b"[INFO] Still running..."
        mock_client.containers.run.return_value = mock_container

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy")

        input_data = np.array([[1.0, 2.0]])

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                workspace = tmp_path / "workspace"
                workspace.mkdir()
                mock_tmpdir.return_value.__enter__.return_value = str(workspace)

                with pytest.raises(InferenceTimeoutError, match="timed out"):
                    runner.run_inference(model_path, input_data)

    def test_container_cleaned_up_on_success(self, tmp_path: Path) -> None:
        """Container is removed after successful inference."""
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"success"
        mock_client.containers.run.return_value = mock_container

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy")

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                workspace = tmp_path / "workspace"
                workspace.mkdir()
                mock_tmpdir.return_value.__enter__.return_value = str(workspace)
                np.save(workspace / "output.npy", np.array([100000]))

                runner.run_inference(model_path, np.array([[1.0]]))

        mock_container.remove.assert_called_once_with(force=True)

    def test_container_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        """Container is removed even after failure."""
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b"error"
        mock_client.containers.run.return_value = mock_container

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy")

        with patch.dict(sys.modules, {"docker": mock_docker}):
            runner = DockerRunner(DockerConfig())

            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                workspace = tmp_path / "workspace"
                workspace.mkdir(exist_ok=True)
                mock_tmpdir.return_value.__enter__.return_value = str(workspace)

                with pytest.raises(DockerExecutionError):
                    runner.run_inference(model_path, np.array([[1.0]]))

        mock_container.remove.assert_called_once_with(force=True)


class TestInferenceScriptPath:
    """Tests for inference script file."""

    def test_inference_script_exists(self) -> None:
        """Inference script file exists."""
        assert _INFERENCE_SCRIPT_PATH.exists()

    def test_inference_script_is_python(self) -> None:
        """Inference script is valid Python."""
        content = _INFERENCE_SCRIPT_PATH.read_text()

        # Should be parseable Python
        compile(content, _INFERENCE_SCRIPT_PATH, "exec")

    def test_inference_script_has_main(self) -> None:
        """Inference script has main function."""
        content = _INFERENCE_SCRIPT_PATH.read_text()

        assert "def main():" in content
        assert 'if __name__ == "__main__":' in content
