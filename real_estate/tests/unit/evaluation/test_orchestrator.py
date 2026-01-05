"""Unit tests for evaluation orchestrator with mocked DockerRunner."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from real_estate.evaluation.docker_runner import InferenceResult
from real_estate.evaluation.errors import DockerExecutionError, InferenceTimeoutError
from real_estate.evaluation.models import EvaluationBatch
from real_estate.evaluation.orchestrator import (
    EvaluationOrchestrator,
    OrchestratorConfig,
)


class TestEvaluationOrchestratorEvaluateAll:
    """Tests for EvaluationOrchestrator.evaluate_all method."""

    @pytest.fixture
    def mock_docker_runner(self) -> MagicMock:
        """Create mock DockerRunner."""
        runner = MagicMock()
        runner.run_inference = MagicMock()
        return runner

    @pytest.fixture
    def sample_features(self) -> np.ndarray:
        """Sample input features."""
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    @pytest.fixture
    def sample_ground_truth(self) -> np.ndarray:
        """Sample ground truth values."""
        return np.array([100000, 200000, 300000])

    @pytest.mark.asyncio
    async def test_evaluate_all_successful(
        self,
        mock_docker_runner: MagicMock,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Successful evaluation returns EvaluationBatch."""
        # Setup mock inference result
        mock_docker_runner.run_inference.return_value = InferenceResult(
            predictions=np.array([105000, 195000, 310000]),
            inference_time_ms=150.0,
            container_logs="[SUCCESS]",
        )

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_docker_runner

        models = {
            "hotkey1": tmp_path / "model1.onnx",
            "hotkey2": tmp_path / "model2.onnx",
        }

        batch = await orchestrator.evaluate_all(
            models=models,
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert isinstance(batch, EvaluationBatch)
        assert batch.successful_count == 2
        assert batch.failed_count == 0
        assert batch.dataset_size == 3

    @pytest.mark.asyncio
    async def test_evaluate_all_mixed_results(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Mixed success/failure returns correct counts."""
        mock_runner = MagicMock()

        # First model succeeds, second fails
        def mock_inference(model_path, input_data):
            if "model1" in str(model_path):
                return InferenceResult(
                    predictions=np.array([105000, 195000, 310000]),
                    inference_time_ms=150.0,
                )
            raise DockerExecutionError("Container failed", exit_code=1, logs="error")

        mock_runner.run_inference.side_effect = mock_inference

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        models = {
            "hotkey1": tmp_path / "model1.onnx",
            "hotkey2": tmp_path / "model2.onnx",
        }

        batch = await orchestrator.evaluate_all(
            models=models,
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert batch.successful_count == 1
        assert batch.failed_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_all_empty_models(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
    ) -> None:
        """Empty models dict returns empty batch."""
        orchestrator = EvaluationOrchestrator(OrchestratorConfig())

        batch = await orchestrator.evaluate_all(
            models={},
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert batch.successful_count == 0
        assert batch.failed_count == 0
        assert batch.get_best() is None

    @pytest.mark.asyncio
    async def test_evaluate_all_timeout_error(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Timeout errors are captured in results."""
        mock_runner = MagicMock()
        mock_runner.run_inference.side_effect = InferenceTimeoutError(
            "Timed out after 300s",
            exit_code=None,
            logs="Still running...",
        )

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        models = {"hotkey1": tmp_path / "model1.onnx"}

        batch = await orchestrator.evaluate_all(
            models=models,
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert batch.successful_count == 0
        assert batch.failed_count == 1
        assert "timeout" in batch.failed_results[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_evaluate_all_respects_concurrency_limit(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Concurrency is limited by max_concurrent config."""
        import asyncio

        # Track concurrent execution count
        current_concurrent = 0
        max_observed_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency(*args, **kwargs):
            nonlocal current_concurrent, max_observed_concurrent
            async with lock:
                current_concurrent += 1
                max_observed_concurrent = max(max_observed_concurrent, current_concurrent)

            # Simulate some work
            await asyncio.sleep(0.01)

            async with lock:
                current_concurrent -= 1

            return InferenceResult(
                predictions=np.array([100000, 200000, 300000]),
                inference_time_ms=100.0,
            )

        mock_runner = MagicMock()
        mock_runner.run_inference = MagicMock(side_effect=lambda *a, **k: asyncio.get_event_loop().run_until_complete(track_concurrency(*a, **k)))

        config = OrchestratorConfig(max_concurrent=2)
        orchestrator = EvaluationOrchestrator(config)

        # Override _evaluate_single_model to track concurrency
        async def tracked_evaluate(*args, **kwargs):
            nonlocal current_concurrent, max_observed_concurrent
            async with lock:
                current_concurrent += 1
                max_observed_concurrent = max(max_observed_concurrent, current_concurrent)

            await asyncio.sleep(0.05)  # Simulate work

            async with lock:
                current_concurrent -= 1

            # Return a mock result
            from real_estate.evaluation.models import (
                EvaluationResult,
                PredictionMetrics,
            )
            return EvaluationResult(
                hotkey=args[0] if args else kwargs.get("hotkey"),
                predictions=sample_ground_truth,
                metrics=PredictionMetrics(
                    mae=1000, mape=0.05, rmse=1200, mdape=0.04,
                    accuracy={}, r2=0.95, n_samples=3
                ),
            )

        orchestrator._evaluate_single_model = tracked_evaluate

        # Create 5 models
        models = {f"hotkey{i}": tmp_path / f"model{i}.onnx" for i in range(5)}

        batch = await orchestrator.evaluate_all(
            models=models,
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        # All should complete
        assert batch.successful_count == 5
        # Concurrency should never exceed the limit
        assert max_observed_concurrent <= 2, f"Max concurrent was {max_observed_concurrent}, expected <= 2"

    @pytest.mark.asyncio
    async def test_evaluate_all_calculates_metrics(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Metrics are calculated for successful evaluations."""
        mock_runner = MagicMock()
        # Return predictions close to ground truth
        mock_runner.run_inference.return_value = InferenceResult(
            predictions=sample_ground_truth * 1.05,  # 5% off
            inference_time_ms=100.0,
        )

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        models = {"hotkey1": tmp_path / "model1.onnx"}

        batch = await orchestrator.evaluate_all(
            models=models,
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        result = batch.successful_results[0]
        assert result.metrics is not None
        assert result.metrics.mape == pytest.approx(0.05, rel=0.01)


class TestEvaluationOrchestratorEvaluateSingleModel:
    """Tests for EvaluationOrchestrator._evaluate_single_model method."""

    @pytest.fixture
    def sample_features(self) -> np.ndarray:
        """Sample input features."""
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    @pytest.fixture
    def sample_ground_truth(self) -> np.ndarray:
        """Sample ground truth values."""
        return np.array([100000, 200000])

    @pytest.mark.asyncio
    async def test_successful_evaluation(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Successful evaluation returns EvaluationResult with metrics."""
        mock_runner = MagicMock()
        mock_runner.run_inference.return_value = InferenceResult(
            predictions=np.array([95000, 210000]),
            inference_time_ms=150.0,
            container_logs="[SUCCESS]",
        )

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        result = await orchestrator._evaluate_single_model(
            hotkey="test_hotkey",
            model_path=tmp_path / "model.onnx",
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert result.success is True
        assert result.hotkey == "test_hotkey"
        assert result.predictions is not None
        assert result.metrics is not None
        assert result.inference_time_ms == 150.0

    @pytest.mark.asyncio
    async def test_failed_evaluation_captures_error(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Failed evaluation captures error without raising."""
        mock_runner = MagicMock()
        mock_runner.run_inference.side_effect = DockerExecutionError(
            "Container crashed",
            exit_code=137,
            logs="Out of memory",
        )

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        result = await orchestrator._evaluate_single_model(
            hotkey="test_hotkey",
            model_path=tmp_path / "model.onnx",
            features=sample_features,
            ground_truth=sample_ground_truth,
        )

        assert result.success is False
        assert result.hotkey == "test_hotkey"
        assert result.error is not None
        assert "Container crashed" in str(result.error)

    @pytest.mark.asyncio
    async def test_includes_model_hash_from_metadata(
        self,
        sample_features: np.ndarray,
        sample_ground_truth: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Model hash is included from metadata."""
        mock_runner = MagicMock()
        mock_runner.run_inference.return_value = InferenceResult(
            predictions=sample_ground_truth,
            inference_time_ms=100.0,
        )

        # Create mock metadata
        mock_metadata = MagicMock()
        mock_metadata.model_hash = "abc123def456"

        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        orchestrator._docker_runner = mock_runner

        result = await orchestrator._evaluate_single_model(
            hotkey="test_hotkey",
            model_path=tmp_path / "model.onnx",
            features=sample_features,
            ground_truth=sample_ground_truth,
            metadata=mock_metadata,
        )

        assert result.model_hash == "abc123def456"
