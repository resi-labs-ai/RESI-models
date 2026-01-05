"""
Integration tests for Docker-based model evaluation.

These tests require Docker to be running and will actually execute
containers. Mark with pytest.mark.integration and skip if Docker unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

from real_estate.evaluation.docker_runner import DockerConfig, DockerRunner
from real_estate.evaluation.errors import DockerExecutionError, InvalidPredictionError
from real_estate.evaluation.orchestrator import (
    EvaluationOrchestrator,
    create_orchestrator,
)
from real_estate.tests.fixtures.evaluation.conftest import (
    create_bad_model,
    create_test_model,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestDockerRunnerIntegration:
    """Integration tests for DockerRunner."""

    @pytest.fixture
    def runner(self) -> DockerRunner:
        """Create DockerRunner with short timeout for tests."""
        config = DockerConfig(
            timeout_seconds=120,  # 2 minutes should be enough
            memory_limit="1g",
        )
        return DockerRunner(config)

    @pytest.fixture
    def model_path(self, tmp_path: Path) -> Path:
        """Create a dummy model."""
        path = tmp_path / "model.onnx"
        create_test_model(n_features=10, output_path=path)
        return path

    @pytest.fixture
    def input_data(self) -> np.ndarray:
        """Create sample input data."""
        np.random.seed(123)
        return np.random.randn(50, 10).astype(np.float32)

    def test_successful_inference(
        self,
        runner: DockerRunner,
        model_path: Path,
        input_data: np.ndarray,
    ) -> None:
        """Run actual inference in Docker container."""
        result = runner.run_inference(model_path, input_data)

        assert result.predictions is not None
        assert len(result.predictions) == 50
        assert result.inference_time_ms > 0
        assert "[SUCCESS]" in result.container_logs or "[INFO]" in result.container_logs

        # Predictions should be valid (non-negative due to ReLU)
        assert np.all(result.predictions >= 0)
        assert not np.any(np.isnan(result.predictions))
        assert not np.any(np.isinf(result.predictions))

    def test_inference_with_single_sample(
        self,
        runner: DockerRunner,
        model_path: Path,
    ) -> None:
        """Inference works with single sample."""
        input_data = np.array([[1.0] * 10], dtype=np.float32)

        result = runner.run_inference(model_path, input_data)

        assert len(result.predictions) == 1

    def test_inference_with_large_batch(
        self,
        runner: DockerRunner,
        model_path: Path,
    ) -> None:
        """Inference works with larger batch."""
        np.random.seed(789)
        input_data = np.random.randn(500, 10).astype(np.float32)

        result = runner.run_inference(model_path, input_data)

        assert len(result.predictions) == 500

    def test_container_logs_contain_diagnostics(
        self,
        runner: DockerRunner,
        model_path: Path,
        input_data: np.ndarray,
    ) -> None:
        """Container logs contain useful diagnostic information."""
        result = runner.run_inference(model_path, input_data)

        # Check for expected log messages
        assert "[INFO]" in result.container_logs
        assert "Input loaded" in result.container_logs or "input" in result.container_logs.lower()

    def test_container_cleaned_up_after_inference(
        self,
        runner: DockerRunner,
        model_path: Path,
        input_data: np.ndarray,
    ) -> None:
        """Container is removed after successful inference."""
        import docker

        client = docker.from_env()

        # Get container IDs before inference
        containers_before = {c.id for c in client.containers.list(all=True)}

        # Run inference
        runner.run_inference(model_path, input_data)

        # Get container IDs after inference
        containers_after = {c.id for c in client.containers.list(all=True)}

        # No new containers should remain
        new_containers = containers_after - containers_before
        assert len(new_containers) == 0, f"Container not cleaned up: {new_containers}"

    def test_container_cleaned_up_after_failure(
        self,
        runner: DockerRunner,
        tmp_path: Path,
        input_data: np.ndarray,
    ) -> None:
        """Container is removed even after inference failure."""
        import docker

        client = docker.from_env()

        # Create an invalid model (empty file)
        bad_model = tmp_path / "bad_model.onnx"
        bad_model.write_bytes(b"not a valid onnx model")

        # Get container IDs before inference
        containers_before = {c.id for c in client.containers.list(all=True)}

        # Run inference (should fail)
        with pytest.raises(DockerExecutionError):
            runner.run_inference(bad_model, input_data)

        # Get container IDs after inference
        containers_after = {c.id for c in client.containers.list(all=True)}

        # No new containers should remain
        new_containers = containers_after - containers_before
        assert len(new_containers) == 0, f"Container not cleaned up after failure: {new_containers}"

    def test_nan_predictions_rejected(
        self,
        runner: DockerRunner,
        tmp_path: Path,
    ) -> None:
        """Model producing NaN predictions raises InvalidPredictionError."""
        # Create model that outputs NaN (0/0)
        nan_model = tmp_path / "nan_model.onnx"
        create_bad_model(nan_model)

        input_data = np.random.randn(10, 10).astype(np.float32)

        with pytest.raises(InvalidPredictionError, match="invalid predictions"):
            runner.run_inference(nan_model, input_data)


class TestOrchestratorIntegration:
    """Integration tests for EvaluationOrchestrator."""

    @pytest.fixture
    def orchestrator(self) -> EvaluationOrchestrator:
        """Create orchestrator with test config."""
        return create_orchestrator(
            max_concurrent=2,
            docker_memory="1g",
            docker_timeout=120,
        )

    @pytest.fixture
    def model_paths(self, tmp_path: Path) -> dict[str, Path]:
        """Create multiple test models with different weights."""
        models = {}
        for i, seed in enumerate([42, 123, 456]):
            path = tmp_path / f"model_{i}.onnx"
            create_test_model(n_features=10, output_path=path, seed=seed)
            models[f"hotkey_{i}"] = path
        return models

    @pytest.fixture
    def features(self) -> np.ndarray:
        """Sample features."""
        np.random.seed(999)
        return np.random.randn(100, 10).astype(np.float32)

    @pytest.fixture
    def ground_truth(self) -> np.ndarray:
        """Sample ground truth (realistic house prices)."""
        np.random.seed(888)
        return np.random.uniform(100000, 1000000, size=100).astype(np.float32)

    @pytest.mark.asyncio
    async def test_evaluate_single_model(
        self,
        orchestrator: EvaluationOrchestrator,
        model_paths: dict[str, Path],
        features: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """Evaluate a single model end-to-end."""
        single_model = {"hotkey_0": model_paths["hotkey_0"]}

        batch = await orchestrator.evaluate_all(
            models=single_model,
            features=features,
            ground_truth=ground_truth,
        )

        assert batch.successful_count == 1
        assert batch.failed_count == 0

        result = batch.successful_results[0]
        assert result.hotkey == "hotkey_0"
        assert result.predictions is not None
        assert result.metrics is not None
        assert result.metrics.mape >= 0  # MAPE is always non-negative

    @pytest.mark.asyncio
    async def test_evaluate_multiple_models(
        self,
        orchestrator: EvaluationOrchestrator,
        model_paths: dict[str, Path],
        features: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """Evaluate multiple models concurrently."""
        batch = await orchestrator.evaluate_all(
            models=model_paths,
            features=features,
            ground_truth=ground_truth,
        )

        assert batch.successful_count == 3
        assert batch.failed_count == 0
        assert batch.total_time_ms > 0

        # All models should have predictions and metrics
        for result in batch.successful_results:
            assert result.predictions is not None
            assert len(result.predictions) == 100
            assert result.metrics is not None

        # Should be able to rank and get best
        ranking = batch.get_ranking()
        assert len(ranking) == 3

        best = batch.get_best()
        assert best is not None
        assert best.hotkey in model_paths

    @pytest.mark.asyncio
    async def test_metrics_calculation(
        self,
        orchestrator: EvaluationOrchestrator,
        model_paths: dict[str, Path],
        features: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """Verify metrics are calculated correctly."""
        single_model = {"hotkey_0": model_paths["hotkey_0"]}

        batch = await orchestrator.evaluate_all(
            models=single_model,
            features=features,
            ground_truth=ground_truth,
        )

        result = batch.successful_results[0]
        metrics = result.metrics

        # Check all metrics are present
        assert metrics.mae > 0
        assert metrics.mape >= 0
        assert metrics.rmse > 0
        assert metrics.mdape >= 0
        assert metrics.n_samples == 100

        # Score should be derived from MAPE
        expected_score = max(0, 1 - metrics.mape)
        assert metrics.score == pytest.approx(expected_score, rel=0.01)


class TestEndToEndWorkflow:
    """Test complete evaluation workflow."""

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from model to ranked results."""
        # 1. Create models with different "quality" (different seeds = different weights)
        models = {}
        for i in range(3):
            np.random.seed(i * 100)
            path = tmp_path / f"miner_{i}.onnx"
            create_test_model(n_features=5, output_path=path)
            models[f"5F{i}abc..."] = path

        # 2. Create validation data
        np.random.seed(42)
        features = np.random.randn(50, 5).astype(np.float32)
        ground_truth = np.abs(np.random.randn(50).astype(np.float32)) * 200000 + 100000

        # 3. Run evaluation
        orchestrator = create_orchestrator(
            max_concurrent=2,
            docker_timeout=120,
        )

        batch = await orchestrator.evaluate_all(
            models=models,
            features=features,
            ground_truth=ground_truth,
        )

        # 4. Verify results
        assert batch.successful_count == 3
        assert batch.dataset_size == 50

        # 5. Get ranking
        ranking = batch.get_ranking()
        print("\n=== Evaluation Results ===")
        for hotkey, score in ranking:
            result = next(r for r in batch.successful_results if r.hotkey == hotkey)
            print(f"{hotkey}: score={score:.4f}, MAPE={result.metrics.mape:.2%}")

        # 6. Best model should have highest score
        best = batch.get_best()
        assert best.score == ranking[0][1]

        # 7. Export results
        results_dict = batch.to_dict()
        assert "ranking" in results_dict
        assert "results" in results_dict
        assert results_dict["successful_count"] == 3
