"""Unit tests for ValidationOrchestrator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from real_estate.evaluation.models import EvaluationBatch, EvaluationResult, PredictionMetrics
from real_estate.incentives import NoValidModelsError
from real_estate.orchestration import ValidationOrchestrator, ValidationResult


def _create_mock_eval_result(
    hotkey: str,
    score: float = 0.9,
    success: bool = True,
    predictions: np.ndarray | None = None,
) -> EvaluationResult:
    """Create a mock EvaluationResult for testing."""
    if predictions is None:
        predictions = np.array([100.0, 200.0, 300.0])

    mape = 1.0 - score  # score = 1 - mape
    metrics = PredictionMetrics(
        mape=mape,
        mae=10000.0,
        rmse=15000.0,
        mdape=mape,
        accuracy={0.05: 0.3, 0.10: 0.6, 0.15: 0.8},
        r2=score,
        n_samples=10,
    ) if success else None

    return EvaluationResult(
        hotkey=hotkey,
        predictions=predictions if success else None,
        metrics=metrics,
        error=None if success else Exception("Eval failed"),
        inference_time_ms=100.0,
        model_hash="abc123",
    )


def _create_mock_metadata(hotkey: str, block_number: int = 1000) -> MagicMock:
    """Create mock ChainModelMetadata."""
    metadata = MagicMock()
    metadata.hotkey = hotkey
    metadata.block_number = block_number
    metadata.model_hash = "abc123"
    return metadata


def _create_mock_dataset(size: int = 10) -> MagicMock:
    """Create mock ValidationDataset."""
    dataset = MagicMock()
    dataset.properties = [MagicMock() for _ in range(size)]
    dataset.ground_truth = [100000.0 + i * 10000 for i in range(size)]
    dataset.__len__ = MagicMock(return_value=size)
    return dataset


class TestValidationOrchestratorRun:
    """Tests for ValidationOrchestrator.run method."""

    @pytest.fixture
    def mock_encoder(self) -> MagicMock:
        """Create mock feature encoder."""
        encoder = MagicMock()
        encoder.encode.return_value = np.array([[1.0, 2.0, 3.0]] * 10)
        return encoder

    @pytest.fixture
    def mock_evaluator(self) -> AsyncMock:
        """Create mock evaluation orchestrator."""
        evaluator = AsyncMock()
        return evaluator

    @pytest.fixture
    def mock_detector(self) -> MagicMock:
        """Create mock duplicate detector."""
        detector = MagicMock()
        return detector

    @pytest.fixture
    def mock_selector(self) -> MagicMock:
        """Create mock winner selector."""
        selector = MagicMock()
        return selector

    @pytest.fixture
    def mock_distributor(self) -> MagicMock:
        """Create mock incentive distributor."""
        distributor = MagicMock()
        return distributor

    @pytest.fixture
    def orchestrator(
        self,
        mock_encoder: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> ValidationOrchestrator:
        """Create orchestrator with mocked dependencies."""
        return ValidationOrchestrator(
            encoder=mock_encoder,
            evaluator=mock_evaluator,
            detector=mock_detector,
            selector=mock_selector,
            distributor=mock_distributor,
        )

    @pytest.mark.asyncio
    async def test_successful_pipeline_returns_validation_result(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Successful pipeline returns ValidationResult with all components."""
        # Setup
        dataset = _create_mock_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
        }

        eval_results = [
            _create_mock_eval_result("A", score=0.95),
            _create_mock_eval_result("B", score=0.90),
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()
        mock_duplicates.groups = []
        mock_detector.detect.return_value = mock_duplicates

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.95
        mock_winner.winner_block = 100
        mock_selector.select_winner.return_value = mock_winner

        mock_weights = MagicMock()
        mock_weights.weights = {"A": 0.99, "B": 0.01}
        mock_weights.total = 1.0
        mock_weights.get_weight.return_value = 0.99
        mock_distributor.calculate_weights.return_value = mock_weights

        # Execute
        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        # Verify
        assert isinstance(result, ValidationResult)
        assert result.winner == mock_winner
        assert result.weights == mock_weights
        assert result.eval_batch == eval_batch
        assert result.duplicate_result == mock_duplicates

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_no_valid_models_error(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
    ) -> None:
        """All model evaluations failing raises NoValidModelsError."""
        dataset = _create_mock_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": _create_mock_metadata("A")}

        # All results are failures
        eval_results = [_create_mock_eval_result("A", success=False)]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        with pytest.raises(NoValidModelsError, match="All model evaluations failed"):
            await orchestrator.run(dataset, model_paths, chain_metadata)

    @pytest.mark.asyncio
    async def test_all_valid_models_are_copiers_raises_error(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
    ) -> None:
        """All valid models being copiers raises NoValidModelsError."""
        dataset = _create_mock_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": _create_mock_metadata("A"),
            "B": _create_mock_metadata("B"),
        }

        eval_results = [
            _create_mock_eval_result("A", score=0.95),
            _create_mock_eval_result("B", score=0.95),
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        # Both are copiers
        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset({"A", "B"})
        mock_duplicates.groups = [MagicMock()]
        mock_detector.detect.return_value = mock_duplicates

        with pytest.raises(NoValidModelsError, match="No valid models"):
            await orchestrator.run(dataset, model_paths, chain_metadata)

    @pytest.mark.asyncio
    async def test_copiers_are_filtered_before_winner_selection(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Copiers are filtered out before winner selection."""
        dataset = _create_mock_dataset()
        model_paths = {
            "A": Path("/model_a.onnx"),
            "B": Path("/model_b.onnx"),
            "C": Path("/model_c.onnx"),
        }
        chain_metadata = {
            "A": _create_mock_metadata("A", 100),
            "B": _create_mock_metadata("B", 200),
            "C": _create_mock_metadata("C", 300),
        }

        eval_results = [
            _create_mock_eval_result("A", score=0.95),
            _create_mock_eval_result("B", score=0.95),  # Same score, copier
            _create_mock_eval_result("C", score=0.80),
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        # B is a copier
        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset({"B"})
        mock_duplicates.groups = [MagicMock()]
        mock_detector.detect.return_value = mock_duplicates

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.95
        mock_winner.winner_block = 100
        mock_selector.select_winner.return_value = mock_winner

        mock_weights = MagicMock()
        mock_weights.weights = {"A": 0.99, "C": 0.01}
        mock_weights.total = 1.0
        mock_weights.get_weight.return_value = 0.99
        mock_distributor.calculate_weights.return_value = mock_weights

        await orchestrator.run(dataset, model_paths, chain_metadata)

        # Verify selector received only non-copier results
        call_args = mock_selector.select_winner.call_args
        valid_results = call_args[0][0]
        valid_hotkeys = [r.hotkey for r in valid_results]
        assert "B" not in valid_hotkeys
        assert "A" in valid_hotkeys
        assert "C" in valid_hotkeys

    @pytest.mark.asyncio
    async def test_pipeline_calls_dependencies_in_order(
        self,
        orchestrator: ValidationOrchestrator,
        mock_encoder: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Pipeline calls dependencies in correct order."""
        dataset = _create_mock_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": _create_mock_metadata("A")}

        eval_results = [_create_mock_eval_result("A")]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()
        mock_duplicates.groups = []
        mock_detector.detect.return_value = mock_duplicates

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.9
        mock_winner.winner_block = 100
        mock_selector.select_winner.return_value = mock_winner

        mock_weights = MagicMock()
        mock_weights.weights = {"A": 1.0}
        mock_weights.total = 1.0
        mock_weights.get_weight.return_value = 1.0
        mock_distributor.calculate_weights.return_value = mock_weights

        await orchestrator.run(dataset, model_paths, chain_metadata)

        # Verify all dependencies were called
        mock_encoder.encode.assert_called_once()
        mock_evaluator.evaluate_all.assert_called_once()
        mock_detector.detect.assert_called_once()
        mock_selector.select_winner.assert_called_once()
        mock_distributor.calculate_weights.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_evaluations_excluded_from_valid_results(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Failed evaluations are excluded from valid results for winner selection."""
        dataset = _create_mock_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": _create_mock_metadata("A"),
            "B": _create_mock_metadata("B"),
        }

        eval_results = [
            _create_mock_eval_result("A", score=0.95),
            _create_mock_eval_result("B", success=False),  # Failed
        ]
        eval_batch = EvaluationBatch(results=eval_results, dataset_size=10)
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = MagicMock()
        mock_duplicates.copier_hotkeys = frozenset()
        mock_duplicates.groups = []
        mock_detector.detect.return_value = mock_duplicates

        mock_winner = MagicMock()
        mock_winner.winner_hotkey = "A"
        mock_winner.winner_score = 0.95
        mock_winner.winner_block = 100
        mock_selector.select_winner.return_value = mock_winner

        mock_weights = MagicMock()
        mock_weights.weights = {"A": 1.0}
        mock_weights.total = 1.0
        mock_weights.get_weight.return_value = 1.0
        mock_distributor.calculate_weights.return_value = mock_weights

        await orchestrator.run(dataset, model_paths, chain_metadata)

        # Verify selector only received successful result
        call_args = mock_selector.select_winner.call_args
        valid_results = call_args[0][0]
        assert len(valid_results) == 1
        assert valid_results[0].hotkey == "A"
