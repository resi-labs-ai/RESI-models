"""Unit tests for ValidationOrchestrator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from real_estate.incentives import NoValidModelsError
from real_estate.orchestration import ValidationOrchestrator, ValidationResult

from .conftest import (
    create_chain_metadata,
    create_dataset,
    create_duplicate_result,
    create_eval_batch,
    create_eval_result,
    create_weights,
    create_winner_result,
)


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
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.90),
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = create_duplicate_result()
        mock_detector.detect.return_value = mock_duplicates

        mock_winner = create_winner_result("A", 0.95, 100)
        mock_selector.select_winner.return_value = mock_winner

        mock_weights = create_weights({"A": 0.99, "B": 0.01})
        mock_distributor.calculate_weights.return_value = mock_weights

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

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
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        eval_batch = create_eval_batch([create_eval_result("A", success=False)])
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
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A"),
            "B": create_chain_metadata("B"),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = create_duplicate_result(frozenset({"A", "B"}), num_groups=1)
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
        dataset = create_dataset()
        model_paths = {
            "A": Path("/model_a.onnx"),
            "B": Path("/model_b.onnx"),
            "C": Path("/model_c.onnx"),
        }
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),  # Same score, copier
            create_eval_result("C", score=0.80),
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = create_duplicate_result(frozenset({"B"}), num_groups=1)
        mock_detector.detect.return_value = mock_duplicates

        mock_selector.select_winner.return_value = create_winner_result("A", 0.95, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 0.99, "C": 0.01})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        # Verify selector received only non-copier results
        call_args = mock_selector.select_winner.call_args
        valid_results = call_args[0][0]
        valid_hotkeys = [r.hotkey for r in valid_results]
        assert "B" not in valid_hotkeys
        assert "A" in valid_hotkeys
        assert "C" in valid_hotkeys

    @pytest.mark.asyncio
    async def test_pipeline_calls_all_dependencies(
        self,
        orchestrator: ValidationOrchestrator,
        mock_encoder: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Pipeline calls all dependencies."""
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        eval_batch = create_eval_batch([create_eval_result("A")])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_detector.detect.return_value = create_duplicate_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.9, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        await orchestrator.run(dataset, model_paths, chain_metadata)

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
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx"), "B": Path("/model_b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A"),
            "B": create_chain_metadata("B"),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", success=False),
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_detector.detect.return_value = create_duplicate_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.95, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        call_args = mock_selector.select_winner.call_args
        valid_results = call_args[0][0]
        assert len(valid_results) == 1
        assert valid_results[0].hotkey == "A"

    @pytest.mark.asyncio
    async def test_single_model_success(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Single successful model becomes winner with full weight."""
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A", 100)}

        eval_batch = create_eval_batch([create_eval_result("A", score=0.92)])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_detector.detect.return_value = create_duplicate_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.92, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        assert result.winner.winner_hotkey == "A"
        call_args = mock_selector.select_winner.call_args
        assert len(call_args[0][0]) == 1

    @pytest.mark.asyncio
    async def test_distributor_receives_correct_arguments(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Distributor receives winner hotkey, score, and cheater hotkeys."""
        dataset = create_dataset()
        model_paths = {
            "A": Path("/model_a.onnx"),
            "B": Path("/model_b.onnx"),
            "C": Path("/model_c.onnx"),
        }
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),  # Copier
            create_eval_result("C", score=0.80),
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = create_duplicate_result(frozenset({"B"}), num_groups=1)
        mock_detector.detect.return_value = mock_duplicates

        mock_selector.select_winner.return_value = create_winner_result("A", 0.95, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 0.99, "C": 0.01})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        mock_distributor.calculate_weights.assert_called_once()
        call_kwargs = mock_distributor.calculate_weights.call_args.kwargs
        assert call_kwargs["winner_hotkey"] == "A"
        assert call_kwargs["winner_score"] == 0.95
        assert call_kwargs["cheater_hotkeys"] == frozenset({"B"})
        assert len(call_kwargs["results"]) == 3

    @pytest.mark.asyncio
    async def test_mixed_failures_copiers_and_valid(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Mixed scenario: some fail, some are copiers, one valid winner."""
        dataset = create_dataset()
        model_paths = {
            "A": Path("/model_a.onnx"),
            "B": Path("/model_b.onnx"),
            "C": Path("/model_c.onnx"),
            "D": Path("/model_d.onnx"),
        }
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
            "D": create_chain_metadata("D", 400),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.90),  # Valid winner
            create_eval_result("B", success=False),  # Failed
            create_eval_result("C", score=0.90),  # Copier of A
            create_eval_result("D", score=0.85),  # Valid runner-up
        ])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_duplicates = create_duplicate_result(frozenset({"C"}), num_groups=1)
        mock_detector.detect.return_value = mock_duplicates

        mock_selector.select_winner.return_value = create_winner_result("A", 0.90, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 0.99, "D": 0.01})

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        # Winner selection receives only A and D (not B-failed, not C-copier)
        selector_call_args = mock_selector.select_winner.call_args
        valid_results = selector_call_args[0][0]
        valid_hotkeys = {r.hotkey for r in valid_results}
        assert valid_hotkeys == {"A", "D"}

        assert result.winner.winner_hotkey == "A"
        assert result.duplicate_result.copier_hotkeys == frozenset({"C"})

    @pytest.mark.asyncio
    async def test_encoded_features_passed_to_evaluator(
        self,
        orchestrator: ValidationOrchestrator,
        mock_encoder: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Encoded features are passed to evaluator."""
        dataset = create_dataset()
        model_paths = {"A": Path("/model_a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        expected_features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_encoder.encode.return_value = expected_features

        eval_batch = create_eval_batch([create_eval_result("A")])
        mock_evaluator.evaluate_all.return_value = eval_batch

        mock_detector.detect.return_value = create_duplicate_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.9, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        call_kwargs = mock_evaluator.evaluate_all.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["features"], expected_features)
