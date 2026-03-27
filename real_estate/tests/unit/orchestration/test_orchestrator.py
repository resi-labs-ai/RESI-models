"""Unit tests for ValidationOrchestrator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from real_estate.generalization_detector import GeneralizationConfig
from real_estate.incentives import NoValidModelsError
from real_estate.orchestration import ValidationOrchestrator, ValidationResult

from .conftest import (
    create_chain_metadata,
    create_dataset,
    create_duplicate_result,
    create_eval_batch,
    create_eval_result,
    create_generalization_result,
    create_inspection_result,
    create_weights,
    create_winner_result,
)


@pytest.fixture
def mock_encoder() -> MagicMock:
    encoder = MagicMock()
    encoder.encode.return_value = np.array([[1.0, 2.0, 3.0]] * 10)
    return encoder


@pytest.fixture
def mock_evaluator() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_detector() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_selector() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_distributor() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_inspector() -> MagicMock:
    inspector = MagicMock()
    inspector.inspect_all = AsyncMock(return_value=create_inspection_result())
    return inspector


@pytest.fixture
def mock_gen_detector() -> MagicMock:
    detector = MagicMock()
    detector.detect.return_value = create_generalization_result()
    return detector


@pytest.fixture
def gen_config() -> GeneralizationConfig:
    return GeneralizationConfig(num_numeric_features=3, lat_index=0, lon_index=1)


@pytest.fixture
def orchestrator(
    mock_encoder: MagicMock,
    mock_evaluator: AsyncMock,
    mock_detector: MagicMock,
    mock_selector: MagicMock,
    mock_distributor: MagicMock,
    mock_inspector: MagicMock,
    mock_gen_detector: MagicMock,
    gen_config: GeneralizationConfig,
) -> ValidationOrchestrator:
    return ValidationOrchestrator(
        encoder=mock_encoder,
        evaluator=mock_evaluator,
        detector=mock_detector,
        selector=mock_selector,
        distributor=mock_distributor,
        model_inspector=mock_inspector,
        generalization_detector=mock_gen_detector,
        generalization_config=gen_config,
    )


def _setup_default_mocks(
    mock_evaluator: AsyncMock,
    mock_detector: MagicMock,
    mock_selector: MagicMock,
    mock_distributor: MagicMock,
    *,
    eval_batch=None,
    perturbed_batch=None,
    spatial_batch=None,
    duplicates=None,
    winner=None,
    weights=None,
):
    """Wire up default mock return values for a successful pipeline run."""
    if eval_batch is None:
        eval_batch = create_eval_batch([create_eval_result("A", score=0.90)])
    if perturbed_batch is None:
        perturbed_batch = eval_batch
    if spatial_batch is None:
        spatial_batch = eval_batch
    mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, spatial_batch]
    mock_detector.detect.return_value = duplicates or create_duplicate_result()
    mock_selector.select_winner.return_value = winner or create_winner_result("A", 0.90, 100)
    mock_distributor.calculate_weights.return_value = weights or create_weights({"A": 1.0})


class TestValidationOrchestratorRun:
    """Tests for the full pipeline."""

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
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.90),
        ])
        mock_winner = create_winner_result("A", 0.95, 100)
        mock_weights = create_weights({"A": 0.99, "B": 0.01})

        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch, winner=mock_winner, weights=mock_weights,
        )

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        assert isinstance(result, ValidationResult)
        assert result.winner == mock_winner
        assert result.weights == mock_weights
        assert result.eval_batch == eval_batch
        assert result.inspection_result is not None
        assert result.generalization_result is not None

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_no_valid_models_error(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """All model evaluations failing raises NoValidModelsError."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        eval_batch = create_eval_batch([create_eval_result("A", success=False)])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
        )

        with pytest.raises(NoValidModelsError, match="All model evaluations failed"):
            await orchestrator.run(dataset, model_paths, chain_metadata)

    @pytest.mark.asyncio
    async def test_all_valid_models_are_copiers_raises_error(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """All valid models being copiers raises NoValidModelsError."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A"),
            "B": create_chain_metadata("B"),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            duplicates=create_duplicate_result(frozenset({"A", "B"}), num_groups=1),
        )

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
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx"), "C": Path("/c.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),
            create_eval_result("C", score=0.80),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            duplicates=create_duplicate_result(frozenset({"B"}), num_groups=1),
            winner=create_winner_result("A", 0.95, 100),
            weights=create_weights({"A": 0.99, "C": 0.01}),
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        valid_results = mock_selector.select_winner.call_args[0][0]
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
        mock_inspector: MagicMock,
        mock_detector: MagicMock,
        mock_gen_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Pipeline calls all dependencies."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        mock_inspector.inspect_all.assert_called_once()
        mock_encoder.encode.assert_called_once()
        assert mock_evaluator.evaluate_all.call_count == 3  # original + perturbed + spatial
        mock_detector.detect.assert_called_once()
        mock_gen_detector.detect.assert_called_once()
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
        """Failed evaluations are excluded from valid results."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A"),
            "B": create_chain_metadata("B"),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", success=False),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            winner=create_winner_result("A", 0.95, 100),
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        valid_results = mock_selector.select_winner.call_args[0][0]
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
        """Single successful model becomes winner."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A", 100)}

        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=create_eval_batch([create_eval_result("A", score=0.92)]),
            winner=create_winner_result("A", 0.92, 100),
        )

        result = await orchestrator.run(dataset, model_paths, chain_metadata)
        assert result.winner.winner_hotkey == "A"

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
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx"), "C": Path("/c.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.95),
            create_eval_result("C", score=0.80),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            duplicates=create_duplicate_result(frozenset({"B"}), num_groups=1),
            winner=create_winner_result("A", 0.95, 100),
            weights=create_weights({"A": 0.99, "C": 0.01}),
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        call_kwargs = mock_distributor.calculate_weights.call_args.kwargs
        assert call_kwargs["winner_hotkey"] == "A"
        assert call_kwargs["winner_score"] == 0.95
        assert "B" in call_kwargs["cheater_hotkeys"]
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
            "A": Path("/a.onnx"), "B": Path("/b.onnx"),
            "C": Path("/c.onnx"), "D": Path("/d.onnx"),
        }
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
            "D": create_chain_metadata("D", 400),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.90),
            create_eval_result("B", success=False),
            create_eval_result("C", score=0.90),
            create_eval_result("D", score=0.85),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            duplicates=create_duplicate_result(frozenset({"C"}), num_groups=1),
            winner=create_winner_result("A", 0.90, 100),
            weights=create_weights({"A": 0.99, "D": 0.01}),
        )

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        valid_results = mock_selector.select_winner.call_args[0][0]
        valid_hotkeys = {r.hotkey for r in valid_results}
        assert valid_hotkeys == {"A", "D"}
        assert result.winner.winner_hotkey == "A"

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
        """Encoded features are passed to first evaluator call."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        expected_features = np.array([[1.0, 2.0, 3.0]] * 10)
        mock_encoder.encode.return_value = expected_features

        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        first_call_kwargs = mock_evaluator.evaluate_all.call_args_list[0].kwargs
        np.testing.assert_array_equal(first_call_kwargs["features"], expected_features)


class TestInspectionRejection:
    """Tests for model inspection rejection in the pipeline."""

    @pytest.mark.asyncio
    async def test_inspection_rejects_models_before_evaluation(
        self,
        orchestrator: ValidationOrchestrator,
        mock_inspector: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Rejected models are removed from evaluation."""
        dataset = create_dataset()
        model_paths = {"good": Path("/good.onnx"), "memorizer": Path("/memorizer.onnx")}
        chain_metadata = {
            "good": create_chain_metadata("good"),
            "memorizer": create_chain_metadata("memorizer"),
        }

        mock_inspector.inspect_all.return_value = create_inspection_result(
            rejected_hotkeys=frozenset({"memorizer"})
        )

        eval_batch = create_eval_batch([create_eval_result("good", score=0.90)])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            winner=create_winner_result("good", 0.90, 100),
            weights=create_weights({"good": 1.0}),
        )

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        # Evaluator should only receive the good model
        first_eval_call = mock_evaluator.evaluate_all.call_args_list[0]
        models_evaluated = first_eval_call.kwargs["models"]
        assert "memorizer" not in models_evaluated
        assert "good" in models_evaluated

        assert result.inspection_result.is_rejected("memorizer")

    @pytest.mark.asyncio
    async def test_rejected_merged_into_cheaters(
        self,
        orchestrator: ValidationOrchestrator,
        mock_inspector: MagicMock,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
    ) -> None:
        """Rejected hotkeys from inspection are included in cheater_hotkeys."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx"), "C": Path("/c.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        mock_inspector.inspect_all.return_value = create_inspection_result(
            rejected_hotkeys=frozenset({"C"})
        )

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.95),
            create_eval_result("B", score=0.90),
        ])
        _setup_default_mocks(
            mock_evaluator, mock_detector, mock_selector, mock_distributor,
            eval_batch=eval_batch,
            duplicates=create_duplicate_result(frozenset({"B"}), num_groups=1),
            winner=create_winner_result("A", 0.95, 100),
        )

        await orchestrator.run(dataset, model_paths, chain_metadata)

        cheaters = mock_distributor.calculate_weights.call_args.kwargs["cheater_hotkeys"]
        assert "B" in cheaters
        assert "C" in cheaters

    @pytest.mark.asyncio
    async def test_all_rejected_raises_no_valid_models(
        self,
        orchestrator: ValidationOrchestrator,
        mock_inspector: MagicMock,
    ) -> None:
        """All models rejected by inspection raises NoValidModelsError."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A"),
            "B": create_chain_metadata("B"),
        }

        mock_inspector.inspect_all.return_value = create_inspection_result(
            rejected_hotkeys=frozenset({"A", "B"})
        )

        with pytest.raises(NoValidModelsError, match="All models rejected"):
            await orchestrator.run(dataset, model_paths, chain_metadata)


class TestGeneralizationDetection:
    """Tests for generalization detection in the pipeline."""

    @pytest.mark.asyncio
    async def test_memorizers_filtered_from_winner_selection(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
        mock_gen_detector: MagicMock,
    ) -> None:
        """Memorizers are filtered from winner selection."""
        dataset = create_dataset()
        model_paths = {"good": Path("/good.onnx"), "memorizer": Path("/memorizer.onnx")}
        chain_metadata = {
            "good": create_chain_metadata("good"),
            "memorizer": create_chain_metadata("memorizer"),
        }

        eval_batch = create_eval_batch([
            create_eval_result("good", score=0.80),
            create_eval_result("memorizer", score=0.95),
        ])
        perturbed_batch = create_eval_batch([
            create_eval_result("good", score=0.75),
            create_eval_result("memorizer", score=0.10),
        ])
        mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, eval_batch]
        mock_detector.detect.return_value = create_duplicate_result()
        mock_gen_detector.detect.return_value = create_generalization_result(
            memorizer_hotkeys=frozenset({"memorizer"})
        )
        mock_selector.select_winner.return_value = create_winner_result("good", 0.80, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"good": 1.0})

        result = await orchestrator.run(dataset, model_paths, chain_metadata)

        assert mock_evaluator.evaluate_all.call_count == 3

        selector_args = mock_selector.select_winner.call_args[0][0]
        assert len(selector_args) == 1
        assert selector_args[0].hotkey == "good"

        cheaters = mock_distributor.calculate_weights.call_args.kwargs["cheater_hotkeys"]
        assert "memorizer" in cheaters
        assert result.generalization_result is not None

    @pytest.mark.asyncio
    async def test_memorizers_and_copiers_combined(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
        mock_gen_detector: MagicMock,
    ) -> None:
        """Memorizers and copiers are combined into cheaters set."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx"), "C": Path("/c.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.90),
            create_eval_result("B", score=0.90),
            create_eval_result("C", score=0.85),
        ])
        perturbed_batch = create_eval_batch([
            create_eval_result("A", score=0.85),
            create_eval_result("B", score=0.85),
            create_eval_result("C", score=0.10),
        ])
        mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, eval_batch]
        mock_detector.detect.return_value = create_duplicate_result(
            frozenset({"B"}), num_groups=1
        )
        mock_gen_detector.detect.return_value = create_generalization_result(
            memorizer_hotkeys=frozenset({"C"})
        )
        mock_selector.select_winner.return_value = create_winner_result("A", 0.90, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        cheaters = mock_distributor.calculate_weights.call_args.kwargs["cheater_hotkeys"]
        assert "B" in cheaters
        assert "C" in cheaters
        assert "A" not in cheaters

    @pytest.mark.asyncio
    async def test_gen_detector_receives_both_result_lists(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
        mock_gen_detector: MagicMock,
    ) -> None:
        """Generalization detector receives original and perturbed results."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx")}
        chain_metadata = {"A": create_chain_metadata("A")}

        eval_batch = create_eval_batch([create_eval_result("A", score=0.90)])
        perturbed_batch = create_eval_batch([create_eval_result("A", score=0.85)])
        spatial_batch = create_eval_batch([create_eval_result("A", score=0.89)])
        mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, spatial_batch]
        mock_detector.detect.return_value = create_duplicate_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.90, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 1.0})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        gen_call = mock_gen_detector.detect.call_args
        assert gen_call[0][0] == eval_batch.results
        assert gen_call[0][1] == perturbed_batch.results
        assert gen_call[0][2] == spatial_batch.results


class TestPerturbedEvalFiltering:
    """Tests for filtering failed models from perturbed evaluation."""

    @pytest.mark.asyncio
    async def test_failed_models_excluded_from_perturbed_eval(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
        mock_gen_detector: MagicMock,
    ) -> None:
        """Models that failed original eval are not passed to perturbed eval."""
        dataset = create_dataset()
        model_paths = {
            "A": Path("/a.onnx"),
            "B": Path("/b.onnx"),
            "C": Path("/c.onnx"),
        }
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
            "C": create_chain_metadata("C", 300),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.90),
            create_eval_result("B", success=False),  # fails original
            create_eval_result("C", score=0.85),
        ])
        perturbed_batch = create_eval_batch([
            create_eval_result("A", score=0.85),
            create_eval_result("C", score=0.80),
        ])
        mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, perturbed_batch]
        mock_detector.detect.return_value = create_duplicate_result()
        mock_gen_detector.detect.return_value = create_generalization_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.90, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 0.99, "C": 0.01})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        # Second evaluate_all call (perturbed) should only have successful models
        perturbed_call = mock_evaluator.evaluate_all.call_args_list[1]
        perturbed_models = perturbed_call.kwargs["models"]
        assert "A" in perturbed_models
        assert "C" in perturbed_models
        assert "B" not in perturbed_models

    @pytest.mark.asyncio
    async def test_all_models_succeed_perturbed_gets_all(
        self,
        orchestrator: ValidationOrchestrator,
        mock_evaluator: AsyncMock,
        mock_detector: MagicMock,
        mock_selector: MagicMock,
        mock_distributor: MagicMock,
        mock_gen_detector: MagicMock,
    ) -> None:
        """When all models succeed, perturbed eval receives all models."""
        dataset = create_dataset()
        model_paths = {"A": Path("/a.onnx"), "B": Path("/b.onnx")}
        chain_metadata = {
            "A": create_chain_metadata("A", 100),
            "B": create_chain_metadata("B", 200),
        }

        eval_batch = create_eval_batch([
            create_eval_result("A", score=0.90),
            create_eval_result("B", score=0.85),
        ])
        perturbed_batch = create_eval_batch([
            create_eval_result("A", score=0.85),
            create_eval_result("B", score=0.80),
        ])
        mock_evaluator.evaluate_all.side_effect = [eval_batch, perturbed_batch, eval_batch]
        mock_detector.detect.return_value = create_duplicate_result()
        mock_gen_detector.detect.return_value = create_generalization_result()
        mock_selector.select_winner.return_value = create_winner_result("A", 0.90, 100)
        mock_distributor.calculate_weights.return_value = create_weights({"A": 0.99, "B": 0.01})

        await orchestrator.run(dataset, model_paths, chain_metadata)

        perturbed_call = mock_evaluator.evaluate_all.call_args_list[1]
        perturbed_models = perturbed_call.kwargs["models"]
        assert len(perturbed_models) == 2


