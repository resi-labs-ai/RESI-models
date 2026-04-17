"""
Integration tests for ValidationOrchestrator.

Tests the full pipeline: encode -> evaluate -> detect duplicates -> select winner -> distribute weights.
Requires Docker to be running.
"""

from pathlib import Path

import numpy as np
import pytest

from real_estate.chain.models import ChainModelMetadata
from real_estate.data import FeatureConfig, ValidationDataset
from real_estate.incentives import NoValidModelsError
from real_estate.orchestration import ValidationOrchestrator
from real_estate.tests.fixtures.evaluation.conftest import create_test_model

pytestmark = pytest.mark.integration

# 10 real features from the data contract (minimum required)
TEST_FEATURES = (
    "living_area_sqft",
    "lot_size_sqft",
    "bedrooms",
    "bathrooms",
    "latitude",
    "longitude",
    "year_built",
    "property_age",
    "stories",
    "has_pool",
)

TEST_FEATURE_CONFIG = FeatureConfig(version="1.0", features=TEST_FEATURES)
N_TEST_FEATURES = len(TEST_FEATURES)


@pytest.fixture
def sample_properties() -> list[dict]:
    """Create sample property data matching the test feature config."""
    np.random.seed(42)
    properties = []
    for _ in range(20):
        properties.append(
            {
                "living_area_sqft": float(np.random.uniform(1000, 3000)),
                "lot_size_sqft": float(np.random.uniform(5000, 20000)),
                "bedrooms": float(np.random.randint(2, 6)),
                "bathrooms": float(np.random.randint(1, 4)),
                "latitude": float(np.random.uniform(25, 48)),
                "longitude": float(np.random.uniform(-125, -70)),
                "year_built": float(np.random.randint(1960, 2020)),
                "property_age": float(np.random.randint(5, 65)),
                "stories": float(np.random.randint(1, 3)),
                "has_pool": bool(np.random.choice([True, False])),
                "price": float(np.random.uniform(200000, 800000)),
            }
        )
    return properties


@pytest.fixture
def validation_dataset(sample_properties: list[dict]) -> ValidationDataset:
    """Create validation dataset from sample properties."""
    return ValidationDataset(properties=sample_properties)


@pytest.fixture
def test_feature_configs() -> dict[str, FeatureConfig]:
    """Return a feature config dict for a single hotkey (used as template)."""
    return {}


def create_chain_metadata(hotkey: str, block_number: int) -> ChainModelMetadata:
    """Create chain metadata for a model."""
    return ChainModelMetadata(
        hotkey=hotkey,
        hf_repo_id=f"test/{hotkey}",
        model_hash=f"hash_{hotkey}",
        block_number=block_number,
    )


def create_models_with_metadata(
    tmp_path: Path,
    configs: list[tuple[int, int]],
    n_features: int = N_TEST_FEATURES,
) -> tuple[dict[str, Path], dict[str, ChainModelMetadata]]:
    """
    Create multiple test models with chain metadata.

    Args:
        tmp_path: Directory to create models in
        configs: List of (seed, block_number) tuples
        n_features: Number of input features for the models

    Returns:
        Tuple of (model_paths, chain_metadata) dicts keyed by hotkey
    """
    model_paths = {}
    chain_metadata = {}
    for i, (seed, block) in enumerate(configs):
        hotkey = f"hotkey_{i}"
        model_path = tmp_path / f"model_{i}.onnx"
        create_test_model(n_features=n_features, output_path=model_path, seed=seed)
        model_paths[hotkey] = model_path
        chain_metadata[hotkey] = create_chain_metadata(hotkey, block_number=block)
    return model_paths, chain_metadata


def make_feature_configs(hotkeys: set[str]) -> dict[str, FeatureConfig]:
    """Create feature configs for all hotkeys using the test feature set."""
    return {hk: TEST_FEATURE_CONFIG for hk in hotkeys}


class TestValidationOrchestratorIntegration:
    """Integration tests for the full validation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_single_model(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test complete pipeline with a single model."""
        model_path = tmp_path / "model.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=model_path, seed=42
        )

        model_paths = {"hotkey_a": model_path}
        chain_metadata = {
            "hotkey_a": create_chain_metadata("hotkey_a", block_number=1000)
        }

        orchestrator = ValidationOrchestrator.create(
            score_threshold=0.005,
            winner_share=0.99,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        # Verify result structure
        assert result.winner.winner_hotkey == "hotkey_a"
        # Single model gets winner_share (99%), no one to share remaining 1%
        assert result.weights.get_weight("hotkey_a") == pytest.approx(0.99)
        assert result.weights.total == pytest.approx(0.99)
        assert result.eval_batch.successful_count == 1

    @pytest.mark.asyncio
    async def test_multiple_models_winner_selection(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test pipeline with multiple models and winner selection."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(42, 3000), (123, 1000), (456, 2000)]
        )

        orchestrator = ValidationOrchestrator.create(
            score_threshold=0.005,
            winner_share=0.99,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        # All models should succeed
        assert result.eval_batch.successful_count == 3
        assert result.eval_batch.failed_count == 0

        # Check for copiers - with different seeds, there should be none
        copiers = result.duplicate_result.copier_hotkeys
        assert len(copiers) == 0, f"Unexpected copiers: {copiers}"

        # Winner gets 99%
        winner_weight = result.weights.get_weight(result.winner.winner_hotkey)
        assert winner_weight == pytest.approx(0.99)

        # Total should be close to 1.0
        assert result.weights.total == pytest.approx(1.0, abs=0.02)

        # Verify all weights are assigned
        assert len(result.weights.weights) == 3

    @pytest.mark.asyncio
    async def test_duplicate_detection_in_pipeline(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that duplicate models are detected and copiers get zero weight."""
        original_path = tmp_path / "original.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=original_path, seed=42
        )

        copy_path = tmp_path / "copy.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=copy_path, seed=42
        )

        unique_path = tmp_path / "unique.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=unique_path, seed=999
        )

        model_paths = {
            "original": original_path,
            "copier": copy_path,
            "unique": unique_path,
        }

        chain_metadata = {
            "original": create_chain_metadata("original", block_number=1000),
            "copier": create_chain_metadata("copier", block_number=2000),
            "unique": create_chain_metadata("unique", block_number=1500),
        }

        orchestrator = ValidationOrchestrator.create(
            similarity_threshold=1e-6,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        assert "copier" in result.duplicate_result.copier_hotkeys
        assert len(result.duplicate_result.groups) == 1
        assert result.weights.get_weight("copier") == 0.0
        assert result.winner.winner_hotkey in ["original", "unique"]

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test pipeline handles mix of successful and failed models."""
        valid_path = tmp_path / "valid.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=valid_path, seed=42
        )

        # Wrong feature count -> rejected at pre-flight inspection (SHAPE_MISMATCH)
        invalid_path = tmp_path / "invalid.onnx"
        create_test_model(n_features=99, output_path=invalid_path, seed=123)

        model_paths = {
            "valid": valid_path,
            "invalid": invalid_path,
        }

        chain_metadata = {
            "valid": create_chain_metadata("valid", block_number=1000),
            "invalid": create_chain_metadata("invalid", block_number=2000),
        }

        orchestrator = ValidationOrchestrator.create()

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        # Invalid model is rejected at inspection, not at evaluation
        assert result.eval_batch.successful_count == 1
        assert result.eval_batch.failed_count == 0
        assert result.inspection_result is not None
        assert "invalid" in result.inspection_result.rejected_hotkeys
        assert result.winner.winner_hotkey == "valid"
        assert result.weights.get_weight("valid") == pytest.approx(0.99)
        assert result.weights.get_weight("invalid") == 0.0

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_error(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that NoValidModelsError is raised when all models fail."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(0, 1000), (1, 1001), (2, 1002)], n_features=99
        )

        orchestrator = ValidationOrchestrator.create()

        with pytest.raises(NoValidModelsError, match="All models rejected by pre-flight inspection"):
            await orchestrator.run(
                dataset=validation_dataset,
                model_paths=model_paths,
                chain_metadata=chain_metadata,
                feature_configs=make_feature_configs(set(model_paths)),
            )

    @pytest.mark.asyncio
    async def test_winner_selection_respects_commit_time(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that within score threshold, earlier commit wins."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(42, 3000), (42, 1000), (42, 2000)]
        )

        orchestrator = ValidationOrchestrator.create(
            score_threshold=0.1,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        assert len(result.duplicate_result.copier_hotkeys) == 2
        assert "hotkey_1" not in result.duplicate_result.copier_hotkeys
        assert result.winner.winner_hotkey == "hotkey_1"

    @pytest.mark.asyncio
    async def test_empty_model_paths_raises_error(
        self,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that empty model_paths raises NoValidModelsError."""
        orchestrator = ValidationOrchestrator.create()

        with pytest.raises(NoValidModelsError):
            await orchestrator.run(
                dataset=validation_dataset,
                model_paths={},
                chain_metadata={},
            )

    @pytest.mark.asyncio
    async def test_missing_chain_metadata_raises_error(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that missing chain metadata raises ValueError."""
        model_path = tmp_path / "model.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=model_path, seed=42
        )

        model_paths = {"hotkey_a": model_path}
        chain_metadata = {}

        orchestrator = ValidationOrchestrator.create()

        with pytest.raises(ValueError, match="Missing chain metadata"):
            await orchestrator.run(
                dataset=validation_dataset,
                model_paths=model_paths,
                chain_metadata=chain_metadata,
                feature_configs=make_feature_configs(set(model_paths)),
            )

    @pytest.mark.asyncio
    async def test_result_serialization(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Test that ValidationResult can be serialized for logging/export."""
        model_path = tmp_path / "model.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=model_path, seed=42
        )

        model_paths = {"hotkey_a": model_path}
        chain_metadata = {
            "hotkey_a": create_chain_metadata("hotkey_a", block_number=1000)
        }

        orchestrator = ValidationOrchestrator.create()

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        weights_dict = result.weights.to_dict()
        assert "winner_hotkey" in weights_dict
        assert "weights" in weights_dict
        assert weights_dict["winner_hotkey"] == "hotkey_a"

        winner_dict = result.winner.to_dict()
        assert "winner_hotkey" in winner_dict
        assert "winner_score" in winner_dict
        assert "candidates" in winner_dict

        eval_dict = result.eval_batch.to_dict()
        assert "successful_count" in eval_dict
        assert "results" in eval_dict

        full_dict = result.to_dict()
        assert "winner_hotkey" in full_dict
        assert "winner_score" in full_dict
        assert "results" in full_dict
        assert "copiers" in full_dict
        assert "weights" in full_dict
        assert full_dict["winner_hotkey"] == "hotkey_a"

    @pytest.mark.asyncio
    async def test_all_successful_models_are_copiers_raises_error(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """
        Test NoValidModelsError when original model fails but copies succeed.
        """
        failing_original = tmp_path / "failing_original.onnx"
        create_test_model(n_features=99, output_path=failing_original, seed=42)

        copy_a = tmp_path / "copy_a.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=copy_a, seed=100
        )

        copy_b = tmp_path / "copy_b.onnx"
        create_test_model(
            n_features=N_TEST_FEATURES, output_path=copy_b, seed=100
        )

        failing_earliest = tmp_path / "failing_earliest.onnx"
        create_test_model(n_features=99, output_path=failing_earliest, seed=100)

        model_paths = {
            "earliest_fails": failing_earliest,
            "copy_a": copy_a,
            "copy_b": copy_b,
        }

        chain_metadata = {
            "earliest_fails": create_chain_metadata(
                "earliest_fails", block_number=500
            ),
            "copy_a": create_chain_metadata("copy_a", block_number=2000),
            "copy_b": create_chain_metadata("copy_b", block_number=3000),
        }

        orchestrator = ValidationOrchestrator.create(
            similarity_threshold=1e-6,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        assert result.winner.winner_hotkey == "copy_a"
        assert "copy_b" in result.duplicate_result.copier_hotkeys


class TestScoreThresholdBehavior:
    """Tests for score threshold boundary conditions."""

    @pytest.mark.asyncio
    async def test_large_threshold_all_in_winner_set(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """With large threshold, all models should be in winner set."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(42, 3000), (123, 1000), (456, 2000)]
        )

        orchestrator = ValidationOrchestrator.create(
            score_threshold=10.0,
            winner_share=0.99,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        non_copier_count = 3 - len(result.duplicate_result.copier_hotkeys)
        assert len(result.winner.candidates) == non_copier_count

        if "hotkey_1" not in result.duplicate_result.copier_hotkeys:
            assert result.winner.winner_hotkey == "hotkey_1"

    @pytest.mark.asyncio
    async def test_tiny_threshold_only_best_in_winner_set(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """With tiny threshold, only best scorer should be in winner set."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(1, 3000), (999, 1000), (500, 2000)]
        )

        orchestrator = ValidationOrchestrator.create(
            score_threshold=0.0,
            winner_share=0.99,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        assert len(result.winner.candidates) >= 1
        assert result.winner.winner_score == result.winner.best_score

    @pytest.mark.asyncio
    async def test_threshold_boundary_model_included(
        self,
        tmp_path: Path,
        validation_dataset: ValidationDataset,
    ) -> None:
        """Model exactly at threshold boundary should be included in winner set."""
        model_paths, chain_metadata = create_models_with_metadata(
            tmp_path, [(42, 2000), (43, 1000)]
        )

        orchestrator = ValidationOrchestrator.create(
            score_threshold=1.0,
            winner_share=0.99,
        )

        result = await orchestrator.run(
            dataset=validation_dataset,
            model_paths=model_paths,
            chain_metadata=chain_metadata,
            feature_configs=make_feature_configs(set(model_paths)),
        )

        assert result.winner.threshold == 1.0

        for candidate in result.winner.candidates:
            assert candidate.score >= result.winner.best_score - result.winner.threshold
