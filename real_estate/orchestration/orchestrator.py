"""Validation orchestrator - coordinates the evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from real_estate.data import (
    ConfigEncoder,
    FeatureConfig,
    create_default_feature_config,
)
from real_estate.duplicate_detector import create_duplicate_detector
from real_estate.evaluation import create_orchestrator as create_eval_orchestrator
from real_estate.generalization_detector import (
    GeneralizationConfig,
    GeneralizationDetector,
    perturb_features,
    perturb_spatial,
)
from real_estate.incentives import (
    DistributorConfig,
    IncentiveDistributor,
    NoValidModelsError,
    WinnerSelector,
)
from real_estate.model_inspector import InspectionConfig, ModelInspector

from .models import EncodedModels, ValidationResult

if TYPE_CHECKING:
    from real_estate.chain.models import ChainModelMetadata
    from real_estate.data import FeatureLayout, ValidationDataset
    from real_estate.duplicate_detector import DuplicateDetector
    from real_estate.evaluation import EvaluationOrchestrator, OrchestratorConfig

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """
    Coordinates the full evaluation pipeline.

    Stateless - all inputs passed explicitly, returns result.
    All dependencies injected.

    Pipeline steps:
    0. Pre-flight model inspection (reject memorizers before evaluation)
    1. Encode features per-model from property data + feature configs
    2. Run inference on all models (original features)
    3. Run inference on all models (perturbed features)
    4. Detect duplicate predictions, identify copiers
    5. Detect memorizers via score comparison (original vs perturbed)
    6. Filter cheaters, select winner (threshold + commit time)
    7. Calculate weight distribution (99/1/0 split)
    """

    def __init__(
        self,
        evaluator: EvaluationOrchestrator,
        detector: DuplicateDetector,
        selector: WinnerSelector,
        distributor: IncentiveDistributor,
        generalization_detector: GeneralizationDetector,
        generalization_config: GeneralizationConfig,
        model_inspector: ModelInspector,
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            evaluator: Model evaluation orchestrator
            detector: Duplicate prediction detector
            selector: Winner selection logic
            distributor: Weight distribution logic
            generalization_detector: Perturbation-based memorizer detector
            generalization_config: Config for feature perturbation
            model_inspector: Pre-flight static ONNX inspector
        """
        self._evaluator = evaluator
        self._duplicate_detector = detector
        self._selector = selector
        self._distributor = distributor
        self._generalization_detector = generalization_detector
        self._generalization_config = generalization_config
        self._model_inspector = model_inspector

    def set_seed(self, seed: int | None) -> None:
        """Update the generalization config seed (for decentralized randomness)."""
        self._generalization_config = GeneralizationConfig(
            global_noise_pct=self._generalization_config.global_noise_pct,
            global_threshold=self._generalization_config.global_threshold,
            seed=seed,
            spatial_noise_std=self._generalization_config.spatial_noise_std,
            spatial_threshold=self._generalization_config.spatial_threshold,
        )

    @classmethod
    def create(
        cls,
        *,
        evaluation_config: OrchestratorConfig | None = None,
        similarity_threshold: float = 1e-6,
        score_threshold: float = 0.01,
        winner_share: float = 0.99,
        docker_timeout: int = 300,
        docker_memory: str = "2g",
        docker_cpu: float = 1.0,
        docker_max_concurrent: int = 4,
        inspection_price_threshold: int = 50_000,
        inspection_reject_unused: bool = True,
    ) -> ValidationOrchestrator:
        """
        Create orchestrator with default dependencies.

        Args:
            evaluation_config: Configuration for model evaluation (overrides docker_* params).
            similarity_threshold: Threshold for duplicate prediction detection.
            score_threshold: Score threshold for winner selection.
            winner_share: Share of emissions allocated to winner (default 99%).
            docker_timeout: Docker inference timeout in seconds.
            docker_memory: Docker memory limit (e.g., '2g').
            docker_cpu: Docker CPU limit (1.0 = 1 core).
            docker_max_concurrent: Maximum concurrent Docker evaluations.
            inspection_price_threshold: Min price-like values to flag (default 50K).
            inspection_reject_unused: Reject models with any unused initializers (default True).

        Returns:
            Configured ValidationOrchestrator ready to run evaluations.
        """
        if evaluation_config is not None:
            evaluator = create_eval_orchestrator(
                max_concurrent=evaluation_config.max_concurrent,
                docker_memory=evaluation_config.docker_config.memory_limit,
                docker_timeout=evaluation_config.docker_config.timeout_seconds,
                metrics_config=evaluation_config.metrics_config,
            )
        else:
            evaluator = create_eval_orchestrator(
                max_concurrent=docker_max_concurrent,
                docker_memory=docker_memory,
                docker_cpu=docker_cpu,
                docker_timeout=docker_timeout,
            )

        generalization_config = GeneralizationConfig()

        return cls(
            evaluator=evaluator,
            detector=create_duplicate_detector(
                similarity_threshold=similarity_threshold
            ),
            selector=WinnerSelector(score_threshold),
            distributor=IncentiveDistributor(
                DistributorConfig(winner_share=winner_share)
            ),
            generalization_detector=GeneralizationDetector(generalization_config),
            generalization_config=generalization_config,
            model_inspector=ModelInspector(
                InspectionConfig(
                    price_count_threshold=inspection_price_threshold,
                    reject_unused_initializers=inspection_reject_unused,
                )
            ),
        )

    @staticmethod
    def _slice_perturbed(
        perturbed_superset: np.ndarray,
        superset_layout: FeatureLayout,
        per_model_layouts: dict[str, FeatureLayout],
        hotkeys: dict[str, Path],
    ) -> dict[str, np.ndarray]:
        """Slice per-model columns from a perturbed superset array.

        Args:
            perturbed_superset: Full (N, 79) perturbed feature matrix.
            superset_layout: Layout of the superset (all features).
            per_model_layouts: Per-model layouts mapping feature names to indices.
            hotkeys: Model paths dict (keys used to iterate models).

        Returns:
            Dict of hotkey -> sliced perturbed array for that model's features.
        """
        name_to_idx = {name: i for i, name in enumerate(superset_layout.feature_names)}
        result: dict[str, np.ndarray] = {}
        for hk in hotkeys:
            col_indices = [
                name_to_idx[name] for name in per_model_layouts[hk].feature_names
            ]
            result[hk] = perturbed_superset[:, col_indices]
        return result

    @staticmethod
    def _encode_models(
        model_paths: dict[str, Path],
        feature_configs: dict[str, FeatureConfig | None],
        properties: list[dict],
    ) -> EncodedModels:
        """Encode features per-model, removing models that fail encoding.

        Returns:
            EncodedModels with filtered model_paths, feature arrays, and layouts.

        Raises:
            NoValidModelsError: If all models fail feature encoding.
        """
        default_config = create_default_feature_config()
        features: dict[str, np.ndarray] = {}
        layouts: dict[str, FeatureLayout] = {}
        failures: list[str] = []

        for hotkey in model_paths:
            config = feature_configs.get(hotkey) or default_config
            try:
                encoder = ConfigEncoder(config)
                features[hotkey] = encoder.encode(properties)
                layouts[hotkey] = encoder.layout
                logger.debug(f"Encoded {hotkey}: {features[hotkey].shape} features")
            except Exception as e:
                logger.warning(f"Feature encoding failed for {hotkey}: {e}, skipping")
                failures.append(hotkey)

        if failures:
            model_paths = {k: v for k, v in model_paths.items() if k not in failures}
            if not model_paths:
                raise NoValidModelsError("All models failed feature encoding")

        return EncodedModels(
            model_paths=model_paths,
            features=features,
            layouts=layouts,
        )

    async def run(
        self,
        dataset: ValidationDataset,
        model_paths: dict[str, Path],
        chain_metadata: dict[str, ChainModelMetadata],
        feature_configs: dict[str, FeatureConfig | None] | None = None,
    ) -> ValidationResult:
        """
        Run full evaluation pipeline.

        Args:
            dataset: Validation dataset with properties and ground truth
            model_paths: Mapping of hotkey -> path to ONNX model file
            chain_metadata: Mapping of hotkey -> chain commitment metadata
            feature_configs: Mapping of hotkey -> FeatureConfig (None = use default)

        Returns:
            ValidationResult with weights and evaluation details

        Raises:
            NoValidModelsError: If no models pass evaluation or all are copiers
        """
        logger.info(
            f"Starting evaluation: {len(model_paths)} models, {len(dataset)} samples"
        )

        feature_configs = feature_configs or {}
        default_config = create_default_feature_config()

        # Step 0: Pre-flight model inspection
        # Pass each model's declared feature count so inspector can flag
        # ONNX input-shape mismatches before we waste time running them.
        logger.info("Running pre-flight model inspection...")
        expected_feature_counts = {
            hk: len((feature_configs.get(hk) or default_config).features)
            for hk in model_paths
        }
        inspection_result = await self._model_inspector.inspect_all(
            model_paths, expected_feature_counts
        )
        rejected = inspection_result.rejected_hotkeys
        if rejected:
            logger.info(
                f"Inspection rejected {len(rejected)} models: {sorted(rejected)}"
            )
            model_paths = {k: v for k, v in model_paths.items() if k not in rejected}

        if not model_paths:
            raise NoValidModelsError("All models rejected by pre-flight inspection")

        # 1. Encode features per-model
        logger.debug("Encoding features per-model...")
        ground_truth = np.array(dataset.ground_truth, dtype=np.float32)

        encoded = self._encode_models(
            model_paths,
            feature_configs,
            dataset.properties,
        )
        model_paths = encoded.model_paths
        per_model_features = encoded.features
        per_model_layouts = encoded.layouts

        # 2. Run evaluation on all models (original features)
        logger.info("Running model evaluation...")
        eval_batch = await self._evaluator.evaluate_all(
            models=model_paths,
            features=per_model_features,
            ground_truth=ground_truth,
            model_metadata=chain_metadata,
        )

        successful_count = len(eval_batch.successful_results)
        failed_count = len(eval_batch.results) - successful_count
        logger.info(
            f"Evaluation complete: {successful_count} successful, {failed_count} failed"
        )

        if not eval_batch.successful_results:
            raise NoValidModelsError("All model evaluations failed")

        # 3. Detect duplicates
        logger.debug("Detecting duplicate predictions...")
        duplicates = self._duplicate_detector.detect(
            eval_batch.results,
            chain_metadata,
        )
        copiers = duplicates.copier_hotkeys

        if copiers:
            logger.info(
                f"Detected {len(duplicates.groups)} duplicate groups, "
                f"{len(copiers)} copiers"
            )

        # 4. Generalization detection (perturbation-based)
        # Only run perturbed eval on models that succeeded on original features
        successful_hotkeys = {r.hotkey for r in eval_batch.successful_results}
        perturbed_model_paths = {
            k: v for k, v in model_paths.items() if k in successful_hotkeys
        }
        skipped = len(model_paths) - len(perturbed_model_paths)
        if skipped:
            logger.info(f"Skipping {skipped} failed models from perturbed evaluation")

        # Encode superset once, perturb once, then slice per-model columns
        logger.info("Running generalization detection (perturbed evaluation)...")
        superset_encoder = ConfigEncoder(default_config)
        superset_features = superset_encoder.encode(dataset.properties)
        superset_layout = superset_encoder.layout

        perturbed_superset = perturb_features(
            superset_features,
            self._generalization_config,
            superset_layout,
        )
        spatial_superset = perturb_spatial(
            superset_features,
            self._generalization_config,
            superset_layout,
        )

        per_model_perturbed = self._slice_perturbed(
            perturbed_superset,
            superset_layout,
            per_model_layouts,
            perturbed_model_paths,
        )
        per_model_spatial = self._slice_perturbed(
            spatial_superset,
            superset_layout,
            per_model_layouts,
            perturbed_model_paths,
        )

        perturbed_batch = await self._evaluator.evaluate_all(
            models=perturbed_model_paths,
            features=per_model_perturbed,
            ground_truth=ground_truth,
            model_metadata=chain_metadata,
        )

        logger.info("Running spatial perturbation pass...")
        spatial_batch = await self._evaluator.evaluate_all(
            models=perturbed_model_paths,
            features=per_model_spatial,
            ground_truth=ground_truth,
            model_metadata=chain_metadata,
        )

        generalization_result = self._generalization_detector.detect(
            eval_batch.results, perturbed_batch.results, spatial_batch.results
        )
        memorizers = generalization_result.memorizer_hotkeys
        if memorizers:
            logger.warning(
                f"Generalization check flagged {len(memorizers)} memorizers: "
                f"{sorted(memorizers)}"
            )

        # Combine all cheaters: copiers + memorizers + rejected (from inspection)
        cheaters = copiers | memorizers | rejected

        # 5. Filter cheaters and select winner
        valid_results = [
            r for r in eval_batch.results if r.success and r.hotkey not in cheaters
        ]

        if not valid_results:
            raise NoValidModelsError(
                "No valid models identified for this validation round"
            )

        logger.debug(f"Selecting winner from {len(valid_results)} valid models...")
        winner = self._selector.select_winner(valid_results, chain_metadata)

        logger.info(
            f"Winner: {winner.winner_hotkey} "
            f"(score={winner.winner_score:.4f}, block={winner.winner_block})"
        )

        # 6. Calculate weight distribution
        weights = self._distributor.calculate_weights(
            results=eval_batch.results,
            winner_hotkey=winner.winner_hotkey,
            winner_score=winner.winner_score,
            cheater_hotkeys=cheaters,
        )

        logger.info(
            f"Weights calculated: winner={weights.get_weight(winner.winner_hotkey):.2%}, "
            f"total={weights.total:.4f}"
        )

        return ValidationResult(
            weights=weights,
            winner=winner,
            eval_batch=eval_batch,
            duplicate_result=duplicates,
            generalization_result=generalization_result,
            inspection_result=inspection_result,
            per_model_num_features={
                hk: f.shape[1] for hk, f in per_model_features.items()
            },
        )
