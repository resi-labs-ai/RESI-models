"""Factory functions for creating duplicate detection components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .detector import PioneerDetector
from .grouper import GrouperConfig, PredictionGrouper
from .models import DuplicateDetectionResult

if TYPE_CHECKING:
    from ..chain.models import ChainModelMetadata
    from ..evaluation.models import EvaluationResult


class DuplicateDetector:
    """
    Facade for complete duplicate detection workflow.

    Combines PredictionGrouper and PioneerDetector into a single interface.

    Usage:
        detector = create_duplicate_detector()
        result = detector.detect(evaluation_results, chain_metadata)
        print(f"Found {len(result.groups)} duplicate groups")
        print(f"Pioneers: {result.pioneer_hotkeys}")
    """

    def __init__(
        self,
        grouper: PredictionGrouper,
        pioneer_detector: PioneerDetector,
    ):
        """
        Initialize detector with components.

        Prefer using create_duplicate_detector() factory.
        """
        self._grouper = grouper
        self._pioneer_detector = pioneer_detector

    def detect(
        self,
        results: list[EvaluationResult],
        metadata: dict[str, ChainModelMetadata],
    ) -> DuplicateDetectionResult:
        """
        Run complete duplicate detection pipeline.

        Args:
            results: Evaluation results from EvaluationOrchestrator
            metadata: Chain metadata for all models. Hotkeys missing from
                     this dict will be skipped (not fail).

        Returns:
            DuplicateDetectionResult with groups, pioneer information,
            and list of skipped hotkeys (those without metadata).
        """
        groups = self._grouper.group_predictions(results)

        if not groups:
            return DuplicateDetectionResult(copier_hotkeys=frozenset())

        pioneer_result = self._pioneer_detector.detect_pioneers(groups, metadata)

        return DuplicateDetectionResult(
            copier_hotkeys=pioneer_result.copier_hotkeys,
            pioneer_hotkeys=pioneer_result.pioneer_hotkeys,
            groups=tuple(groups),
            skipped_hotkeys=tuple(pioneer_result.skipped_hotkeys),
        )


def create_duplicate_detector(
    similarity_threshold: float = 1e-6,
) -> DuplicateDetector:
    """
    Create a duplicate detector with custom configuration.

    This is the main entry point for the duplicate_detector module.

    Args:
        similarity_threshold: Precision threshold for prediction comparison.
            Predictions within this threshold are considered identical.
            Default 1e-6 (exact match to 6 decimal places).

    Returns:
        Configured DuplicateDetector ready to use

    Example:
        detector = create_duplicate_detector()
        result = detector.detect(evaluation_results, chain_metadata)

        # Or for stricter matching:
        detector = create_duplicate_detector(similarity_threshold=1e-9)
    """
    config = GrouperConfig(similarity_threshold=similarity_threshold)
    grouper = PredictionGrouper(config)
    pioneer_detector = PioneerDetector()

    return DuplicateDetector(grouper, pioneer_detector)
