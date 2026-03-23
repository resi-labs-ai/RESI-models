"""Generalization detector for identifying memorizing models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import (
    GeneralizationConfig,
    GeneralizationDetectionResult,
    GeneralizationTestResult,
)

if TYPE_CHECKING:
    from ..evaluation.models import EvaluationResult

logger = logging.getLogger(__name__)


class GeneralizationDetector:
    """
    Detect models that memorized training data instead of learning to generalize.

    Compares scores from original vs perturbed features.
    A model is flagged as a memorizer if perturbed_score / original_score
    falls below the threshold.

    Usage:
        detector = GeneralizationDetector(GeneralizationConfig())
        result = detector.detect(original_results, perturbed_results)
        if result.is_memorizer(hotkey):
            weight = 0
    """

    def __init__(self, config: GeneralizationConfig):
        self._config = config

    @property
    def config(self) -> GeneralizationConfig:
        """Get the detector configuration."""
        return self._config

    def detect(
        self,
        original_results: list[EvaluationResult],
        perturbed_results: list[EvaluationResult],
    ) -> GeneralizationDetectionResult:
        """
        Compare original vs perturbed evaluation results for memorization.

        Args:
            original_results: Evaluation results on original features.
            perturbed_results: Evaluation results on perturbed features.

        Returns:
            GeneralizationDetectionResult with memorizer hotkeys.
        """
        perturbed_by_hotkey = {r.hotkey: r for r in perturbed_results}

        memorizers: set[str] = set()
        test_results: list[GeneralizationTestResult] = []

        for original in original_results:
            if not original.success:
                continue

            perturbed = perturbed_by_hotkey.get(original.hotkey)
            if perturbed is None or not perturbed.success:
                logger.warning(
                    f"Skipping {original.hotkey}: succeeded on original but "
                    f"{'missing from' if perturbed is None else 'failed on'} "
                    f"perturbed evaluation"
                )
                continue

            original_score = original.score
            perturbed_score = perturbed.score

            global_ratio = (
                perturbed_score / original_score if original_score > 0 else 0.0
            )
            is_memorizer = global_ratio < self._config.global_threshold

            if is_memorizer:
                memorizers.add(original.hotkey)
                logger.warning(
                    f"Memorizer detected: {original.hotkey} "
                    f"(global_ratio={global_ratio:.3f})"
                )

            test_results.append(
                GeneralizationTestResult(
                    hotkey=original.hotkey,
                    original_score=original_score,
                    perturbed_score=perturbed_score,
                    global_ratio=global_ratio,
                    is_memorizer=is_memorizer,
                )
            )

        if test_results:
            logger.info(
                f"Generalization check: {len(test_results)} tested, "
                f"{len(memorizers)} memorizers"
            )

        return GeneralizationDetectionResult(
            memorizer_hotkeys=frozenset(memorizers),
            test_results=tuple(test_results),
        )
