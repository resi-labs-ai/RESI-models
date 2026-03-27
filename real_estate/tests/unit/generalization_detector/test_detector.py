"""Tests for GeneralizationDetector."""

from __future__ import annotations

import numpy as np

from real_estate.evaluation.models import EvaluationResult, PredictionMetrics
from real_estate.generalization_detector import (
    GeneralizationConfig,
    GeneralizationDetector,
)


def _make_metrics(score: float = 0.8) -> PredictionMetrics:
    mape = 1.0 - score
    return PredictionMetrics(
        mape=mape, mae=10000, rmse=15000, mdape=mape,
        accuracy={0.05: 0.3, 0.10: 0.6}, r2=score, n_samples=100,
    )


def _make_result(hotkey: str, score: float = 0.8, success: bool = True) -> EvaluationResult:
    if not success:
        return EvaluationResult(hotkey=hotkey, error=Exception("fail"))
    return EvaluationResult(
        hotkey=hotkey,
        predictions=np.array([1.0, 2.0]),
        metrics=_make_metrics(score),
    )


class TestGeneralizationDetector:
    def test_memorizer_detected(self) -> None:
        """Model with big score drop on perturbed data is a memorizer."""
        detector = GeneralizationDetector(GeneralizationConfig(global_threshold=0.70))
        original = [_make_result("bad", score=0.9)]
        perturbed = [_make_result("bad", score=0.2)]
        detection = detector.detect(original, perturbed)

        assert detection.is_memorizer("bad")
        assert len(detection.memorizer_hotkeys) == 1

    def test_robust_model_passes(self) -> None:
        """Model with stable score on perturbed data passes."""
        detector = GeneralizationDetector(GeneralizationConfig())
        original = [_make_result("good", score=0.9)]
        perturbed = [_make_result("good", score=0.85)]
        detection = detector.detect(original, perturbed)

        assert not detection.is_memorizer("good")
        assert len(detection.test_results) == 1
        assert not detection.test_results[0].is_memorizer

    def test_at_threshold_passes(self) -> None:
        """Model exactly at threshold passes."""
        detector = GeneralizationDetector(GeneralizationConfig(global_threshold=0.70))
        original = [_make_result("edge", score=0.9)]
        perturbed = [_make_result("edge", score=0.63)]  # 0.63/0.9 = 0.70
        detection = detector.detect(original, perturbed)

        assert not detection.is_memorizer("edge")

    def test_just_below_threshold_fails(self) -> None:
        """Model just below threshold is flagged."""
        detector = GeneralizationDetector(GeneralizationConfig(global_threshold=0.70))
        original = [_make_result("edge", score=0.9)]
        perturbed = [_make_result("edge", score=0.629)]  # 0.629/0.9 ≈ 0.699
        detection = detector.detect(original, perturbed)

        assert detection.is_memorizer("edge")

    def test_failed_original_skipped(self) -> None:
        """Failed original evaluations are skipped."""
        detector = GeneralizationDetector(GeneralizationConfig())
        original = [_make_result("failed", success=False)]
        perturbed = [_make_result("failed", score=0.8)]
        detection = detector.detect(original, perturbed)

        assert len(detection.test_results) == 0

    def test_failed_perturbed_skipped(self) -> None:
        """Models that failed perturbed evaluation are skipped (likely OOM)."""
        detector = GeneralizationDetector(GeneralizationConfig())
        original = [_make_result("ok", score=0.9)]
        perturbed = [_make_result("ok", success=False)]
        detection = detector.detect(original, perturbed)

        assert len(detection.test_results) == 0
        assert not detection.is_memorizer("ok")

    def test_missing_perturbed_skipped(self) -> None:
        """Models with no perturbed result are skipped."""
        detector = GeneralizationDetector(GeneralizationConfig())
        original = [_make_result("alone", score=0.9)]
        perturbed: list[EvaluationResult] = []
        detection = detector.detect(original, perturbed)

        assert len(detection.test_results) == 0
        assert not detection.is_memorizer("alone")

    def test_mixed_models(self) -> None:
        """Mix of memorizers, good models, failed original, and OOM on perturbed."""
        detector = GeneralizationDetector(GeneralizationConfig(global_threshold=0.70))
        original = [
            _make_result("memorizer", score=0.9),
            _make_result("good1", score=0.9),
            _make_result("good2", score=0.85),
            _make_result("failed_orig", success=False),
            _make_result("oom_perturbed", score=0.9),
        ]
        perturbed = [
            _make_result("memorizer", score=0.1),  # 0.1/0.9 ≈ 0.11
            _make_result("good1", score=0.85),  # 0.85/0.9 ≈ 0.94
            _make_result("good2", score=0.80),  # 0.80/0.85 ≈ 0.94
            _make_result("failed_orig", success=False),
            _make_result("oom_perturbed", success=False),  # OOM on perturbed — skipped
        ]
        detection = detector.detect(original, perturbed)

        assert detection.memorizer_hotkeys == frozenset({"memorizer"})
        assert len(detection.test_results) == 3  # good1, good2, memorizer (oom skipped)

    def test_zero_original_score(self) -> None:
        """Model with zero original score is not a memorizer (can't memorize with score<=0)."""
        detector = GeneralizationDetector(GeneralizationConfig(global_threshold=0.70))
        original = [_make_result("zero", score=0.0)]
        perturbed = [_make_result("zero", score=0.0)]
        detection = detector.detect(original, perturbed)

        # ratio = 1.0 (passes) — model with score<=0 can't be a memorizer
        assert not detection.is_memorizer("zero")
        assert detection.test_results[0].global_ratio == 1.0

    def test_result_contains_scores(self) -> None:
        """Test results include original and perturbed scores."""
        detector = GeneralizationDetector(GeneralizationConfig())
        original = [_make_result("model", score=0.9)]
        perturbed = [_make_result("model", score=0.85)]
        detection = detector.detect(original, perturbed)

        tr = detection.test_results[0]
        assert tr.original_score == original[0].score
        assert tr.perturbed_score == perturbed[0].score

    def test_spatial_below_threshold_flagged(self) -> None:
        """Model with spatial ratio below spatial_threshold is flagged."""
        detector = GeneralizationDetector(
            GeneralizationConfig(global_threshold=0.60, spatial_threshold=0.90)
        )
        original = [_make_result("knn", score=0.9)]
        perturbed = [_make_result("knn", score=0.9)]  # global passes (ratio=1.0)
        spatial = [_make_result("knn", score=0.72)]  # spatial ratio=0.80 < 0.90

        detection = detector.detect(original, perturbed, spatial)

        assert detection.is_memorizer("knn")
        tr = detection.test_results[0]
        assert tr.spatial_ratio is not None
        assert tr.spatial_ratio < 0.90

    def test_spatial_above_threshold_passes(self) -> None:
        """Model with spatial ratio above spatial_threshold passes."""
        detector = GeneralizationDetector(
            GeneralizationConfig(global_threshold=0.60, spatial_threshold=0.90)
        )
        original = [_make_result("good", score=0.9)]
        perturbed = [_make_result("good", score=0.88)]  # global ratio ~0.98
        spatial = [_make_result("good", score=0.89)]  # spatial ratio ~0.99

        detection = detector.detect(original, perturbed, spatial)

        assert not detection.is_memorizer("good")
        tr = detection.test_results[0]
        assert tr.spatial_ratio is not None
        assert tr.spatial_ratio >= 0.90

    def test_global_passes_spatial_fails_flagged(self) -> None:
        """Either check failing triggers memorizer flag."""
        detector = GeneralizationDetector(
            GeneralizationConfig(global_threshold=0.60, spatial_threshold=0.90)
        )
        original = [_make_result("sneaky", score=0.9)]
        perturbed = [_make_result("sneaky", score=0.85)]  # global ratio ~0.94 — passes
        spatial = [_make_result("sneaky", score=0.70)]  # spatial ratio ~0.78 — fails

        detection = detector.detect(original, perturbed, spatial)
        assert detection.is_memorizer("sneaky")

    def test_spatial_passes_global_fails_flagged(self) -> None:
        """Global failing still flags even when spatial passes."""
        detector = GeneralizationDetector(
            GeneralizationConfig(global_threshold=0.60, spatial_threshold=0.90)
        )
        original = [_make_result("bad", score=0.9)]
        perturbed = [_make_result("bad", score=0.2)]  # global ratio ~0.22 — fails
        spatial = [_make_result("bad", score=0.88)]  # spatial ratio ~0.98 — passes

        detection = detector.detect(original, perturbed, spatial)
        assert detection.is_memorizer("bad")

    def test_no_spatial_results_falls_back_to_global_only(self) -> None:
        """When spatial_results is None, only global check applies."""
        detector = GeneralizationDetector(
            GeneralizationConfig(global_threshold=0.60, spatial_threshold=0.90)
        )
        original = [_make_result("model", score=0.9)]
        perturbed = [_make_result("model", score=0.85)]  # global ratio ~0.94

        detection = detector.detect(original, perturbed)  # no spatial

        assert not detection.is_memorizer("model")
        tr = detection.test_results[0]
        assert tr.spatial_ratio is None

