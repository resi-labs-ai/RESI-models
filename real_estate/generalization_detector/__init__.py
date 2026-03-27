"""
Generalization detection module for identifying memorizing models.

Usage:
    from real_estate.generalization_detector import (
        GeneralizationConfig,
        GeneralizationDetector,
        perturb_features,
    )

    config = GeneralizationConfig()
    perturbed = perturb_features(features, config)
    # Run evaluation on both original and perturbed features
    result = detector.detect(original_results, perturbed_results)

    if result.is_memorizer(hotkey):
        weight = 0
"""

from .detector import GeneralizationDetector
from .models import (
    GeneralizationConfig,
    GeneralizationDetectionResult,
    GeneralizationTestResult,
)
from .perturbation import perturb_features, perturb_spatial

__all__ = [
    "GeneralizationConfig",
    "GeneralizationDetectionResult",
    "GeneralizationDetector",
    "GeneralizationTestResult",
    "perturb_features",
    "perturb_spatial",
]
