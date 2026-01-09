"""
Duplicate detection module for identifying copied models.

This module provides:
- Prediction grouping to find models with identical outputs
- Pioneer detection to find the earliest committer in each group
- Result aggregation for scoring adjustments

Usage:
    from real_estate.duplicate_detector import (
        create_duplicate_detector,
        DuplicateDetectionResult,
    )

    detector = create_duplicate_detector()
    result = detector.detect(evaluation_results, chain_metadata)

    # Check if a model should be penalized
    if result.is_copier(hotkey):
        score = 0
"""

from .detector import PioneerDetectionResult, PioneerDetector
from .factory import DuplicateDetector, create_duplicate_detector
from .grouper import GrouperConfig, PredictionGrouper
from .models import DuplicateDetectionResult, DuplicateGroup

__all__ = [
    # Factory (main entry point)
    "create_duplicate_detector",
    # Models
    "DuplicateGroup",
    "DuplicateDetectionResult",
    "PioneerDetectionResult",
    # Config
    "GrouperConfig",
    # Components (for advanced usage/testing)
    "DuplicateDetector",
    "PredictionGrouper",
    "PioneerDetector",
]
