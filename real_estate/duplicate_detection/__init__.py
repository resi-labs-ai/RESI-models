"""
Duplicate detection module for identifying copied models.

This module provides:
- Prediction grouping to find models with identical outputs
- Pioneer detection to find the earliest committer in each group
- Result aggregation for scoring adjustments

Usage:
    from real_estate.duplicate_detection import (
        create_duplicate_detector,
        DuplicateDetectionResult,
    )

    detector = create_duplicate_detector()
    result = detector.detect(evaluation_results, chain_metadata)

    # Pioneers get full score, copiers get penalized
    for hotkey in result.copier_hotkeys:
        print(f"{hotkey} is a copier")
"""

from .detector import PioneerDetectionResult, PioneerDetector
from .errors import DuplicateDetectionError, PioneerDetectionError
from .factory import DuplicateDetector, create_duplicate_detector
from .grouper import GrouperConfig, PredictionGrouper
from .models import DuplicateDetectionResult, DuplicateGroup

__all__ = [
    # Factory (main entry point)
    "create_duplicate_detector",
    # Errors
    "DuplicateDetectionError",
    "PioneerDetectionError",
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
