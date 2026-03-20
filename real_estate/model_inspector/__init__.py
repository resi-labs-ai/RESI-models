"""
Model inspection module for pre-flight memorization detection.

Usage:
    from real_estate.model_inspector import InspectionConfig, ModelInspector

    inspector = ModelInspector(InspectionConfig())
    result = inspector.inspect_all(model_paths)

    if result.is_rejected(hotkey):
        # Skip evaluation, assign 0 weight
        ...
"""

from .inspector import ModelInspector
from .models import InspectionBatchResult, InspectionConfig, ModelInspectionResult, RejectionReason

__all__ = [
    "InspectionBatchResult",
    "InspectionConfig",
    "ModelInspectionResult",
    "ModelInspector",
    "RejectionReason",
]
