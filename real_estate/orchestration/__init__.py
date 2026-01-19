"""
Validation orchestration - coordinates the evaluation pipeline.

This module provides:
- ValidationOrchestrator: Stateless pipeline coordinator
- ValidationResult: Output from evaluation pipeline

The orchestrator is pure business logic with no infrastructure concerns.
All dependencies are injected, making it easy to test.

Usage:
    from real_estate.orchestration import ValidationOrchestrator, ValidationResult

    orchestrator = ValidationOrchestrator.create()
    result = await orchestrator.run(dataset, model_paths, chain_metadata)
"""

from .models import ValidationResult
from .orchestrator import ValidationOrchestrator

__all__ = [
    "ValidationOrchestrator",
    "ValidationResult",
]
