"""
Evaluation module for running ONNX models and calculating metrics.

This module provides:
- Docker-based isolated model execution
- Comprehensive prediction metrics (MAE, MAPE, RMSE, MdAPE, Accuracy, R²)
- Evaluation orchestration for multiple models
- Result aggregation and ranking

Usage:
    from real_estate.evaluation import (
        EvaluationOrchestrator,
        calculate_metrics,
        MetricsConfig,
        create_orchestrator,
    )

    # Quick metrics calculation
    metrics = calculate_metrics(y_true, y_pred, MetricsConfig())
    print(f"MAPE: {metrics.mape:.2%}, Score: {metrics.score:.2f}")

    # Full evaluation with Docker
    orchestrator = create_orchestrator()
    batch = await orchestrator.evaluate_all(models, features, ground_truth)
    best = batch.get_best()

See docs/EVALUATION_METRICS.md for metric documentation.
"""

from .docker_runner import DockerConfig, DockerRunner, InferenceResult
from .errors import (
    DockerError,
    DockerExecutionError,
    DockerImageError,
    DockerNotAvailableError,
    EmptyDatasetError,
    EvaluationError,
    InferenceTimeoutError,
    InvalidPredictionError,
    MetricsError,
)
from .metrics import (
    MetricsConfig,
    calculate_metrics,
    mape_to_score,
    score_to_mape,
    validate_predictions,
)
from .models import EvaluationBatch, EvaluationResult, PredictionMetrics
from .orchestrator import (
    EvaluationOrchestrator,
    OrchestratorConfig,
    create_orchestrator,
)

__all__ = [
    # Factory (main entry points)
    "create_orchestrator",
    "calculate_metrics",
    # Orchestrator
    "EvaluationOrchestrator",
    "OrchestratorConfig",
    # Docker
    "DockerRunner",
    "DockerConfig",
    "InferenceResult",
    # Metrics
    "MetricsConfig",
    "mape_to_score",
    "score_to_mape",
    "validate_predictions",
    # Models
    "PredictionMetrics",
    "EvaluationResult",
    "EvaluationBatch",
    # Errors
    "EvaluationError",
    "MetricsError",
    "EmptyDatasetError",
    "DockerError",
    "DockerNotAvailableError",
    "DockerImageError",
    "DockerExecutionError",
    "InferenceTimeoutError",
    "InvalidPredictionError",
]
