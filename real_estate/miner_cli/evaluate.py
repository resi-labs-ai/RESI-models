"""
Local model evaluation for miners.

Runs ONNX inference directly (no Docker) and calculates metrics
to help miners validate their models before submission.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from ..evaluation import MetricsConfig, calculate_metrics
from .config import get_expected_num_features, get_test_data
from .errors import (
    EvaluationError,
    InvalidONNXFormatError,
    ModelInterfaceError,
    ModelNotFoundError,
    ModelSizeExceededError,
)
from .models import EvaluateResult

logger = logging.getLogger(__name__)


def validate_model_file(model_path: Path, max_size_mb: int = 200) -> None:
    """
    Validate model file exists, is within size limit, and is valid ONNX.

    Args:
        model_path: Path to ONNX model file.
        max_size_mb: Maximum model size in MB.

    Raises:
        ModelNotFoundError: If file doesn't exist.
        ModelSizeExceededError: If file exceeds size limit.
        InvalidONNXFormatError: If file is not valid ONNX.
    """
    if not model_path.exists():
        raise ModelNotFoundError(f"Model file not found: {model_path}")

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ModelSizeExceededError(
            f"Model size {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"
        )

    logger.debug(f"Model size: {size_mb:.2f}MB")

    # Validate ONNX format
    try:
        onnx.checker.check_model(str(model_path))
    except onnx.checker.ValidationError as e:
        raise InvalidONNXFormatError(f"Invalid ONNX format: {e}") from e
    except Exception as e:
        raise InvalidONNXFormatError(f"Failed to read ONNX model: {e}") from e


def validate_model_interface(session: ort.InferenceSession) -> str:
    """
    Validate model input/output interface matches expected format.

    Args:
        session: ONNX runtime inference session.

    Returns:
        Input name for the model.

    Raises:
        ModelInterfaceError: If interface doesn't match expected format.
    """
    expected_features = get_expected_num_features()

    # Validate inputs
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ModelInterfaceError(
            f"Model has {len(inputs)} inputs, expected 1. "
            "Model should have a single input for property features."
        )

    input_info = inputs[0]
    input_shape = input_info.shape

    # Expect shape like (batch, num_features) or (None, num_features)
    if len(input_shape) != 2:
        raise ModelInterfaceError(
            f"Input shape {input_shape} invalid. "
            f"Expected 2D shape (batch, {expected_features})."
        )

    # Check feature dimension
    feature_dim = input_shape[1]
    if isinstance(feature_dim, int) and feature_dim != expected_features:
        raise ModelInterfaceError(
            f"Model expects {feature_dim} features, but validator expects {expected_features}. "
            "Ensure your model was trained with the correct feature set."
        )

    # Validate outputs
    outputs = session.get_outputs()
    if len(outputs) != 1:
        raise ModelInterfaceError(
            f"Model has {len(outputs)} outputs, expected 1. "
            "Model should output a single price prediction per sample."
        )

    output_shape = outputs[0].shape

    # Accept (batch,), (batch, 1), or dynamic shapes
    if len(output_shape) == 2:
        out_dim = output_shape[1]
        if isinstance(out_dim, int) and out_dim != 1:
            raise ModelInterfaceError(
                f"Output shape {output_shape} invalid. "
                "Expected (batch,) or (batch, 1) for price predictions."
            )

    logger.debug(
        f"Model interface: input={list(input_shape)}, output={list(output_shape)}"
    )
    return input_info.name


def run_inference(
    session: ort.InferenceSession,
    input_name: str,
    features: np.ndarray,
) -> np.ndarray:
    """
    Run inference on the model.

    Args:
        session: ONNX runtime inference session.
        input_name: Name of the model's input.
        features: Input features array of shape (batch, num_features).

    Returns:
        Predictions array of shape (batch,).

    Raises:
        EvaluationError: If inference fails or produces invalid output.
    """
    try:
        outputs = session.run(None, {input_name: features})
        predictions = outputs[0].flatten()
    except Exception as e:
        raise EvaluationError(f"Inference failed: {e}") from e

    # Validate predictions
    if np.any(np.isnan(predictions)):
        nan_count = np.sum(np.isnan(predictions))
        raise EvaluationError(
            f"Model produced {nan_count} NaN predictions. "
            "Check for numerical instability in your model."
        )

    if np.any(np.isinf(predictions)):
        inf_count = np.sum(np.isinf(predictions))
        raise EvaluationError(
            f"Model produced {inf_count} Inf predictions. "
            "Check for overflow issues in your model."
        )

    return predictions


def evaluate_model(
    model_path: str | Path,
    max_size_mb: int = 200,
) -> EvaluateResult:
    """
    Evaluate an ONNX model locally.

    This runs the same validation as validators (without Docker):
    1. Validates model file (exists, size, ONNX format)
    2. Validates model interface (input/output shapes)
    3. Runs inference on test samples
    4. Calculates metrics (MAPE, score)

    Args:
        model_path: Path to ONNX model file.
        max_size_mb: Maximum model size in MB.

    Returns:
        EvaluateResult with metrics and pass/fail status.
    """
    model_path = Path(model_path)

    # Step 1: Validate model file
    try:
        validate_model_file(model_path, max_size_mb)
        logger.debug("Model file validation passed")
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=str(e),
        )

    # Step 2: Load model
    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=f"Failed to load model: {e}",
        )

    # Step 3: Validate interface
    try:
        input_name = validate_model_interface(session)
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=str(e),
        )

    # Step 4: Load test data
    features, ground_truth = get_test_data()
    logger.debug(f"Loaded {len(features)} test samples")

    # Step 5: Run inference
    try:
        start = time.time()
        predictions = run_inference(session, input_name, features)
        inference_time_ms = (time.time() - start) * 1000
        logger.debug(f"Inference completed in {inference_time_ms:.0f}ms")
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=str(e),
        )

    # Step 6: Calculate metrics
    try:
        metrics = calculate_metrics(ground_truth, predictions, MetricsConfig())
        logger.debug(f"MAPE: {metrics.mape:.2%}, Score: {metrics.score:.4f}")
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=f"Metrics calculation failed: {e}",
        )

    return EvaluateResult(
        model_path=str(model_path),
        success=True,
        metrics=metrics,
        inference_time_ms=inference_time_ms,
    )
