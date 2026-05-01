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

from ..data.config_encoder import (
    IMAGES_FEATURE_NAME,
    FeatureConfig,
    TabularEncoder,
    create_default_feature_config,
    load_feature_config,
)
from ..evaluation import MetricsConfig, calculate_metrics
from .config import get_expected_num_features, load_test_samples
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


def validate_model_interface(
    session: ort.InferenceSession,
    expected_features: int | None = None,
    has_images: bool = False,
) -> str:
    """
    Validate model input/output interface matches expected format.

    Args:
        session: ONNX runtime inference session.
        expected_features: Expected number of numeric/boolean input features.
        has_images: Whether the model declared property_images.

    Returns:
        Input name for the features (numeric) input.

    Raises:
        ModelInterfaceError: If interface doesn't match expected format.
    """
    if expected_features is None:
        expected_features = get_expected_num_features()

    inputs = session.get_inputs()
    input_names = {i.name for i in inputs}

    if has_images:
        # Multi-input model: expect "features", "images", "image_counts"
        required_inputs = {"features", "images", "image_counts"}
        missing = required_inputs - input_names
        if missing:
            raise ModelInterfaceError(
                f"Image model missing required inputs: {sorted(missing)}. "
                f"Model has: {sorted(input_names)}"
            )
        features_input = next(i for i in inputs if i.name == "features")
    else:
        # Single-input model
        if len(inputs) != 1:
            raise ModelInterfaceError(
                f"Model has {len(inputs)} inputs, expected 1. "
                "Model should have a single input for property features."
            )
        features_input = inputs[0]

    input_shape = features_input.shape

    # Expect shape like (batch, num_features) or (None, num_features)
    if len(input_shape) != 2:
        raise ModelInterfaceError(
            f"Features input shape {input_shape} invalid. "
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
    return features_input.name


def run_inference(
    session: ort.InferenceSession,
    input_name: str,
    features: np.ndarray,
    image_block: object | None = None,
) -> np.ndarray:
    """
    Run inference on the model.

    Args:
        session: ONNX runtime inference session.
        input_name: Name of the features input.
        features: Input features array of shape (batch, num_features).
        image_block: ImageBlockConfig if model uses images (generates dummy data).

    Returns:
        Predictions array of shape (batch,).

    Raises:
        EvaluationError: If inference fails or produces invalid output.
    """
    try:
        input_feed = {input_name: features}

        if image_block is not None:
            batch_size = features.shape[0]
            c, h, w = image_block.dim
            max_imgs = image_block.max_images_per_property

            # Generate zero-padded dummy images (simulates properties with no photos)
            dummy_images = np.zeros(
                (batch_size, max_imgs, c, h, w), dtype=np.uint8
            )
            dummy_counts = np.zeros(batch_size, dtype=np.int32)

            input_feed["images"] = dummy_images
            input_feed["image_counts"] = dummy_counts

            logger.debug(
                f"Using dummy images: shape={dummy_images.shape} "
                f"(zero-padded, simulating no available photos)"
            )

        outputs = session.run(None, input_feed)
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


def resolve_feature_config(
    feature_config_path: str | Path | None = None,
) -> FeatureConfig:
    """
    Load feature config from path, or return default (all 79 features).

    Args:
        feature_config_path: Path to feature_config.json, or None for default.

    Returns:
        Parsed FeatureConfig.

    Raises:
        FeatureConfigError: If the file is invalid.
        FileNotFoundError: If the file doesn't exist.
    """
    if feature_config_path is not None:
        feature_config = load_feature_config(Path(feature_config_path))
        logger.debug(f"Feature config loaded: {len(feature_config.features)} features")
    else:
        feature_config = create_default_feature_config()
        logger.debug("Using default feature config (all features)")
    return feature_config


def encode_test_data(
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test samples and encode features using the given config.

    Args:
        feature_config: Feature config defining which features to encode.

    Returns:
        Tuple of (features, ground_truth) arrays.

    Raises:
        EvaluationError: If test data loading or encoding fails.
    """
    samples = load_test_samples()
    encoder = TabularEncoder(feature_config)
    properties = [s["features"] for s in samples]
    features = encoder.encode(properties)
    ground_truth = np.array(
        [float(s["actual_price"]) for s in samples], dtype=np.float32
    )
    logger.debug(f"Encoded {len(features)} test samples ({features.shape[1]} features)")
    return features, ground_truth


def evaluate_model(
    model_path: str | Path,
    max_size_mb: int = 200,
    feature_config_path: str | Path | None = None,
) -> EvaluateResult:
    """
    Evaluate an ONNX model locally.

    This runs the same validation as validators (without Docker):
    1. Validates model file (exists, size, ONNX format)
    2. Validates feature config (if provided)
    3. Validates model interface (input/output shapes)
    4. Runs inference on test samples
    5. Calculates metrics (MAPE, score)

    Args:
        model_path: Path to ONNX model file.
        max_size_mb: Maximum model size in MB.
        feature_config_path: Path to feature_config.json. If None, uses all default features.

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

    # Step 2: Load and validate feature config
    try:
        feature_config = resolve_feature_config(feature_config_path)
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=f"Invalid feature_config.json: {e}",
        )

    has_images = feature_config.image_block is not None
    # Numeric/boolean features only — property_images isn't a column
    expected_features = len([
        f for f in feature_config.features if f != IMAGES_FEATURE_NAME
    ])

    # Step 3: Load model
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

    # Step 4: Validate interface
    try:
        input_name = validate_model_interface(session, expected_features, has_images)
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=str(e),
        )

    # Step 5: Load test data and encode with feature config
    try:
        features, ground_truth = encode_test_data(feature_config)
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=f"Failed to encode test data: {e}",
        )

    # Step 6: Run inference
    try:
        start = time.time()
        predictions = run_inference(
            session, input_name, features, feature_config.image_block
        )
        inference_time_ms = (time.time() - start) * 1000
        logger.debug(f"Inference completed in {inference_time_ms:.0f}ms")
    except Exception as e:
        return EvaluateResult(
            model_path=str(model_path),
            success=False,
            error_message=str(e),
        )

    # Step 7: Calculate metrics
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
