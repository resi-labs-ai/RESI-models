"""
Inference script that runs inside Docker container.

This script is copied into the container workspace and executed.
It loads an ONNX model, runs inference, and saves predictions.

Exit codes:
    0 - Success
    1 - onnxruntime not installed
    2 - Failed to load input
    3 - Failed to load model
    4 - Inference failed
    5 - Failed to save output
    6 - Output validation failed
"""

import sys
import time
import traceback

import numpy as np

print("[INFO] Inference script starting...")
print(f"[INFO] Python version: {sys.version}")
print(f"[INFO] NumPy version: {np.__version__}")

try:
    import onnxruntime as ort

    print(f"[INFO] ONNX Runtime version: {ort.__version__}")
except ImportError as e:
    print(f"[ERROR] onnxruntime not installed: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    model_path = "/workspace/model.onnx"
    input_path = "/workspace/input.npy"
    output_path = "/workspace/output.npy"

    total_start = time.time()

    # Load input
    print(f"[INFO] Loading input from {input_path}...")
    try:
        load_start = time.time()
        input_data = np.load(input_path)
        load_time = (time.time() - load_start) * 1000
        print(
            f"[INFO] Input loaded: shape={input_data.shape}, dtype={input_data.dtype}"
        )
        print(f"[INFO] Input load time: {load_time:.2f}ms")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Failed to load input: {e}", file=sys.stderr)
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(2)

    # Load model
    print(f"[INFO] Loading ONNX model from {model_path}...")
    try:
        load_start = time.time()
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        load_time = (time.time() - load_start) * 1000
        print(f"[INFO] Model loaded successfully in {load_time:.2f}ms")

        # Log model metadata
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"[INFO] Model inputs: {[(i.name, i.shape, i.type) for i in inputs]}")
        print(f"[INFO] Model outputs: {[(o.name, o.shape, o.type) for o in outputs]}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(3)

    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"[INFO] Using input name: {input_name}")

    # Run inference
    print(f"[INFO] Running inference on {len(input_data)} samples...")
    try:
        inference_start = time.time()
        input_tensor = input_data.astype(np.float32)
        outputs = session.run(None, {input_name: input_tensor})
        predictions = outputs[0]
        inference_time = (time.time() - inference_start) * 1000
        print(f"[INFO] Inference completed in {inference_time:.2f}ms")
        print(
            f"[INFO] Predictions shape: {predictions.shape}, dtype: {predictions.dtype}"
        )
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        print(f"[ERROR] Input tensor shape: {input_data.shape}", file=sys.stderr)
        print(f"[ERROR] Input tensor dtype: {input_data.dtype}", file=sys.stderr)
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(4)

    # Validate predictions
    print("[INFO] Validating predictions...")
    predictions_flat = predictions.flatten()
    nan_count = np.sum(np.isnan(predictions_flat))
    inf_count = np.sum(np.isinf(predictions_flat))
    neg_count = np.sum(predictions_flat < 0)

    if nan_count > 0:
        print(f"[ERROR] Predictions contain {nan_count} NaN values", file=sys.stderr)
        sys.exit(6)
    if inf_count > 0:
        print(f"[ERROR] Predictions contain {inf_count} Inf values", file=sys.stderr)
        sys.exit(6)
    if neg_count > 0:
        print(f"[INFO] Predictions contain {neg_count} negative values")

    print(
        f"[INFO] Prediction stats: min={predictions_flat.min():.2f}, max={predictions_flat.max():.2f}, mean={predictions_flat.mean():.2f}"
    )

    # Save output
    print(f"[INFO] Saving predictions to {output_path}...")
    try:
        save_start = time.time()
        np.save(output_path, predictions)
        save_time = (time.time() - save_start) * 1000
        print(f"[INFO] Output saved in {save_time:.2f}ms")
    except Exception as e:
        print(f"[ERROR] Failed to save output: {e}", file=sys.stderr)
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(5)

    total_time = (time.time() - total_start) * 1000
    print(
        f"[SUCCESS] Generated {len(predictions_flat)} predictions in {total_time:.2f}ms total"
    )


if __name__ == "__main__":
    main()
