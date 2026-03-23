"""
ONNX model inspection script that runs INSIDE a Docker container.

Analyzes model weights for memorization indicators (price-like values,
lookup patterns, unused initializers) and writes results to /workspace/.

Usage (inside container):
    python /workspace/inspection_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx

# --- Constants ---
PRICE_MIN = 10_000
PRICE_MAX = 50_000_000
PRICE_COUNT_THRESHOLD = 100
LARGE_DIM_THRESHOLD = 10_000

MODEL_PATH = Path("/workspace/model.onnx")
OUTPUT_PATH = Path("/workspace/inspection_results.json")


def numpy_from_tensor(tensor_proto) -> np.ndarray | None:
    try:
        return np.array(onnx.numpy_helper.to_array(tensor_proto))
    except Exception:
        return None


def find_used_initializer_names(graph) -> set[str]:
    """Find initializer names that are actually referenced by graph nodes."""
    used = set()
    for node in graph.node:
        for inp in node.input:
            used.add(inp)
    return used


def analyze_model(model_path: Path) -> dict:
    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        return {"error": str(e)}

    graph = model.graph
    used_names = find_used_initializer_names(graph)

    results: dict = {
        "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "total_params": 0,
        "total_initializers": len(graph.initializer),
        "unused_initializers": 0,
        "unused_initializer_names": [],
        "node_count": len(graph.node),
        "op_types": {},
        "suspicious_tensors": [],
        "price_like_values_total": 0,
        "largest_tensor_shape": [],
        "largest_tensor_params": 0,
    }

    # Count operation types
    for node in graph.node:
        results["op_types"][node.op_type] = results["op_types"].get(node.op_type, 0) + 1

    # Analyze each initializer
    for init in graph.initializer:
        shape = list(init.dims)
        n_params = int(np.prod(shape)) if shape else 0
        results["total_params"] += n_params

        is_unused = init.name not in used_names
        if is_unused:
            results["unused_initializers"] += 1
            results["unused_initializer_names"].append(init.name)

        if n_params > results["largest_tensor_params"]:
            results["largest_tensor_params"] = n_params
            results["largest_tensor_shape"] = shape

        tensor = numpy_from_tensor(init)
        if tensor is None or tensor.size == 0:
            continue

        flat = tensor.flatten().astype(np.float64)
        price_mask = (np.abs(flat) >= PRICE_MIN) & (np.abs(flat) <= PRICE_MAX)
        price_count = int(np.sum(price_mask))

        if price_count >= PRICE_COUNT_THRESHOLD:
            price_values = flat[price_mask]
            results["suspicious_tensors"].append({
                "name": init.name,
                "shape": shape,
                "params": n_params,
                "unused": is_unused,
                "price_like_count": price_count,
                "price_like_pct": round(100.0 * price_count / flat.size, 1),
                "price_like_min": round(float(np.min(price_values)), 2),
                "price_like_max": round(float(np.max(price_values)), 2),
                "price_like_mean": round(float(np.mean(price_values)), 2),
            })

        results["price_like_values_total"] += price_count

        for dim in shape:
            if dim >= LARGE_DIM_THRESHOLD and price_count < PRICE_COUNT_THRESHOLD:
                results["suspicious_tensors"].append({
                    "name": init.name,
                    "shape": shape,
                    "params": n_params,
                    "unused": is_unused,
                    "reason": f"large dimension ({dim:,})",
                })
                break

    # Detect lookup patterns
    results["has_topk"] = "TopK" in results["op_types"]
    results["has_argmin"] = "ArgMin" in results["op_types"]
    results["has_argmax"] = "ArgMax" in results["op_types"]
    results["has_gather"] = "Gather" in results["op_types"]
    results["lookup_pattern"] = (
        (results["has_topk"] or results["has_argmin"] or results["has_argmax"])
        and results["has_gather"]
    )

    # Classify
    flags = []
    if results["price_like_values_total"] >= PRICE_COUNT_THRESHOLD:
        flags.append("PRICES_IN_WEIGHTS")
    if results["lookup_pattern"]:
        flags.append("LOOKUP_PATTERN")
    if results["largest_tensor_params"] >= LARGE_DIM_THRESHOLD:
        flags.append("LARGE_TENSOR")
    if results["unused_initializers"] > results["total_initializers"] * 0.5:
        flags.append("MANY_UNUSED_INITIALIZERS")

    results["classification"] = ("SUSPICIOUS: " + ", ".join(flags)) if flags else "CLEAN"
    results["unused_initializer_ratio"] = (
        round(results["unused_initializers"] / results["total_initializers"], 3)
        if results["total_initializers"] > 0
        else 0.0
    )

    return results


def main():
    if not MODEL_PATH.exists():
        print(json.dumps({"error": "model.onnx not found in /workspace/"}))
        sys.exit(1)

    results = analyze_model(MODEL_PATH)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[RESULT] {json.dumps(results, default=str)}")


if __name__ == "__main__":
    main()
