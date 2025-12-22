"""Fixtures for evaluation integration tests."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

FIXTURES_DIR = Path(__file__).parent


def create_test_model(
    n_features: int,
    output_path: Path,
    seed: int = 42,
    use_relu: bool = True,
) -> None:
    """
    Create a simple ONNX model for testing.

    Model: output = ReLU(input @ weights + bias)  [if use_relu=True]
           output = input @ weights + bias        [if use_relu=False]

    ReLU ensures non-negative outputs (valid house prices).

    Args:
        n_features: Number of input features
        output_path: Where to save the model
        seed: Random seed for reproducibility
        use_relu: Whether to add ReLU activation (ensures non-negative outputs)
    """
    np.random.seed(seed)
    weights = np.random.randn(n_features, 1).astype(np.float32) * 10000
    bias = np.array([250000.0], dtype=np.float32)

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, n_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1]
    )

    weight_init = helper.make_tensor(
        "weights", TensorProto.FLOAT, [n_features, 1], weights.flatten().tolist()
    )
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [1], bias.tolist())

    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "weights"], outputs=["matmul_out"]
    )

    if use_relu:
        add_node = helper.make_node(
            "Add", inputs=["matmul_out", "bias"], outputs=["add_out"]
        )
        relu_node = helper.make_node("Relu", inputs=["add_out"], outputs=["output"])
        nodes = [matmul_node, add_node, relu_node]
        graph_name = "relu-price-model"
    else:
        add_node = helper.make_node(
            "Add", inputs=["matmul_out", "bias"], outputs=["output"]
        )
        nodes = [matmul_node, add_node]
        graph_name = "linear-price-model"

    graph = helper.make_graph(
        nodes=nodes,
        name=graph_name,
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(
        graph,
        producer_name="resi-test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    # IR version 9 is well-supported by onnxruntime 1.20.1
    # (onnxruntime supports IR versions 3-10)
    model.ir_version = 9

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def create_bad_model(output_path: Path) -> None:
    """
    Create a model that produces NaN outputs (for error testing).

    Model: output = input @ weights * nan_multiplier
    The nan_multiplier is 0/0 = NaN, so all outputs are NaN.
    """
    n_features = 10

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, n_features]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1]
    )

    # Normal weights for matmul
    weights = np.ones((n_features, 1), dtype=np.float32)
    weight_init = helper.make_tensor(
        "weights", TensorProto.FLOAT, [n_features, 1], weights.flatten().tolist()
    )

    # Zero constant for 0/0 = NaN
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])

    # MatMul: input @ weights -> (N, 1)
    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "weights"], outputs=["matmul_out"]
    )

    # Div: 0/0 = NaN
    div_node = helper.make_node(
        "Div", inputs=["zero", "zero"], outputs=["nan_scalar"]
    )

    # Mul: matmul_out * NaN = NaN for all outputs
    mul_node = helper.make_node(
        "Mul", inputs=["matmul_out", "nan_scalar"], outputs=["output"]
    )

    graph = helper.make_graph(
        nodes=[matmul_node, div_node, mul_node],
        name="nan-model",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, zero_init],
    )

    model = helper.make_model(
        graph,
        producer_name="resi-test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    # IR version 9 is well-supported by onnxruntime 1.20.1
    model.ir_version = 9

    onnx.save(model, str(output_path))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def dummy_linear_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return path to a dummy linear model."""
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "linear_model.onnx"
    create_test_model(n_features=10, output_path=model_path)
    return model_path


@pytest.fixture(scope="session")
def dummy_input_data() -> np.ndarray:
    """Create sample input features (100 samples, 10 features)."""
    np.random.seed(123)
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture(scope="session")
def dummy_ground_truth() -> np.ndarray:
    """Create sample ground truth prices."""
    np.random.seed(456)
    # Realistic house prices: $100k - $1M
    return np.random.uniform(100000, 1000000, size=100).astype(np.float32)
