import onnx
from onnx import TensorProto, helper

# Define a simple model: Y = X
input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 3, 224, 224]
)
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 3, 224, 224]
)

node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

graph = helper.make_graph([node], "identity-model", [input_tensor], [output_tensor])

model = helper.make_model(
    graph,
    producer_name="dummy-model",
    ir_version=8,
    opset_imports=[helper.make_opsetid("", 17)],
)
onnx.save(model, "dummy_identity.onnx")
