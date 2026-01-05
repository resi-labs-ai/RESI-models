import onnx
from onnx import helper, TensorProto

# Define a simple model: Y = X
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

node = helper.make_node(
    'Identity',
    inputs=['input'],
    outputs=['output']
)

graph = helper.make_graph(
    [node],
    'identity-model',
    [input_tensor],
    [output_tensor]
)

model = helper.make_model(graph, producer_name='dummy-model')
onnx.save(model, 'dummy_identity.onnx')
