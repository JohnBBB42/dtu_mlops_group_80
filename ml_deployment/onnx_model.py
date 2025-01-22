import torch
from energy.model import NeuralNetwork

input_size = 10
model = NeuralNetwork(input_size=input_size)
model.eval()

dummy_input = torch.randn(1, input_size)
model.to_onnx(
    file_path="NN.onnx",
    input_sample=dummy_input,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
