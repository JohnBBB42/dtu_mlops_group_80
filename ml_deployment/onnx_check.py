import onnx

model = onnx.load("NN.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
