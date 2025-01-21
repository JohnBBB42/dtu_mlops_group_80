import onnxruntime as rt

# Configure session options for optimization
sess_options = rt.SessionOptions()

# Set graph optimization level to extended optimizations
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Specify the path to save the optimized model
sess_options.optimized_model_filepath = "optimized_model.onnx"

# Initialize an inference session with the original model and session options
session = rt.InferenceSession("NN.onnx", sess_options)

# At this point, the optimized model is saved to "optimized_model.onnx"
print("Optimized model saved.")
