from __future__ import annotations

import bentoml
import numpy as np
from onnxruntime import InferenceSession


@bentoml.service(workers=4)
class EnergyPricePredictorService:
    """
    Energy price predictor service using ONNX model.
    """

    def __init__(self) -> None:
        # Load your ONNX model
        self.model = InferenceSession("optimized_model.onnx")

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=128,
        max_latency_ms=1000,
    )
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict energy price given input features.
        The input must be a NumPy array of shape [batch_size, input_size].
        """
        # Ensure the input is 2D: [batch_size, input_size]
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim != 2:
            raise ValueError("Input must be a 2D array (batch_size, input_size).")

        # Perform inference in ONNX Runtime
        output = self.model.run(None, {"input": features.astype(np.float32)})
        return output[0]

