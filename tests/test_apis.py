from fastapi.testclient import TestClient
from ml_deployment.bentoml_service import EnergyPricePredictorService
import random

# Initialize the BentoML service and get the ASGI app
service = EnergyPricePredictorService()
client = TestClient(service.asgi_app)


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    # Adjust the expected response based on your service's actual root response
    # For example:
    # assert response.json() == {"message": "Welcome to the Energy Price Predictor API!"}
    # If there's no root endpoint defined, you might skip or adjust this test.


def test_predict():
    """Test the predict endpoint with valid input."""
    features = [random.uniform(0, 100) for _ in range(10)]
    payload = {"features": features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Assuming the model returns a numerical prediction
    # Adjust the assertion based on your actual output
    assert "output" in response.json()
    assert isinstance(response.json()["output"], list)
    assert len(response.json()["output"]) > 0


def test_predict_invalid_input():
    """Test the predict endpoint with invalid input."""
    # Example: Sending only 5 features instead of 10
    payload = {"features": [1, 2, 3, 4, 5]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Expecting a Bad Request
    # Optionally, check for specific error messages
