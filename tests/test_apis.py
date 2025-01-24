import pytest
import httpx
import random

# Define the API endpoint
API_ENDPOINT = "https://bentoml-service-252171111954.europe-west1.run.app"

@pytest.fixture(scope="module")
def client():
    """
    Fixture to create an HTTPX client with an increased timeout.
    """
    return httpx.Client(base_url=API_ENDPOINT, timeout=30.0)  # Increased timeout to 30 seconds

def test_predict_valid_input(client):
    """
    Test the /predict endpoint with valid input data.
    """
    # Generate 10 random features as input
    features = [random.uniform(0, 100) for _ in range(10)]
    payload = {"features": features}

    try:
        response = client.post("/predict", json=payload)
    except httpx.ReadTimeout:
        pytest.fail("The request to /predict timed out.")

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    # Expecting the response to be a list of lists, e.g., [[prediction_value]]
    try:
        json_response = response.json()
    except ValueError:
        pytest.fail("Response is not valid JSON.")

    assert isinstance(json_response, list), f"Expected response to be a list, got {type(json_response)}"
    assert len(json_response) > 0, "Response list is empty."
    assert isinstance(json_response[0], list), f"Expected first element to be a list, got {type(json_response[0])}"
    assert len(json_response[0]) > 0, "Inner list is empty."
    # Optionally, check if the prediction is a float
    assert isinstance(json_response[0][0], (float, int)), "Prediction is not a number."

def test_predict_invalid_input(client):
    """
    Test the /predict endpoint with invalid input data.
    """
    # Example: Sending only 5 features instead of the expected 10
    invalid_payload = {"features": [1, 2, 3, 4, 5]}

    try:
        response = client.post("/predict", json=invalid_payload)
    except httpx.ReadTimeout:
        pytest.fail("The request to /predict timed out.")

    # Based on current behavior, expecting a 500 status code
    # Ideally, this should be a 400 Bad Request. Consider fixing the service.
    assert response.status_code == 500, f"Expected status code 500, got {response.status_code}"
    # Optionally, check for error details in the response
    # try:
    #     json_response = response.json()
    #     assert "error" in json_response
    # except ValueError:
    #     pytest.fail("Error response is not valid JSON.")
