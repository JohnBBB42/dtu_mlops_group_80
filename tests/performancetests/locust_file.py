import json
import random
from locust import HttpUser, between, task


def generate_features():
    """Generate a list of 10 random numerical features."""
    return [random.uniform(0, 100) for _ in range(10)]


class BentoMLUser(HttpUser):
    """Locust user class for sending prediction requests to the BentoML server."""

    host = "http://localhost:3000"  # Specify your BentoML service URL here
    wait_time = between(1, 2)  # Wait between 1 to 2 seconds between tasks

    @task
    def send_prediction_request(self):
        """Send a prediction request to the BentoML server."""
        features = generate_features()
        payload = {"features": features}  # Package the features as JSON
        headers = {"Content-Type": "application/json"}
        self.client.post("/predict", json=payload, headers=headers)
