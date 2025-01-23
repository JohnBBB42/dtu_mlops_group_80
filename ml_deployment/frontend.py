# frontend.py
import os
import numpy as np

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/united-concord-447713-c7/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "bentoml_service":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name

def predict_energy_price(features, backend):
    """Send the features to the backend for prediction."""
    predict_url = f"{backend}/predict"
    payload = {"features": features.tolist()}
    try:
        response = requests.post(predict_url, json=payload, timeout=42)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        st.error("Backend service not found")
        return

    st.title("Energy Price Predictor")

    st.write("Enter the input features for energy price prediction:")

    # Define your actual feature names here
    feature_names = [
        "Feature 1",
        "Feature 2",
        "Feature 3",
        "Feature 4",
        "Feature 5",
        "Feature 6",
        "Feature 7",
        "Feature 8",
        "Feature 9",
        "Feature 10"
    ]
    
    features = []
    for name in feature_names:
        value = st.number_input(f"{name}", value=0.0, format="%.4f")
        features.append(value)

    if st.button("Predict"):
        if len(features) != 10:
            st.error("Please enter all 10 features.")
            return

        result = predict_energy_price(np.array(features), backend=backend)

        if result is not None and "prediction" in result:
            prediction = result["prediction"]
            st.success(f"Predicted Energy Price: {prediction[0]:.2f}")
        else:
            st.error("Failed to get prediction")

if __name__ == "__main__":
    main()
