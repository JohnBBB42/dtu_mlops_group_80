import os
import pandas as pd
import requests
import streamlit as st
import numpy as np
from google.cloud import run_v2


def get_backend_url():
    """
    Get the URL of the backend service dynamically from Google Cloud Run.
    """
    # Replace with your project ID and region
    project_id = "united-concord-447713-c7"
    region = "europe-west1"

    # Initialize Google Cloud Run client
    parent = f"projects/{project_id}/locations/{region}"
    client = run_v2.ServicesClient()

    # Fetch all services in the specified region
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "bentoml-service":  # Match the backend service name
            return service.uri

    # Fallback to environment variable if the service is not found
    return os.environ.get("BACKEND", None)


def send_features_to_backend(features, backend_url):
    """
    Send features to the backend and get predictions.
    """
    try:
        predict_url = f"{backend_url}/predict"
        response = requests.post(predict_url, json={"features": features}, timeout=60)
        if response.status_code == 200:
            return response.json()  # Parse the JSON response
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None


def validate_csv(data):
    """
    Validate the uploaded CSV to ensure it has exactly 10 columns.
    """
    if data.shape[1] != 10:
        st.error(f"Uploaded file must have exactly 10 columns. Your file has {data.shape[1]} columns.")
        return False
    return True


def main():
    """
    Main Streamlit frontend app.
    """
    st.title("Energy Price Prediction")
    st.markdown(
        "Upload a CSV file containing exactly **10 features per row** for prediction. "
        "The app will send the data to the backend and display predictions."
    )

    # Dynamically fetch the backend URL
    backend_url = get_backend_url()
    if backend_url is None:
        st.error("Backend service not found! Ensure it is deployed in Google Cloud Run.")
        st.stop()

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(data)

            # Validate CSV file structure
            if validate_csv(data):
                if st.button("Predict"):
                    # Convert the CSV data into a list of lists (one list per row)
                    features = data.values.tolist()
                    result = send_features_to_backend(features, backend_url)
                    if result:
                        # Backend returns nested lists, extract predictions
                        predictions = [item[0] for item in result]
                        st.write("Predictions:")
                        st.dataframe(pd.DataFrame(predictions, columns=["Predicted Price"]))
                    else:
                        st.error("No predictions received from backend.")
        except Exception as e:
            st.error(f"Failed to process the uploaded file. Error: {e}")


if __name__ == "__main__":
    main()
