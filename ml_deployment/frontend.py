import os

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
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image}, timeout=42)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # Sort probabilities and take the top 10
            sorted_probs = sorted(
                zip([f"Class {i}" for i in range(len(probabilities))], probabilities),
                key=lambda x: x[1],  # Sort by probability (the second item in the tuple)
                reverse=True,  # Highest probabilities first
            )
            top_ten = sorted_probs[:10]

            # Split the sorted top 10 into separate lists for DataFrame
            classes, probs = zip(*top_ten)

            # Create a bar chart
            data = {"Class": classes, "Probability": probs}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
