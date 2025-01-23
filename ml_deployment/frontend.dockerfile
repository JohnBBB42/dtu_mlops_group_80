# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt update && apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements_frontend.txt /app/requirements_frontend.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

# Copy frontend code
COPY frontend.py /app/frontend.py

# Expose port
EXPOSE 8001

# Entrypoint - Properly resolve the environment variable for PORT
ENTRYPOINT ["sh", "-c", "streamlit run frontend.py --server.port=$PORT --server.address=0.0.0.0"]
