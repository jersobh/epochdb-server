# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies required for compiling C/C++ dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install dependencies directly via pip using PyTorch CPU wheel index.
# This prevents downloading gigabytes of CUDA packages locked in poetry.lock.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "epochdb==1.2.0" \
    "uvicorn>=0.49.0,<0.50.0" \
    "fastapi>=0.138.1,<0.139.0" \
    "pydantic>=2.13.4,<3.0.0" \
    "sentence-transformers>=5.6.0,<6.0.0" \
    "httpx>=0.28.0"

# Copy the server source code
COPY src/ ./src/

# Expose the API port
EXPOSE 8080

# Start the FastAPI server using Python direct entrypoint
CMD ["python", "src/server.py"]
