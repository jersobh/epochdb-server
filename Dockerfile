# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for compiling C/C++ dependencies like hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install torch CPU version first to prevent huge CUDA wheel downloads and connection timeouts
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry: Do not create a virtual environment in the container
RUN poetry config virtualenvs.create false

# Install runtime dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the server source code
COPY src/ ./src/

# Expose the API port
EXPOSE 8080

# Start the FastAPI server using Python direct entrypoint
CMD ["python", "src/server.py"]
