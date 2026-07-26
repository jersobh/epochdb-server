# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Prevent hnswlib build from assuming host-native instruction extensions (e.g. AVX-512)
ENV HNSWLIB_NO_NATIVE=1

# Keep container logs clean: no tqdm "Batches" bars from sentence-transformers,
# no HuggingFace Hub download progress bars.
ENV TQDM_DISABLE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# Install system dependencies required for compiling C/C++ dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Ensure /app is in the Python search path for modules
ENV PYTHONPATH=/app

# Install deps via PyTorch CPU index to avoid CUDA packages, then force a portable
# hnswlib source build so the image boots on CPUs without AVX-512.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "epochdb>=1.8.4" \
    "uvicorn>=0.49.0,<0.50.0" \
    "fastapi>=0.138.1,<0.139.0" \
    "pydantic>=2.13.4,<3.0.0" \
    "sentence-transformers>=5.6.0,<6.0.0" \
    "httpx>=0.28.0" \
    "gunicorn>=22.0.0" \
    "sse-starlette>=2.1.0,<4.0.0" \
    && pip install --force-reinstall --no-cache-dir --no-binary=hnswlib hnswlib \
    && apt-get purge -y build-essential g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the server source code
COPY src/ ./src/

# Expose the API port
EXPOSE 8080

# Start the FastAPI server using Gunicorn process manager with Uvicorn workers, first recursively deleting any stale locks
CMD ["sh", "-c", "find /data -name '*.lock' -delete 2>/dev/null || true && gunicorn src.server:app --workers 1 --timeout 120 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080"]
