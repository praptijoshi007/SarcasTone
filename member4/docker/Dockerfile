# ==============================================================================
# SarcasTone Backend Service Container
# Python 3.11 with FastAPI and Multimodal Dependencies
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system audio dependencies (libsndfile for audio processing, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir fastapi uvicorn python-multipart pydantic PyYAML pandas numpy

# Copy codebase
COPY . .

# Set environment
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "member4.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
