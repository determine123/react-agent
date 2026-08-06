# Agent Project Dockerfile
# Multi-stage build for a minimal production image.

# ----------------------------------------------------------------------
# Stage 1: Builder
# ----------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ----------------------------------------------------------------------
# Stage 2: Runtime
# ----------------------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure local bin is on PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create persistent directories
RUN mkdir -p /app/logs/vector_store /app/logs/agent_notes

# Default env (override via docker-compose or -e flags)
ENV LLM_PROVIDER=deepseek
ENV LLM_MODEL=deepseek-chat
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 7860

# Default: start the API server
CMD ["python", "main.py", "--api", "--host", "0.0.0.0", "--port", "8000"]
