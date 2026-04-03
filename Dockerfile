# ── base ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── dependencies ─────────────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── final ────────────────────────────────────────────────────────────────────
FROM deps AS final

# Copy only source — no data, no model weights, no secrets
COPY src/           ./src/
COPY config/        ./config/
COPY main.py        ./main.py
COPY api.py         ./api.py

# Non-root user
RUN useradd -m appuser
USER appuser

# API server on port 8000
# Mount model weights at runtime: -v $(pwd)/models:/app/models
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
