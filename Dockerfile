# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ── Stage 2: install dependencies ─────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ── Stage 3: final image ───────────────────────────────────────────────────────
FROM deps AS final
COPY server/    ./server/
COPY raw_docs/  ./raw_docs/

RUN mkdir -p sales_store model_vault

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker for Free F1 plan — reduces memory and startup time
CMD ["uvicorn", "server.startup:retail_application", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
