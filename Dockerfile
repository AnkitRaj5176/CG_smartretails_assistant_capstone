# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# ── Stage 2: install dependencies ─────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')"

CMD ["uvicorn", "server.startup:retail_application", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
