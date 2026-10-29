# ICTU_AI_LearningSystem - Module 3 (AI Agent)
# Multi-stage build: compilers stay in the builder, the runtime image is slim
# and runs as a non-root user.  No .env / secrets are copied into the image.
# Works seamlessly whether Docker build context is repo root (.) or ./module3-ai-agent.

# ---- builder ---------------------------------------------------------
FROM python:3.11.9-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Only needed if a dependency has no prebuilt wheel for the platform.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.tx[t] module3-ai-agent/requirements.tx[t] ./
RUN pip install -r requirements.txt

# ---- runtime ---------------------------------------------------------
FROM python:3.11.9-slim

LABEL description="ICTU AI Learning System - Module 3 (RAG, question generation, evaluation)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=FALSE \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8003 \
    CHROMA_PERSIST_DIRECTORY=/app/data/chroma

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/data/chroma /app/data/ai_versions /app/uploads \
    && chown -R app:app /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app ap[p] module3-ai-agent/ap[p] ./app/

USER app

EXPOSE 8003

# /health always answers 200 (status "degraded" when Ollama/Chroma is down), so a missing
# model never makes the container unhealthy.  Python is used so curl is not required.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request as u; p=os.environ.get('PORT') or os.environ.get('APP_PORT','8003'); u.urlopen('http://127.0.0.1:%s/health/live' % p, timeout=5)"]

# Bind to cloud platform $PORT when provided (e.g. Render), falling back to $APP_PORT (8003).
CMD ["sh", "-c", "exec uvicorn app.main:app --host ${APP_HOST} --port ${PORT:-${APP_PORT}}"]
