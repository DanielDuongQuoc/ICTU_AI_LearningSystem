# Deployment

Stack: FastAPI (this module) + Ollama (Qwen + embedding model) + ChromaDB (embedded, on disk).
No other services. Status: pre-1.0 release candidate (`app/version.py`).

## Docker Compose (reference deployment)
```bash
cp .env.example .env            # optional; compose overrides OLLAMA_BASE_URL / CHROMA_PERSIST_DIRECTORY
docker compose config -q        # validate
docker compose build
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
curl http://localhost:8003/health/ready    # 200 once Chroma + Ollama + models are available
python scripts/smoke_test.py               # full flow against the running stack (needs the models)
```
- API container -> Ollama: `http://ollama:11434` (set by compose; never `localhost` inside a container).
- Volumes: `chroma_data` (`/app/data/chroma`), `uploads_data` (temporary upload files), `ollama_data` (models).
- Ollama is published on `127.0.0.1:11434` only. The API publishes `8003`; put a reverse proxy / network
  policy in front of it - Module 3 has **no authentication** (out of scope).
- Reproducibility: Python deps are pinned; the base images are tag-pinned (`python:3.11-slim`) but
  `ollama/ollama:latest` floats. Pin it with `OLLAMA_IMAGE=ollama/ollama:<tag-you-tested>`.
- The container runs as non-root user `app` (uid 10001), has a Python-based healthcheck on `/health/live`
  (so a missing model never kills the container) and `no-new-privileges`.

## Health vs readiness
| Endpoint | Meaning | Calls |
|---|---|---|
| `GET /health/live` | process is running (used by Docker healthcheck) | nothing |
| `GET /health` | always 200; `status` is `ok` or `degraded` + per-dependency detail | Chroma ping, Ollama `/api/tags` |
| `GET /health/ready` (alias `GET /ready`) | 200 only if Chroma, Ollama and the model are available, else 503 | same as `/health` |
None of them runs an LLM generation. Use readiness for load-balancer / deploy gating.

## Timeout chain
`client -> Module 3 -> Ollama`. Module 3 -> Ollama: connect `OLLAMA_CONNECT_TIMEOUT` (10 s), read
`OLLAMA_TIMEOUT` (120 s, also bounds a stalled stream), health probes `OLLAMA_HEALTH_TIMEOUT` (5 s),
one retry on HTTP 502/503/504. Module 3 imposes no total request deadline of its own, so callers must use
a read timeout **larger than** `OLLAMA_TIMEOUT` (+ one retry + retrieval, i.e. >= ~250 s with defaults, or
lower `OLLAMA_TIMEOUT`). The first request after model load is the slowest. Streaming: a stalled upstream
yields `OLLAMA_TIMEOUT` as an SSE `error` followed by `done`; a client disconnect cancels the upstream call.

## Resource limits (all env-configurable)
`MAX_UPLOAD_SIZE_MB` (30), `MAX_QUESTION_CHARS` (2000), `MAX_ANSWER_CHARS` (10000, student and expected
answer), `MAX_TOP_K` (20), `MAX_CONTEXT_CHARS` (12000), `number_of_questions` <= 20, `max_score` <= 100,
`OLLAMA_NUM_PREDICT` (LLM output size). Oversized text -> HTTP 400 `INPUT_TOO_LONG` before any
embedding/LLM call. These length limits were added in 0.14.0 (earlier builds had none): a caller that
sent longer text must raise the limits.

Backpressure (Phase 16): `LLM_MAX_CONCURRENCY` (default 4, `0` = unlimited) caps simultaneous Ollama
generation calls; a call waiting longer than `LLM_QUEUE_TIMEOUT_S` (30) gets HTTP 503 `SERVICE_OVERLOADED`.
`LOG_LEVEL` overrides the default (DEBUG in development, INFO otherwise). `GET /metrics` is for internal
scraping only (no authentication): do not publish it. See [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).

## Graceful shutdown
`docker compose stop` sends SIGTERM to uvicorn (started via `exec`), which stops accepting requests and
lets in-flight ones finish; compose waits `stop_grace_period: 30s`. On shutdown the lifespan closes the
shared Ollama HTTP clients. Streams still running after the grace period are cut by SIGKILL (the client
sees a connection error, no `done` event) - schedule restarts at low traffic.

## Concurrency notes
PDF extraction and the Chroma health ping run in worker threads; Ollama calls are async with shared
clients. Chroma queries/inserts made by the RAG and indexing services are synchronous calls on the event loop (local, short, but they block other requests while running) - a known limitation, not changed in Phase 14. Ollama itself serialises/queues generations per its own settings, so latency grows under load;
Module 3 does not queue or rate-limit requests.

## CORS
Off by default. `CORS_ALLOWED_ORIGINS=https://app.example` enables exactly those origins (no wildcard unless
you write one yourself - do not in production).

## Local (no Docker)
`pip install -r requirements.txt`, run Ollama, `uvicorn app.main:app --port 8003`. See README.
