# Module 3 - Handover (Release Candidate)

Status: **Release Candidate**. Only facts verified in the repository or by a test run are listed;
anything not verified in this environment is marked *NOT VERIFIED*.

## 1. Purpose
AI agent for ICTU_AI_LearningSystem: ingest a PDF, index it, answer questions grounded in it,
generate quiz questions and evaluate student answers (also as an SSE stream). LLM = Ollama + Qwen.

## 2. Architecture
```
PDF -> extraction -> cleaning -> chunking -> embedding (Ollama) -> ChromaDB
    -> RAG retrieval -> context -> Ollama/Qwen -> question generation / evaluation / streaming
```
Layers: `app/api` (thin routers) -> `app/services` (business logic) -> `app/repositories`
(ChromaDB only) / LLM + embedding services (Ollama only). Schemas: `app/schemas` (Pydantic v2).
Config: `app/core/config.py` (env / `.env`). Prompts: `app/core/prompts.py`.
Backend-owned fields: `document_id`, `chunk_id`, `page`, retrieval `score`, `sources`, `percentage`,
`max_score`, `question_id` - LLM-supplied values for these are ignored.

## 3. Requirements
Python 3.11+, Ollama with the chat and embedding models pulled, Docker (optional).
`pip install -r requirements.txt` (versions pinned).

## 4. Local setup
```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                # defaults work for local Ollama
uvicorn app.main:app --port 8003
```
Swagger: `http://localhost:8003/docs`, ReDoc: `/redoc`, schema: `/openapi.json`.

## 5. Docker
```bash
docker compose config
docker compose build
docker compose up -d
docker compose ps
```
Inside compose the API reaches Ollama at `http://ollama:11434` (compose overrides `.env`);
never `localhost`. Chroma data and Ollama models live in named volumes. `docker compose down`
keeps volumes. *Docker build/up was NOT VERIFIED in the Phase 11/12 environment (no daemon).*

## 6. Ollama / Qwen
```bash
ollama pull qwen2.5:7b          # = OLLAMA_MODEL
ollama pull nomic-embed-text    # = OLLAMA_EMBEDDING_MODEL
```
Smaller machine: set `OLLAMA_MODEL=qwen2.5:3b` and pull that tag.

## 7. Environment variables
See `.env.example` (no secrets) and README section 7. Key ones: `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_TIMEOUT`, `OLLAMA_CONNECT_TIMEOUT`,
`OLLAMA_HEALTH_TIMEOUT`, `TOP_K`, `MAX_TOP_K`, `SIMILARITY_THRESHOLD`, `MAX_CONTEXT_CHARS`,
`MAX_UPLOAD_SIZE_MB`, chunking settings.

## 8. API (frozen contract)
| Method | Path |
|---|---|
| GET | `/health` (also `/health/live`, `/health/ready`) |
| POST | `/api/documents/upload` |
| GET / DELETE | `/api/documents/{document_id}` |
| POST | `/api/documents/{document_id}/search` |
| POST | `/api/rag/retrieve`, `/api/rag/answer` |
| POST | `/api/questions/generate` |
| POST | `/api/evaluations/evaluate`, `/api/evaluations/evaluate/stream` |

All errors: `{"error": {"code", "message", "request_id"}}`. `X-Request-ID` is echoed (invalid or
missing -> new UUID). Request/response schemas: `/docs`.

## 9. Tests
```bash
pytest -v                    # offline (Ollama/Docker not needed); integration deselected
pytest -v -m integration     # needs live Ollama + models; skipped otherwise
```
Offline tests use a real in-memory ChromaDB with fake embedding and fake LLM (`tests/fakes.py`).

## 10. Benchmark / 11. RAG evaluation
```bash
python scripts/benchmark.py            # needs Ollama; results are hardware dependent
python scripts/evaluate_rag.py         # Hit@K / Precision@K / Recall@K on the bundled dataset
```
Result files are git-ignored. Numbers are only reported when actually run.

## 12. Troubleshooting
README section 18. Most common: Ollama not running (`/health` -> `degraded`), model not pulled
(`OLLAMA_MODEL_NOT_FOUND`), first request slow (model load; raise `OLLAMA_TIMEOUT`).

## 13. Known limitations
- Requires Ollama; quality/latency depend on model and hardware.
- No authentication/authorisation (out of scope).
- Single ChromaDB collection on local disk; no horizontal scaling.
- Streaming validation problems arrive as SSE `error` events (HTTP 200).
- Scanned (image-only) PDFs are rejected (`NO_EXTRACTABLE_TEXT`); no OCR.

## 14. Integration
See [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md).

## Future improvements (not implemented)
Authentication, OCR, document listing, persisted evaluation history, CI pipeline.


---

## Phase 15 additions (release candidate handover)

**Release information**: version 0.14.0 (pre-1.0), status *release candidate NOT approved* until the unverified items in
[FINAL_RELEASE_REPORT.md](FINAL_RELEASE_REPORT.md) are run (Docker deployment, real Ollama, benchmark/RAG regression).

| Topic | Where / summary |
|---|---|
| Responsibility | Module 3 owns PDF indexing, retrieval, question generation and answer evaluation; it is the evaluation authority (`percentage`, `sources`, `max_score` are backend-computed) |
| Module 1 integration | [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md) - Module 1 contract TBD; `document_id` is generated by Module 3 |
| Module 2 integration | same document - request/response fields are Module 3's; Module 2 contract TBD |
| RAG | per-`document_id` retrieval, `TOP_K`/`MAX_TOP_K`, `SIMILARITY_THRESHOLD`, `MAX_CONTEXT_CHARS`; insufficient context -> `INSUFFICIENT_CONTEXT`, no LLM call |
| Question generation | types short_answer / multiple_choice / true_false / essay; dedupe; backend `question_id` and `source_chunks` |
| Evaluation | LLM returns score/correct/feedback/strengths/missing_points/errors; backend validates 0 <= score <= max_score |
| Streaming | SSE `status -> token -> result -> done` (`error -> done` on failure); consumers parse `data: {...}` frames |
| Errors | [ERROR_MATRIX.md](ERROR_MATRIX.md) (checked against the code by a test) |
| Deployment / Docker | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Backup / recovery | [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md) (`scripts/backup.py`, `scripts/restore.py`) |
| Security | [SECURITY.md](SECURITY.md) |
| Release process, rollback, checklist | [RELEASE.md](RELEASE.md) |
| Tools | `scripts/release_check.py`, `scripts/smoke_test.py`, `scripts/benchmark.py`, `scripts/evaluate_rag.py` |
| CI | `../.github/workflows/ci.yml` (relative to the module: project root) |

## Phase 16 additions (operations)

| Item | Where |
|---|---|
| Metrics (`GET /metrics`, internal, hidden from OpenAPI) | `app/core/metrics.py`, `app/api/metrics.py` |
| LLM concurrency cap, `SERVICE_OVERLOADED` (503) | `app/core/concurrency.py`; `LLM_MAX_CONCURRENCY`, `LLM_QUEUE_TIMEOUT_S` |
| `LOG_LEVEL` override | `app/core/config.py` |
| Runbook, incidents, maintenance | [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md), [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md), [MAINTENANCE.md](MAINTENANCE.md) |
| Ops scripts | `scripts/ops_check.py`, `scripts/rotate_backups.py` |
| Report | [PHASE16_REPORT.md](PHASE16_REPORT.md) |