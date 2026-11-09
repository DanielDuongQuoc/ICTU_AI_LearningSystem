# ICTU AI Learning System — Module 3 (AI Agent)

AI backend of the ICTU AI Learning System: PDF/DOCX indexing, RAG retrieval and answers, question
generation (single list or exam matrix), and student-answer evaluation (plain JSON or SSE stream).

- **LLM + embeddings:** Google AI Studio **Gemini API** (the only provider).
- **Vector store:** ChromaDB (embedded, persisted on disk).
- **API:** FastAPI. The public contract for other modules is in
  [docs/INTEGRATION_CONTRACT.md](docs/INTEGRATION_CONTRACT.md); security notes in [docs/SECURITY.md](docs/SECURITY.md).

```
client (C# backend / Module 1 / Module 2)
   -> FastAPI (request id, logging, rate limit, body limit, error envelope)
   -> services: documents / rag / questions / evaluations
        -> GeminiEmbeddingService  -> Gemini batchEmbedContents
        -> ChromaDB (cosine, every query filtered by document_id)
        -> GeminiLLMService        -> Gemini generateContent / streamGenerateContent (SSE)
```

All answers are grounded in the uploaded document: if retrieval finds too little relevant text the
service returns `INSUFFICIENT_CONTEXT` and **does not call the LLM**. Sources, ids, `percentage` and
`max_score` are always computed by the backend, never taken from model output.

## 1. Run locally

Requirements: Python 3.11+, a Gemini API key from <https://aistudio.google.com/app/apikey>.

```powershell
cd module3-ai-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # Linux/macOS: cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (never commit .env)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
```

Open <http://localhost:8003/docs> (Swagger) and <http://localhost:8003/health>.

Without `GEMINI_API_KEY` the app still starts, `/health` reports `degraded`, and LLM/embedding endpoints
answer with the service-unavailable error codes (see the contract).

Smoke test against any running instance (uploads a tiny generated PDF, makes ~3 small model calls):

```powershell
python scripts/smoke_test.py --base-url http://localhost:8003
```

## 2. Environment variables

Loaded from the environment / `.env` (`app/core/config.py`). Invalid values stop the app at startup
(`APPLICATION_CONFIGURATION_ERROR`). `.env.example` lists every setting with its default.

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `development` / `staging` / `production` / `test` |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8003` | Docker uses `$PORT` when the platform sets it (Render) |
| `GEMINI_API_KEY` | (empty) | **Secret.** Single key or comma-separated keys (`key1,key2,key3`) for round-robin + 429 failover |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | changing it requires re-indexing documents |
| `LLM_TIMEOUT` / `LLM_CONNECT_TIMEOUT` / `LLM_HEALTH_TIMEOUT` | `120` / `10` / `5` | seconds |
| `LLM_TEMPERATURE` / `LLM_MAX_OUTPUT_TOKENS` | `0.2` / `1024` | |
| `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF_S` | `1` / `0.5` | retries on 429/500/502/503/504 and dropped connections |
| `LLM_MAX_CONCURRENCY` / `LLM_QUEUE_TIMEOUT_S` | `4` / `30` | concurrent Gemini calls; excess waits, then `SERVICE_OVERLOADED` |
| `EMBEDDING_BATCH_SIZE` / `MAX_CONCURRENT_EMBEDDINGS` | `64` / `4` | |
| `REQUEST_TIMEOUT_S` | `300` | whole-request budget, must be >= `LLM_TIMEOUT` |
| `CHROMA_PERSIST_DIRECTORY` / `CHROMA_COLLECTION_NAME` | `./data/chroma` / `learning_documents` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | |
| `TOP_K` / `MAX_TOP_K` / `SIMILARITY_THRESHOLD` / `MAX_CONTEXT_CHARS` | `5` / `20` / `0.3` / `12000` | |
| `MAX_UPLOAD_SIZE_MB`, `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS_PER_DOCUMENT` | `30`, `500`, `2000000`, `3000` | ingestion limits |
| `RATE_LIMIT_PER_MINUTE` | `120` | per client, expensive routes; `0` = off |
| `CORS_ALLOWED_ORIGINS` | (empty) | comma-separated; empty = CORS disabled |
| `LOG_LEVEL` / `LOG_FORMAT` | (auto) / `text` | `DEBUG` is rejected in production |
| `MIN_FREE_DISK_MB` | `100` | readiness fails below this free space |

## 3. Docker

```powershell
copy .env.example .env      # set GEMINI_API_KEY
docker compose up -d --build
curl http://localhost:8003/health
```

The image is multi-stage, runs as a non-root user, never contains `.env`, and has a `/health/live`
healthcheck. It builds from either context: `docker build -f module3-ai-agent/Dockerfile .` (repo root)
or `docker build .` inside `module3-ai-agent/`. Volumes `chroma_data` and `uploads_data` keep indexed
documents across restarts (`docker compose down -v` deletes them).

## 4. Deploy on Render

`render.yaml` (repo root) is a Render Blueprint: Docker runtime, `dockerContext: ./module3-ai-agent`,
health check `/health/live`, `APP_ENV=production`. Set `GEMINI_API_KEY` in the Render dashboard
(`sync: false`, never in the file). Render auto-deploys the `main` branch.

Free-plan notes: the filesystem is ephemeral, so ChromaDB data is lost on every redeploy/restart —
documents must be re-uploaded (attach a persistent disk at `/app/data/chroma` on a paid plan to keep them).
`MIN_FREE_DISK_MB=10` is set because the free instance has little disk.

## 5. Gemini quota notes

- **Multi-key rotation & failover:** `GEMINI_API_KEY` accepts one key or multiple comma-separated keys
  (`GEMINI_API_KEY=key1,key2,key3`). Healthy keys are rotated round-robin across LLM and embedding calls;
  if a key hits `429` (quota / rate limit) or `5xx`, it is placed in a 60s cooldown and the request
  immediately fails over to the next healthy key in the pool.
- Free-tier keys have low per-minute and per-day limits (note: Google AI Studio enforces free-tier quota
  per Google Cloud **Project**, so create each key in a separate Google Cloud Project to multiply quota).
  A `429` from a single key is retried (`LLM_MAX_RETRIES`) after `LLM_RETRY_BACKOFF_S`; if all keys fail,
  the API returns `502` `OLLAMA_GENERATION_ERROR` (legacy code name, see the contract).
- Each upload makes `ceil(chunks / EMBEDDING_BATCH_SIZE)` embedding calls; each RAG answer, question
  generation or evaluation makes 1 embedding call + 1 generation call (+1 if the model returns invalid JSON).
  A matrix request makes one generation call per row.
- Keep `LLM_MAX_CONCURRENCY` low on free keys to avoid bursts of 429s.
- `/health` calls the Gemini models endpoint (no tokens consumed).

## 6. Rubrics, grading and the Grading Agent

Two assessment contexts (contract details in [docs/INTEGRATION_CONTRACT.md](docs/INTEGRATION_CONTRACT.md#rubrics-and-grading)):

| `assessment_context` | Graded by | Result | Status |
|---|---|---|---|
| `homework` | AI (essays) / Grading Agent (code), automatically | **PASS / FAIL** + feedback | `final` (or `needs_review` if flagged) |
| `in_class` | AI proposes, **lecturer decides** | points per criterion + `suggested_total` | always `pending_review` |

Flow: `POST /api/rubrics/generate` (draft) → lecturer edits, `POST /api/rubrics/validate`, sets
`"status": "approved"` → `POST /api/grading/grade` or `/api/grading/batch`. The model only picks one level per
criterion; points, totals and PASS/FAIL are computed by the backend.

**Grading Agent** (code submissions, runs from this folder, uses the same grading core):

```powershell
copy agent\examples\agent.yaml D:\BT03\agent.yaml     # edit paths, accepted_extensions, assessment_context
python -m agent scan  -c D:\BT03\agent.yaml           # who submitted what, which files are accepted / rejected
python -m agent grade -c D:\BT03\agent.yaml           # grade (resumable), then writes results.csv / feedback/
python -m agent export -c D:\BT03\agent.yaml          # re-export from saved progress
```

- One sub-folder or `.zip` per student named `<MSSV>_<Name>` (`naming_pattern`).
- Only files with `accepted_extensions` are graded; `bin/`, `obj/`, `.git/`, binaries, empty and oversized
  files are rejected with a reason (`scan_report.csv`). Zips are extracted safely (no `../` paths, size limits).
- Nothing gradable → `no_valid_files`; submission cut to `max_total_kb` → `truncated`: both go to the lecturer
  (`needs_review`), never auto-FAIL.
- Progress is saved after every student in `output_dir/state.json`; on quota errors the run stops and the same
  command continues where it left off. `mode: api` calls a running AI Service instead of grading in-process.
- Optional tests (`tests_dir`, `test_command`, `docker_image`): run in Docker with no network, read-only mounts and
  CPU/RAM/time limits; the pass/fail counts are given to the grader as evidence and exported. Without Docker the
  submission is flagged `not_executed` (lecturer review).
- Handwritten answers: `POST /api/submissions/handwritten/transcribe` (page images → text, page-1 header masked),
  then grade the text with `submission_kind: "handwritten_transcript"` and `extra_flags` from the transcription.

## 7. Tests and checks

```powershell
pytest -q                          # offline suite, Gemini is faked (no network)
pytest -m integration              # optional: needs a real GEMINI_API_KEY
ruff check --select E9,F app scripts tests agent
python scripts/release_check.py    # static release checks
python scripts/evaluate_rag.py     # offline retrieval-quality metrics (Hit/Precision/Recall@K)
python scripts/prompt_eval.py --repeat 2   # REAL Gemini: grading golden cases (tests/golden), report in reports/
```

Backups of the Chroma directory: `python scripts/backup.py` / `python scripts/restore.py` (stop the API first).
