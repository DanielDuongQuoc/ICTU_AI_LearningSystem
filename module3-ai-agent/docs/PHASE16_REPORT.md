# Phase 16 Report

Module 3 (`module3-ai-agent`), post-release operations. Date: 2026-10-24. Version string unchanged
(`0.14.0`; no version bump was requested). No git repository exists, so no commit id.

## 1. Initial Audit

Findings are classified by severity. Items marked NOT VERIFIED could not be tested here (Docker
daemon and Ollama unavailable on the development machine).

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | No operational metrics: only log lines existed | MEDIUM | Added in-process metrics + `/metrics` |
| 2 | No cap on simultaneous Ollama generation calls; overload would queue behind the model and cascade into `OLLAMA_TIMEOUT` | MEDIUM | Added configurable gate, `SERVICE_OVERLOADED` (503) |
| 3 | Log level not configurable independently of `APP_ENV` | LOW | Added `LOG_LEVEL` |
| 4 | No backup retention tooling | LOW | Added `scripts/rotate_backups.py` |
| 5 | No runbook, incident or maintenance documents | MEDIUM | Added (section 10) |
| 6 | Chroma calls in RAG/indexing are synchronous on the event loop | MEDIUM | NOT changed (known limitation, section 12) |
| 7 | `/metrics` and the rest of the API have no authentication | MEDIUM | Documented: internal network only |
| 8 | Dependency vulnerability status unknown | INFORMATIONAL | Scanner not run (section 13) |
| 9 | Existing request-id middleware, structured request/stream/LLM logs, error contract, health/readiness (no LLM call), SSE ordering, backup/restore scripts | INFORMATIONAL | Reviewed, adequate, left as is |

No CRITICAL or HIGH finding. No privacy leak was found in logs (verified by test, section 7).

- **Existing monitoring**: request duration logs, slow-request warning, stream first-token/total latency logs, `/health`, `/health/live`, `/ready`.
- **Existing logging**: stdout, `request_id` on every record via a filter, bodies/prompts never logged.
- **Existing recovery**: checksummed backup and guarded restore scripts; degraded start when Ollama/Chroma are down.
- **Existing operational documentation**: DEPLOYMENT, BACKUP_AND_RECOVERY, SECURITY, RELEASE, HANDOVER.

## 2. Changes Implemented
- `app/core/metrics.py`, `app/api/metrics.py`: bounded-cardinality metrics, `GET /metrics` (excluded from OpenAPI).
- `app/core/middleware.py`: records request count/latency/in-flight/cancellations (existing logging untouched).
- `app/core/error_handlers.py`, `app/services/evaluation_service.py`: count errors by code (HTTP and SSE).
- `app/services/embedding_service.py`, `rag_service.py`, `llm_service.py`: stage timings.
- `app/core/concurrency.py` + `llm_service.py`: LLM concurrency gate wrapping the existing `_chat` / `generate_stream` implementations (renamed `_chat_unlocked`, `_generate_stream_unlocked`; behaviour otherwise identical).
- `app/core/exceptions.py`, `config.py`, `openapi.py`, `.env.example`: `SERVICE_OVERLOADED`, `LLM_MAX_CONCURRENCY`, `LLM_QUEUE_TIMEOUT_S`, `LOG_LEVEL`.
- `scripts/ops_check.py`, `scripts/rotate_backups.py`.

## 3. Metrics and Logging
Metrics (per process, reset on restart): request totals by method/route template/status, request
duration histogram, errors by code, stage histograms (`embedding`, `chroma_query`, `llm_generate`,
`llm_first_token`, `llm_stream_total`), stream outcomes, in-flight, cancellations, LLM
active/waiting/rejected, resident memory (Linux only). Labels never contain `document_id`,
`request_id`, prompts, answers or PDF text (tests assert this). Not measured separately: PDF
processing (it is the upload route's duration), CPU. Dependency availability is reported by `/ready`;
Ollama/Chroma failure counts come from `module3_errors_total` by code. No Prometheus/Grafana/OTel
dependency was added. Logging was audited and kept; only `LOG_LEVEL` was added.

## 4. Concurrency and Reliability
`LLM_MAX_CONCURRENCY` (default 4, 0 = unlimited) and `LLM_QUEUE_TIMEOUT_S` (default 30). Tests show:
excess calls are rejected with `SERVICE_OVERLOADED` (503); slots are released after success,
failure, cancellation and stream close; a queued request proceeds when a cancelled one frees its
slot; peak concurrency never exceeds the limit. **Behaviour change to note**: with the default of 4, a
5th simultaneous generation waits up to 30 s and is then rejected (previously it queued inside
Ollama until `OLLAMA_TIMEOUT`). Set `LLM_MAX_CONCURRENCY=0` to restore the old behaviour. The default
is not derived from measured Ollama capacity (NOT VERIFIED on real hardware).

## 5. Dependency Recovery
Verified with mocked Ollama and real in-memory/persistent ChromaDB: Ollama unavailable/timeout then
recovery on the same service instance; mid-stream timeout and invalid JSON, then recovery; `/ready`
goes 503 -> 200 as Ollama and Chroma recover while `/health/live` stays 200; Chroma failure returns
`CHROMA_ERROR` (503) without leaking internals, then recovers; failed stream returns one `error`
then one `done`, next stream succeeds. Startup with Chroma and/or Ollama down degrades instead of
crashing. Shutdown closes shared HTTP clients, tolerates a failing close, and runs even if the app
body raises. Document lifecycle: deleting one document leaves another intact; failed ingestion leaves
nothing indexed; invalid upload stores nothing; duplicate uploads follow the current behaviour
(distinct ids, not idempotent). Real Ollama restart recovery: NOT VERIFIED.

## 6. Backup and Recovery Drill
Real ChromaDB persistence in an isolated temporary directory (`tests/test_operations_scripts.py`):
add chunks, backup (archive + sha256 verified), delete the data, restore into a new directory, chunk
count matches, retrieval works. Restore refuses without confirmation and on a corrupted archive.
Retention script: dry run deletes nothing; keeps newest valid archives; never counts or deletes a
corrupt archive; ignores foreign files. This is **not** a Docker-volume drill and not production
data (NOT VERIFIED in Docker). No RPO/RTO is claimed.

## 7. Security and Privacy
Verified by test: logs never contain the student answer, the question, or PDF text across upload,
evaluate, stream, retrieve and a validation-error path; request id is preserved from the client header
and shared by response header, error body and log; a forced 500 hides internals and is logged with a
traceback exactly once; `/metrics` shows no ids or content and is not in OpenAPI. `/metrics` has no
authentication: keep it internal. No monitoring data leaves the system (no outbound alerting).
Dependency CVE scanning was not run.

## 8. Tests
- Total collected in the default run: 626 (617 run + 9 integration deselected)
- Passed: 617
- Failed: 0
- Skipped: 0 in the default run; the 9 `integration` tests are deselected by default and skip when Ollama is absent (not run, NOT VERIFIED)
- Lint (`ruff check --select E9,F app scripts tests`): passed
- `scripts/release_check.py`: all 10 items PASS
- New tests: 53 (`test_operational_metrics`, `test_concurrency_limits`, `test_dependency_recovery`, `test_request_logging`, `test_shutdown`, `test_operations_scripts`), all offline with mocks or in-memory/tmp Chroma.

The complete suite was re-run after the last edit: 617 passed, 9 deselected, 0 failed (45.6 s). Lint and `scripts/release_check.py` were re-run afterwards with no failures.

## 9. Performance Results
- **Baseline**: none recorded for real Ollama; `scripts/benchmark.py` and the RAG evaluation were NOT RUN (Ollama unavailable). No comparison with Phase 10-12 baselines is possible.
- **Current (offline harness only)**: real FastAPI app, real in-memory Chroma, fake embedding and fake LLM, single process on a Windows development machine, 5 concurrent rounds of 20 requests per level. Measures framework, middleware and retrieval overhead, **not** model latency.

| Call | c=1 p50 / p95 ms | c=5 p50 / p95 ms | c=10 p50 / p95 ms |
|---|---|---|---|
| `/api/rag/retrieve` | 35 / 43 | 36 / 46 | 20 / 24 |
| `/api/evaluations/evaluate` | 35 / 50 | 32 / 57 | 22 / 46 |

Upload of a 5-page PDF: about 70 ms. Fake-LLM stream first chunk: 43-128 ms (varies by run).
All 1,000+ requests returned 200; in-flight returned to idle (the metrics read itself counts as 1).
A 1,500-request retrieve soak (tracemalloc, GC before each sample) showed Python heap of 139, 143,
143, 159, 159 KB per 300 requests: no sign of a leak, but this is a short run and does not cover
Ollama connections, process RSS or long uptime.
- **Limitations**: no real PDF-processing, embedding, LLM or first-token numbers; no regression comparison; run-to-run variance is visible (stream first chunk). Treat these as a smoke check for crashes, deadlocks and gross leaks only.

## 10. Files Created
- `app/core/metrics.py`, `app/core/concurrency.py`, `app/api/metrics.py`
- `scripts/ops_check.py`, `scripts/rotate_backups.py`
- `tests/test_operational_metrics.py`, `test_concurrency_limits.py`, `test_dependency_recovery.py`, `test_request_logging.py`, `test_shutdown.py`, `test_operations_scripts.py`
- `docs/OPERATIONS_RUNBOOK.md`, `docs/INCIDENT_RESPONSE.md`, `docs/MAINTENANCE.md`, `docs/PHASE16_REPORT.md`

## 11. Files Modified
- `app/main.py`, `app/core/middleware.py`, `app/core/error_handlers.py`, `app/core/exceptions.py`, `app/core/config.py`, `app/core/openapi.py`
- `app/services/llm_service.py`, `embedding_service.py`, `rag_service.py`, `evaluation_service.py`
- `.env.example`, `docs/ERROR_MATRIX.md` (regenerated, 49 rows), `docs/DEPLOYMENT.md`, `docs/BACKUP_AND_RECOVERY.md`, `docs/HANDOVER.md`
- The user edited `tests/test_phase15.py` (collection name and result access); the file passes.

API request/response contracts, existing status codes, error format, SSE events, document
isolation, source and score integrity are unchanged. Additive only: one new error code
(`SERVICE_OVERLOADED`, 503, already within the documented 503 class) and a hidden `/metrics` route.

## 12. Known Limitations
- Metrics are per process, reset on restart, memory is Linux only; no PDF-stage or CPU metric.
- No automatic alerting backend: use `ops_check.py` or your own monitor.
- `/metrics` and the API have no authentication or rate limiting.
- The LLM gate is per process (multiple workers/replicas each have their own limit); embedding is not gated.
- Chroma calls still run synchronously on the event loop (unchanged from earlier phases).
- Uploads are not idempotent; `ollama/ollama:latest` floats unless pinned.
- `llm_first_token` includes time spent waiting for a gate slot.
- `.github/workflows/ci.yml` has not run on GitHub; its local steps pass.

## 13. Unverified Items
- Docker: NOT RUN (daemon unavailable): build, compose, container networking to `ollama:11434`, restart persistence, Docker-volume backup/restore.
- Real Ollama: NOT RUN: model/embedding availability, real recovery after a restart, `pytest -m integration`.
- `scripts/benchmark.py` and `scripts/evaluate_rag.py`: NOT RUN; no RAG-quality or performance regression comparison.
- Dependency vulnerability scan: NOT RUN (tool not installed, needs network); status unknown.
- Alert thresholds and `LLM_MAX_CONCURRENCY` default against real traffic and hardware: NOT VERIFIED.
- Module 1 / Module 2 live integration: PENDING (not present in the repository).

## 14. Final Status
**COMPLETE WITH LIMITATIONS** (offline verification passed; production-environment checks above remain NOT VERIFIED). The Phase 15 release decision (candidate not approved until Docker/Ollama/benchmark/Module 1-2 checks run) is unchanged by this phase.
