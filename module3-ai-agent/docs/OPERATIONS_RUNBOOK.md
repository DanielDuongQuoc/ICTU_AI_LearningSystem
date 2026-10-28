# Operations Runbook (Phase 16)

Module 3 (`module3-ai-agent`). Written against the real implementation; items needing Docker or a
live Ollama on a production host are marked **NOT VERIFIED** (neither was available on the
development machine). Nothing here changes the public API.

## 1. Monitoring signals

| Signal | Source | Notes |
|---|---|---|
| Liveness | `GET /health/live` | process up, checks nothing else |
| Readiness | `GET /ready` (alias `/health/ready`) | 503 if Chroma, Ollama or the model is down. Never calls LLM generation |
| Metrics | `GET /metrics` (Prometheus text) | in-process, per process, reset on restart |
| Request log | stdout, logger `app.request` | `request_id method path status duration_ms`; slow requests WARN above `SLOW_REQUEST_THRESHOLD_MS` |
| Stream log | logger `app.services.evaluation_service` | outcome, `retrieval_ms`, `first_token_latency_ms`, `total_latency_ms` |
| LLM log | logger `app.services.llm_service` | `llm_time_ms`, `retry_count`; prompt and answer text are never logged |

Metric families (labels are bounded; no `document_id`, `request_id` or free text):

| Metric | Meaning |
|---|---|
| `module3_http_requests_total{method,route,status}` | `route` is the route template, unknown paths are `unmatched` |
| `module3_http_request_duration_seconds` | histogram per method+route (covers upload, question generation, evaluation, SSE until stream end) |
| `module3_errors_total{code}` | every API error by `ErrorCode` (includes `OLLAMA_*`, `CHROMA_ERROR`, `SERVICE_OVERLOADED`; SSE `error` events too) |
| `module3_stage_duration_seconds{stage}` | `embedding`, `chroma_query`, `llm_generate`, `llm_first_token`, `llm_stream_total` |
| `module3_streams_total{outcome}` | `ok`, `error`, `client_disconnected` |
| `module3_http_requests_inflight`, `module3_http_requests_cancelled_total` | concurrency and cancellations |
| `module3_llm_active`, `module3_llm_waiting`, `module3_llm_rejected_total` | LLM concurrency gate state |
| `module3_process_resident_memory_bytes` | Linux only (absent on Windows) |
| `module3_events_total{event,outcome}` | Phase 20: fixed allow-list (`llm_request`, `llm_retry`, `embedding_request`, `rag_retrieval`, `question_generation`, `evaluation`, `document_processing`, `index`, `request_budget`); see OBSERVABILITY.md |
| `module3_stage_duration_seconds{stage}` (Phase 20) | also `pdf_extract`, `chunking`, `index_write`, `context_build`, `json_parse`, `validation` |

> [!WARNING]
> `/metrics` has no authentication (the module has none). Keep it on the internal network or block
> it at the reverse proxy. It is excluded from OpenAPI. It exposes only counts, route templates,
> error codes and timings.

`llm_first_token` includes any wait for an LLM concurrency slot (it is the user-perceived time).
PDF processing time is the upload route's request duration. Per-pipeline-stage CPU is not measured.

## 2. Alerting

There is **no automatic alerting backend** and nothing is sent anywhere. Use the metrics above with
your own monitor, or run `python scripts/ops_check.py --base-url http://host:8000` (read-only, exit
0 ok / 1 alert / 2 unreachable) from cron. Thresholds are options, not policy.

| Alert | Condition (suggested, tune on real traffic) |
|---|---|
| NOT_READY | `/ready` != 200 for several minutes |
| HIGH_5XX_RATIO | 5xx / total above `--max-5xx-ratio` (default 0.05, needs `--min-requests` 20) |
| SLOW_P95 | route p95 above `--max-p95-seconds` (default 60) |
| LLM_OVERLOAD | `module3_llm_rejected_total` above `--max-llm-rejections` (default 10) |
| Ollama failing | rate of `module3_errors_total{code=~"OLLAMA_.*"}` |
| Chroma failing | rate of `module3_errors_total{code="CHROMA_ERROR"}` |
| Streams interrupted | `module3_streams_total{outcome!="ok"}` rate |
| Memory growth | `module3_process_resident_memory_bytes` rising over hours (Linux) |

## 3. Backpressure

`LLM_MAX_CONCURRENCY` (default 4, `0` = unlimited) caps simultaneous Ollama generation calls
(including streams, which hold a slot until they end or the client disconnects). A call that cannot
get a slot within `LLM_QUEUE_TIMEOUT_S` (default 30) gets HTTP 503 `SERVICE_OVERLOADED` (an SSE
`error` event on the stream endpoint). Do not raise `OLLAMA_TIMEOUT` to hide overload: look at
`module3_llm_waiting` and `module3_llm_rejected_total` first. Embedding calls are not gated.

---

## 4. Cases

Always start from the `X-Request-ID` of a failing call and grep logs for it.

### Case A - API does not start
- **Symptoms**: container exits / restart loop; `APPLICATION_CONFIGURATION_ERROR` in the log.
- **Diagnostics**: `docker compose logs api`; `python -c "from app.core.config import Settings; Settings()"`; check the port is free; check the user can write `CHROMA_PERSIST_DIRECTORY` and `uploads/`.
- **Likely causes**: invalid environment value (the module refuses to start by design), port in use, unwritable data directory.
- **Safe remediation**: fix the variable named in the log, free the port, fix directory permissions. A degraded start with Ollama/Chroma down is *expected*; the process still serves `/health/live`.
- **Verification**: `/health/live` 200, then `/ready` 200, then `scripts/smoke_test.py`.
- **Escalation**: still failing after configuration, port and permission checks.

### Case B - Ollama unavailable
- **Symptoms**: `/ready` 503 with `ollama: error`; errors `OLLAMA_UNAVAILABLE` / `OLLAMA_TIMEOUT` / `OLLAMA_MODEL_NOT_FOUND`.
- **Diagnostics**: `docker compose ps`, `docker compose logs ollama`; from the API container `curl http://ollama:11434/api/tags`; confirm `OLLAMA_BASE_URL` uses the service name in Docker (not `localhost`); confirm the model with `ollama list`; compare latency with `OLLAMA_TIMEOUT`.
- **Likely causes**: Ollama stopped or still loading, wrong URL, model not pulled, CPU overload.
- **Safe remediation**: restart only Ollama (`docker compose restart ollama`), `ollama pull <model>`, correct the URL. The API recovers without restart (verified by `tests/test_dependency_recovery.py` with mocked Ollama).
- **Verification**: `/ready` 200; one RAG answer and one stream succeed.
- **Escalation**: repeated timeouts with Ollama healthy and idle.

### Case C - ChromaDB errors
- **Symptoms**: `CHROMA_ERROR` (503); `/ready` shows `chroma: error`.
- **Diagnostics**: `df -h` / volume free space; permissions on the persistence path; logs for `ChromaDB`; check the collection name equals `CHROMA_COLLECTION_NAME`.
- **Likely causes**: disk full, read-only volume, wrong path or collection name, corrupted files.
- **Safe remediation**: free disk, fix permissions or path. **Do not delete the data directory or the collection to try to fix it.** If corruption is confirmed, stop the API, copy the directory aside, and restore the latest valid backup into a fresh directory (BACKUP_AND_RECOVERY.md).
- **Verification**: `/ready` 200; retrieval on a known `document_id` returns that document only.
- **Escalation**: any suspected data loss (treat as SEV-1, see INCIDENT_RESPONSE.md).

### Case D - RAG returns irrelevant results
- **Symptoms**: wrong or empty `sources`, frequent `INSUFFICIENT_CONTEXT`.
- **Diagnostics**: confirm the `document_id` is the intended one (`GET /api/documents/{id}`, chunk count > 0); inspect chunks and metadata; confirm `OLLAMA_EMBEDDING_MODEL` is the model used at indexing time; check `TOP_K` and `SIMILARITY_THRESHOLD`; run `python scripts/evaluate_rag.py` and compare with the recorded baseline.
- **Likely causes**: wrong id, document not ingested (scanned PDF), embedding model changed after indexing, threshold too strict.
- **Safe remediation**: re-upload the document (gets a new id, not idempotent), restore the original embedding model, adjust threshold deliberately. Do not just raise `TOP_K`.
- **Verification**: evaluation dataset metrics back to baseline; isolation tests green.
- **Escalation**: results from another document (release blocker, SEV-1).

### Case E - Streaming interrupted
- **Symptoms**: SSE ends with an `error` event, or `module3_streams_total{outcome="client_disconnected"}` rises.
- **Diagnostics**: Ollama logs; `OLLAMA_TIMEOUT` vs generation time; `module3_streams_total`, `module3_llm_waiting`; the stream log line (`outcome=`); client or proxy idle timeout (disable proxy buffering, SSE needs `X-Accel-Buffering: no`, already sent).
- **Likely causes**: Ollama restart, read timeout, proxy timeout, client disconnect, `SERVICE_OVERLOADED`.
- **Safe remediation**: fix the specific cause; clients may retry a failed stream (evaluation is read-only, no duplicate data).
- **Verification**: a stream shows `result` once, then `done` once, no `error`; `module3_llm_active` returns to 0.
- **Escalation**: `module3_llm_active` stays above 0 with no traffic (slot leak).

### Case F - Data missing after restart
- **Symptoms**: `DOCUMENT_NOT_FOUND` for documents that existed.
- **Diagnostics**: `docker compose config` and `docker inspect` to confirm the Chroma path is a volume or bind mount; compare `CHROMA_PERSIST_DIRECTORY`; check the container is not started with an anonymous volume or a different project name.
- **Likely causes**: no persistent volume, `docker compose down -v`, project renamed, path changed.
- **Safe remediation**: reattach the original volume; otherwise restore from backup. **Never run `docker compose down -v` while investigating.**
- **Verification**: known `document_id` retrievable after `docker compose restart` (**NOT VERIFIED in Docker here**; persistence across a PersistentClient reopen is covered by tests).
- **Escalation**: confirmed loss with no usable backup (SEV-1).

### Case G - Latency increased
- **Symptoms**: SLOW_P95 alert, slow-request warnings.
- **Diagnostics**: compare `module3_stage_duration_seconds` by stage (`embedding`, `chroma_query`, `llm_generate`, `llm_first_token`); `module3_llm_waiting`; host CPU/RAM and Ollama load; run `scripts/benchmark.py` under controlled load.
- **Likely causes**: model slower than before (CPU only, other load), queueing behind the concurrency cap, large documents/answers, disk pressure for Chroma.
- **Safe remediation**: address the slow stage. Do not raise timeouts or concurrency before the cause is known; lower or raise `LLM_MAX_CONCURRENCY` only based on measured Ollama capacity.
- **Verification**: p95 back under the threshold for the same workload.
- **Escalation**: slowdown with no load and no configuration change.

---

## 5. Phase 20 incident scenarios

Every scenario: start from `X-Request-ID`, check `GET /metrics`, then `python -m scripts.check_reliability --url http://host:8000/metrics`.
Alert conditions referenced here are defined in OBSERVABILITY.md. Verification commands marked **NOT VERIFIED** were not run against a real deployment.

### Scenario 1 - Ollama unavailable
- **Symptoms**: `/ready` 503 (`ollama: error`); `OLLAMA_UNAVAILABLE` rising; `llm_request{outcome="error"}` rising.
- **Checks**: `curl $OLLAMA_BASE_URL/api/tags`; `docker compose ps/logs ollama` (NOT VERIFIED here); URL uses the service name in Docker.
- **Likely cause**: Ollama stopped/loading, wrong URL, model not pulled.
- **Immediate action**: restart only Ollama; `ollama pull <model>`. Do not restart the API (it recovers by itself; `tests/reliability/test_ollama_failure.py`).
- **Recovery**: next request succeeds without restart.
- **Verification**: `/ready` 200; one evaluation succeeds; `llm_request{outcome="ok"}` increments.
- **Escalation**: refused connections while Ollama reports healthy.

### Scenario 2 - Ollama timeouts increase
- **Symptoms**: `OLLAMA_TIMEOUT` rising; `slow_llm` warnings (logger `app.slow`); `llm_request{outcome="timeout"}`; `request_budget{outcome="exhausted"}`.
- **Checks**: `module3_llm_waiting`/`module3_llm_active`; host CPU/RAM; prompt/context size (`context_build`); model size.
- **Likely cause**: CPU-bound model, too many concurrent generations, oversized context.
- **Immediate action**: reduce load or `LLM_MAX_CONCURRENCY` to measured capacity. Timeouts are NOT retried by design; do not raise `OLLAMA_TIMEOUT` without raising `REQUEST_TIMEOUT_S` (startup validates `REQUEST_TIMEOUT_S >= OLLAMA_TIMEOUT`).
- **Recovery**: timeouts stop when load/capacity matches.
- **Verification**: timeout share of `llm_request` back under the SLO_SLA.md target.
- **Escalation**: timeouts at idle.

### Scenario 3 - ChromaDB unavailable
- **Symptoms**: `CHROMA_ERROR`; `/ready` `chroma: error`; `rag_retrieval{outcome="error"}`; `index{outcome="failed"}`.
- **Checks**: disk space, permissions, path, collection name (see Case C).
- **Likely cause**: disk full, read-only volume, wrong path/collection.
- **Immediate action**: fix disk/permissions. **Never delete the data directory.** Failed uploads clean their own partial vectors (`index{outcome="cleaned_up"}`).
- **Recovery**: queries succeed once Chroma is back; re-upload any document whose upload returned an error.
- **Verification**: `/ready` 200; retrieval on a known document.
- **Escalation**: any suspicion of data loss (SEV-1, INCIDENT_RESPONSE.md).

### Scenario 4 - API error rate increases
- **Symptoms**: 5xx share of `module3_http_requests_total` up; Availability row of `check_reliability` FAIL.
- **Checks**: `module3_errors_total` by `code` (which dependency?); `RATE_LIMITED`/`ADMIN_AUTH_REQUIRED` indicate abuse or misconfigured clients (Phase 19), not outage.
- **Likely cause**: a dependency (Ollama/Chroma), overload (`SERVICE_OVERLOADED`), or a bad AI version.
- **Immediate action**: follow the scenario for the dominant error code.
- **Recovery/Verification**: 5xx share back under target for a full window.
- **Escalation**: `INTERNAL_SERVER_ERROR` (unhandled exception) persists: collect request ids.

### Scenario 5 - Latency spike
- **Symptoms**: p95 up; `slow_request` warnings include `slowest_stage=<stage>:<ms>`.
- **Checks**: the named stage (`pdf_extract`, `embedding`, `chroma_query`, `llm_generate`...); `module3_stage_duration_seconds`.
- **Likely cause**: see Case G; slow LLM is the usual one.
- **Immediate action**: fix the slow stage. Do not raise timeouts first.
- **Recovery/Verification**: stage histogram back to normal.
- **Escalation**: slowdown with no load/config change.

### Scenario 6 - Disk nearly full
- **Symptoms**: Chroma write errors, `index{outcome="failed"}`, upload 5xx.
- **Checks**: free space on the Chroma volume, `uploads/`, backups, container logs. (`/ready` reports `storage: error` when free space on the data directory is below `MIN_FREE_DISK_MB`.)
- **Likely cause**: old backups, log growth, large PDFs.
- **Immediate action**: rotate backups (`scripts/rotate_backups.py`), rotate logs. Never delete the Chroma directory.
- **Recovery/Verification**: space freed; a test upload returns `indexed` and the document is retrievable.
- **Escalation**: Chroma reports errors after space is freed.

### Scenario 7 - Memory grows unusually
- **Symptoms**: `module3_process_resident_memory_bytes` rising over hours (Linux only).
- **Checks**: concurrent uploads (`MAX_UPLOAD_SIZE_MB`, upload slots), `module3_http_requests_inflight` not returning to 0, `module3_llm_active` stuck above 0 (slot leak).
- **Likely cause**: large concurrent uploads, leaked streams, Ollama sharing the host.
- **Immediate action**: restart the API container (state is on disk; in-process metrics reset); lower upload limits.
- **Recovery/Verification**: memory plateaus; `inflight`/`llm_active` return to 0 at idle.
- **Escalation**: growth with constant load = suspected leak (NOT VERIFIED either way: no long-run soak test was possible).

### Scenario 8 - AI quality regression
- **Symptoms**: lower quality-gate/eval scores or user complaints after a model/prompt change.
- **Checks**: log fields `config_version` / `model` in `request completed` and pipeline traces; `python scripts/quality_gate.py`; compare with the Phase 17 baseline.
- **Likely cause**: new model/prompt/RAG config, or embedding model changed after indexing.
- **Immediate action**: if tied to a version, roll back (Scenario 10).
- **Recovery/Verification**: gate PASS again on the same datasets. Note quality and latency trade off: judge both (RELIABILITY.md).
- **Escalation**: regression with no config change.

### Scenario 9 - SSE streaming failures
- **Symptoms**: `module3_streams_total{outcome!="ok"}` up; clients see `error` then `done`.
- **Checks**: Case E; `OLLAMA_*` codes in `module3_errors_total`; proxy buffering/timeouts; `STREAM_MAX_DURATION_S`.
- **Likely cause**: Ollama timeout/restart, overload, proxy idle timeout, client disconnects.
- **Immediate action**: fix the dependency; clients may simply retry (read-only, one generation per request).
- **Recovery/Verification**: `result` once then `done` once; `module3_llm_active` returns to 0.
- **Escalation**: `llm_active` stuck above 0.

### Scenario 10 - Bad AI version, rollback
- **Symptoms**: errors/latency/quality worsen right after a promotion (audit trail in the governance repository).
- **Checks**: `GET` active version and audit events (AI_VERSION_GOVERNANCE.md); `config_version` in logs vs the time the degradation started.
- **Likely cause**: the promoted version.
- **Immediate action**: roll back to the previous approved version per MODEL_PROMOTION_ROLLBACK.md (refuses never-approved versions and models that are no longer installed).
- **Recovery**: effective config returns to the previous version without redeploy.
- **Verification**: new requests log the previous `config_version`; error/latency back to baseline.
- **Escalation**: rollback refused or fails.
