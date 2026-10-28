# Phase 20 Report - Observability, SLO/SLA, Reliability & Failure Testing

## 1. Status: **PARTIAL**
All code, tests, docs and offline failure/recovery testing are done and green (956 passed, 9 deselected). It is PARTIAL because
no SLO, availability, real-LLM performance, Docker/SIGTERM or real-dependency-outage claim can be verified here (no Docker, no
Ollama, no production traffic). See section 10.

## 2. Audit summary (what already existed, reused)
Phase 16 metrics registry + `/metrics`, request-id middleware and log filter, `execution_trace` (AI version fields), LLM
concurrency gate, `/health`, `/ready`, shutdown client closing, SSE `error`->`done` contract, Phase 18 lock + revision guard,
Phase 19 error codes. Gaps found: no per-stage metrics for PDF/chunk/index/context/parse/validation; no dependency event
counters; no JSON/redacted logging; no whole-request time budget (worst case several sequential 120 s calls with retry);
no embedding retry; connection resets not retried; no cleanup/verification of a partial index; no storage readiness.

## 3. Changes
| File | Change | Reason |
|---|---|---|
| `app/core/request_context.py` | deadline + stage-time contextvars | request budget, slowest-stage hint |
| `app/core/timeouts.py` (new) | `call_timeout`, `can_retry`, `timeout_chain` | timeout chain |
| `app/core/config.py`, `.env.example` | `SLOW_STAGE_THRESHOLD_MS`, `SLOW_LLM_THRESHOLD_MS`, `REQUEST_TIMEOUT_S`, `MIN_FREE_DISK_MB`, `LOG_FORMAT` + validation | configurable, validated |
| `app/core/metrics.py` | new stages, bounded `module3_events_total`, slow-stage warnings | observability, bounded cardinality |
| `app/core/logging.py` | `RedactionFilter`, optional `JsonFormatter` | no secrets in logs, structured logs |
| `app/core/middleware.py` | deadline, `config_version` + `slowest_stage` in logs, question/evaluation events | tracing |
| `app/services/llm_service.py` | budget clamp, connection-reset retry, retry guard, events, `json_parse` stage | retry/timeout policy |
| `app/services/embedding_service.py` | budget clamp, one bounded retry, events | same policy as LLM |
| `app/services/document_service.py`, `vector_service.py`, `rag_service.py`, `evaluation_service.py` | stage timings, rag events | pipeline tracing |
| `app/api/documents.py` | document/index events; post-write chunk-count verification; cleanup of partial index on any failure incl. cancellation | no false "indexed" |
| `app/api/health.py` | `check_storage_health`; additive `services.storage`; `/ready` 503 on low disk | storage readiness |
| `tests/test_health.py` | one assertion made additive-tolerant | contract keys unchanged, new key added |
| `tests/reliability/` (7 files + conftest) | 78 tests | failure injection, recovery, concurrency, SSE, observability, SLO script |
| `scripts/check_reliability.py` (new) | SLO table with Target/Observed/Status/Data source | INCONCLUSIVE by default |
| `scripts/release_check.py` | 5 new required docs | |
| `docs/OBSERVABILITY.md`, `SLO_SLA.md`, `RELIABILITY.md`, `FAILURE_RECOVERY.md` (new), `OPERATIONS_RUNBOOK.md` (+10 scenarios) | documentation | |

CI: `../.github/workflows/ci.yml` already runs `pytest -q`, which includes `tests/security` and `tests/reliability`; all are offline (no Ollama).

## 4. Metrics implemented
`module3_events_total{event,outcome}`: `llm_request`, `llm_retry`, `embedding_request`, `embedding_retry`, `rag_retrieval`,
`question_generation`, `evaluation`, `document_processing`, `index`, `request_budget` (fixed allow-list).
New stages in `module3_stage_duration_seconds`: `pdf_extract`, `chunking`, `index_write`, `context_build`, `json_parse`,
`validation`. Labels never contain request/document ids, filenames or versions (tested). AI-version identity is in logs.

## 5. SLO
| SLO | Target (proposal) | Observed | Status |
|---|---|---|---|
| Availability | >= 99.0% | NO PRODUCTION TRAFFIC DATA | NOT VERIFIED |
| Retrieve p95 | <= 2 s | harness only (fake embedding): <= 50 ms, 40 requests | PASS (harness only) - not production evidence |
| Question generation / evaluation p95 | <= 90 s / <= 60 s | not measurable offline | INCONCLUSIVE |
| LLM success, doc processing, index write, stream completion | 95-99% | too few samples | INCONCLUSIVE |
| Retrieve p95 vs baseline | <= 1.5x | no baseline file | INCONCLUSIVE |
Error budget concept documented in `SLO_SLA.md`; consumed budget unknown.

## 6. Failure tests (all mocks; no real outage)
| Scenario | Expected | Actual | Status |
|---|---|---|---|
| Ollama refused / timeout | specific code, no retry | as expected | PASS (mock) |
| Ollama 404 model missing | `OLLAMA_MODEL_NOT_FOUND`, no retry | as expected | PASS (mock) |
| Ollama malformed/unexpected body | `OLLAMA_INVALID_RESPONSE` | as expected | PASS (mock) |
| 502/503/504, connection reset | one retry then success; bounded when persistent | as expected (persistent 503 surfaces `OLLAMA_UNAVAILABLE` or `OLLAMA_GENERATION_ERROR`) | PASS (mock) |
| 400/401/403/422/500 | not retried | as expected | PASS |
| Budget exhausted / too little for retry | no call / no retry | as expected | PASS |
| Chroma query failure (3 exception types) | controlled 5xx, no sources | as expected | PASS (mock) |
| Embedding failure on upload | nothing indexed | as expected | PASS |
| Chroma write fails after partial write (2 types) | partial vectors removed, error response | as expected | PASS (mock) |
| Cleanup also fails | original error preserved | as expected | PASS |
| Count mismatch after write | not reported indexed | as expected | PASS |
| PDF parser exception; invalid/empty/corrupt PDF | rejected, nothing stored | as expected | PASS |
| Filesystem error (temp dir `OSError`) | 500 without internals | as expected | PASS (injected error, not a real full disk) |
| Delete failure | data intact, retry works | as expected | PASS |
| Concurrent retrieve/evaluate/delete/upload | consistent, no corruption, excess uploads 503 | as expected | PASS |
| SSE exception / budget exhaustion / reconnect / cancellation | one `error`, one `done`, no leak, one generation per request | as expected | PASS (mock) |

## 7. Recovery tests
Ollama down -> up (service level), Chroma query fail -> ok, LLM fail -> ok (evaluation), embedding fail -> ok, Chroma write fail
-> ok, PDF parser fail -> ok, delete fail -> ok, low disk -> ok (`/ready` 503 -> 200, `/health/live` always 200),
SSE failure -> next stream OK. All succeeded without restart. These are mock-based.

## 8. Performance
Baseline vs current: **INCONCLUSIVE** (no baseline file for this script; Phase 16 harness retrieve p50/p95 35/43 ms and
Phase 20 harness p95 <= 50 ms use fakes, same machine, bucket-coarse; no regression conclusion drawn). No real LLM numbers.
Added overhead of stage/event instrumentation was not separately benchmarked.

## 9. Remaining risks
Chroma calls are synchronous on the event loop; in-process limits/metrics (single instance only); no hard cancellation at
the request deadline (running calls end at their clamped timeout); no startup log of the active AI version/timeout chain;
no automatic metadata<->Chroma reconciliation job; uploads are not idempotent by design; the unauthenticated `/metrics`
must stay internal; `/health` `services` gained an additive `storage` key (clients with strict schemas should tolerate it).

## 10. NOT VERIFIED
Any SLO/availability value; real Ollama latency, outage or timeout behaviour; real ChromaDB corruption or full disk;
Docker build, container SIGTERM/graceful shutdown with in-flight SSE; multi-replica behaviour; long-run memory growth;
JSON log output on a real deployment; runbook procedures on a real host; Prometheus/Grafana (specified only).
