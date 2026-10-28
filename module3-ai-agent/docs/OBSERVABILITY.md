# Observability (Phase 20)

Upgrades the Phase 16 metrics/logging; there is no second system. Everything is in-process, no new dependency.
**Dashboards and alerts below are specifications. No production dashboard or alerting backend exists.**

## 1. Request lifecycle
`RequestIdMiddleware` (`app/core/middleware.py`) assigns/reuses `X-Request-ID`, stores it in a contextvar
(`app/core/request_context.py`), and every log line gets it via a logging filter. It is echoed in the response header
and in every error body. Services never create their own ids. The middleware also sets the request deadline
(`REQUEST_TIMEOUT_S`) and a per-request stage-time map.

`request completed` log: `request_id method endpoint status duration_ms error_code config_version`.
Slow requests (`SLOW_REQUEST_THRESHOLD_MS`) log WARNING with `slowest_stage=<stage>:<ms>`. Payloads are never logged.

## 2. Pipeline stages (`module3_stage_duration_seconds{stage}`)
| Stage | Where |
|---|---|
| `pdf_extract`, `chunking` | `document_service` |
| `embedding` | `embedding_service` |
| `index_write` | `vector_service` |
| `chroma_query` | `rag_service` retrieval |
| `context_build` | `rag_service` |
| `llm_generate`, `llm_first_token`, `llm_stream_total` | `llm_service` |
| `json_parse` | `llm_service.generate_json` |
| `validation` | `evaluation_service.validate_llm_result` |

Slow-stage WARNINGs (logger `app.slow`): `slow_llm` (> `SLOW_LLM_THRESHOLD_MS`, default 60000), `slow_retrieval`,
`slow_pdf`, `slow_embedding`, `slow_index`, `slow_stage` (> `SLOW_STAGE_THRESHOLD_MS`, default 5000).

## 3. Event counters (`module3_events_total{event,outcome}`)
Names and outcomes are a fixed allow-list in `app/core/metrics.py`; anything else is ignored.

| event | outcomes |
|---|---|
| `llm_request` | ok, error, timeout, rejected |
| `llm_retry`, `embedding_retry` | transient, connection_reset |
| `embedding_request` | ok, error, timeout |
| `rag_retrieval` | ok, no_context, error |
| `question_generation`, `evaluation` | ok, error (5xx = error, 4xx not counted) |
| `document_processing` | ok, failed, timeout, limit_exceeded |
| `index` | ok, failed, cleaned_up, verification_failed |
| `request_budget` | exhausted, retry_skipped |

Plus Phase 16 metrics: requests total/duration, errors by `ErrorCode`, streams, inflight, LLM gate gauges.
Security signals (Phase 19) are the existing `module3_errors_total{code="RATE_LIMITED"|"ADMIN_AUTH_REQUIRED"|...}`.

## 4. Cardinality rules
Allowed label values: HTTP method, **route template**, status, fixed stage names, fixed event/outcome pairs, `ErrorCode`.
Never: `request_id`, `document_id`, student id, filename, prompt/answer text, model/version strings.
`tests/reliability/test_observability.py` asserts ids and filenames never appear in `/metrics`.
Model / version / prompt / rag-fingerprint belong in logs and traces (`config_version` in `request completed`, full set in
`execution_trace` logs), not in labels.

## 5. Logging
* `LOG_FORMAT=text` (default) or `json` (fields: `timestamp level service module request_id message` + lifted `key=value`).
* `RedactionFilter` is always on: `token=`, `api_key=`, `password=`, `secret=`, `Authorization: Bearer ...` are masked
  (`first_token_ms`, `max_tokens` are not touched).
* Not logged by design: PDF text, full prompts, student answers, secrets.

| Level | Use |
|---|---|
| DEBUG | development only, off by default |
| INFO | request completed, startup/shutdown, major step |
| WARNING | slow request/stage, retry, degraded dependency, insufficient context, budget retry skipped |
| ERROR | request failure, Ollama/Chroma/parser failure, unexpected exception |
| CRITICAL | not used for per-request failures |

## 6. Health
`/health/live`: process only. `/ready` and `/health/ready`: Chroma, Ollama, model, **storage** (free space of the data
directory >= `MIN_FREE_DISK_MB`). `/health` always 200 with `status`. Health never calls generation.

## 7. Dashboard specification (not implemented)
* **API**: req/s by route, 5xx share, p50/p95/p99 from `module3_http_request_duration_seconds`.
* **AI**: `llm_request` outcome rates, `llm_generate` p95, `module3_llm_waiting/active/rejected`.
* **RAG**: `chroma_query`/`context_build` p95, `rag_retrieval{no_context}` share.
* **Pipeline**: `pdf_extract`, `chunking`, `embedding`, `index_write` p95; `index{failed|cleaned_up}`.
* **Evaluation/questions**: `evaluation`/`question_generation` error share and latency.
* **Streams**: `module3_streams_total` by outcome.
* **Infra** (external, not exported by the app except RSS on Linux): CPU, memory, disk, Ollama up.

## 8. Alert specification (not implemented; thresholds are starting points to tune on real traffic)
| Alert | Condition | Severity | Impact | Action | Runbook |
|---|---|---|---|---|---|
| High error rate | 5xx share > 5% for 10 min (>= 20 req) | high | users get errors | find dominant `error code` | Scenario 4 |
| High p95 latency | route p95 above SLO for 15 min | medium | slow answers | read `slowest_stage` | Scenario 5 |
| Ollama unavailable | `/ready` ollama error > 3 min | high | no generation | restart Ollama | Scenario 1 |
| Ollama timeouts | `llm_request{timeout}` share > 5% | medium | failed evaluations | check load/capacity | Scenario 2 |
| ChromaDB unavailable | `CHROMA_ERROR` rate > 0 for 5 min | high | no retrieval/upload | disk/permissions | Scenario 3 |
| Disk low | `/ready` storage error | high | writes fail | free space | Scenario 6 |
| Memory growth | RSS rising for hours | medium | OOM risk | restart, investigate | Scenario 7 |
| Repeated evaluation failure | `evaluation{error}` share > 10% | medium | grading unavailable | check Ollama/version | Scenarios 1,10 |
| Stream failures | `streams{error}` share > 5% | medium | SSE broken | check Ollama/proxy | Scenario 9 |
| Abuse | `RATE_LIMITED` spike | low | throttled clients | inspect source | SECURITY.md |

## 9. Dependency failure matrix
See FAILURE_RECOVERY.md.

## 10. Reading the numbers
`python -m scripts.check_reliability --url http://host:8000/metrics` (see SLO_SLA.md). Metrics are per process and reset
on restart; with several replicas, scrape each.
