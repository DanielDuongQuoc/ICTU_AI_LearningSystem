# Reliability (Phase 20)

## 1. Timeout chain
```
REQUEST_TIMEOUT_S (whole request, default 300)
   >= OLLAMA_TIMEOUT (one model call)  >= OLLAMA_CONNECT_TIMEOUT
```
Validated at startup (`REQUEST_TIMEOUT_S >= OLLAMA_TIMEOUT`). The request budget is a **deadline**, not a cancellation
(`app/core/timeouts.py`): each Ollama/embedding call gets a tighter httpx timeout when the budget is smaller than its own
timeout, is refused (`OLLAMA_TIMEOUT`, event `request_budget{exhausted}`) when the budget is spent, and is not retried when
too little time is left (`request_budget{retry_skipped}`). Other bounds: `PDF_PROCESSING_TIMEOUT_S`, `LLM_QUEUE_TIMEOUT_S`,
`STREAM_MAX_DURATION_S`. `timeout_chain()` returns the effective values.
Not enforced: a hard HTTP-server cancellation of in-flight work at the deadline. Pipelines issuing several sequential model
calls stop issuing new ones after the deadline; a call already running ends at its own (clamped) timeout.

## 2. Retry policy (audited; one layer only, in the services)
| Operation | Retried | Not retried |
|---|---|---|
| LLM non-stream | HTTP 502/503/504, connection reset (`RemoteProtocolError`/`ReadError`/`WriteError`): at most `OLLAMA_MAX_RETRIES` (1), backoff `OLLAMA_RETRY_BACKOFF_S`, only if budget remains | refused connection, timeouts, 404 model missing, other 4xx, 500 |
| LLM stream | 502/503/504 before any output | mid-stream failures (client already received text) |
| Embedding | same as LLM non-stream (new in Phase 20) | same |
| `generate_json` | one re-ask on invalid JSON (output may differ) | |
| Question/evaluation loops | `MAX_LLM_ATTEMPTS=2` on semantic validation failure | transport errors propagate |
Tests: `tests/reliability/test_ollama_failure.py`. No infinite or unbounded retry exists.

## 3. Failure handling guarantees (tested offline with mocks)
* Ollama refused / timeout / 404 / malformed / unexpected shape -> specific `ErrorCode`, `request_id`, no stack trace.
* Chroma query failure -> controlled 5xx, no invented sources; next request succeeds after recovery.
* Upload: embedding or Chroma write failure (even after a partial write), count mismatch, client cancellation ->
  partial vectors removed (`index{cleaned_up}`), response is an error, never `status: indexed`.
  A failing cleanup is logged and never masks the original error.
* PDF parser exception / invalid / empty / corrupt PDF -> nothing stored.
* Unexpected exceptions -> catch-all JSON 500 without internals.

## 4. Idempotency
| Operation | Idempotent? | Behaviour |
|---|---|---|
| Upload | No | each upload gets a new `document_id`; duplicates are isolated copies (documented Phase 16) |
| Index same `document_id` again | Yes | deterministic ids + upsert; no duplicate vectors (tested) |
| Delete | Yes in effect | second delete -> 404/200, no error or corruption; concurrent deletes are safe |
| Retrieve / answer / evaluate / generate questions | Read-only | repeat = new generation; no stored state |
| Evaluation stream reconnect | Not resumable | a reconnect is a new request = exactly one generation (tested) |
| AI-version promote/rollback | Guarded | lock + revision check; concurrent promotions: exactly one wins (Phase 18 test) |

## 5. Concurrency and backpressure
* LLM gate: `LLM_MAX_CONCURRENCY`, `LLM_QUEUE_TIMEOUT_S`, excess -> 503 `SERVICE_OVERLOADED`; cancellation releases the slot (tested).
* Upload slots reject excess concurrent uploads with 503 (tested).
* Parallel retrieves/evaluations on one document return consistent sources; one failing request does not affect others (tested).
* Phase 18 promote/rollback use a file lock and revision check (existing tests).
* **Limitation**: gate, upload slots, rate limiter and metrics are in-process. With several replicas each has its own limits
  and the file-lock/state file assumes a shared single filesystem. The deployment is single-instance.
* **Limitation**: Chroma calls are synchronous (known Phase 16 limit): a slow Chroma call blocks the event loop.

## 6. Shutdown
Existing tests cover closing shared HTTP clients, never raising on close failures and degraded startup
(`tests/test_shutdown.py`). Real SIGTERM behaviour in Docker and in-flight SSE draining are **NOT VERIFIED**.

## 7. Quality vs latency vs reliability
Do not pick an AI version on quality alone. Compare per version: quality-gate result (Phase 17), latency p95 by stage,
error/timeout share, and memory/CPU (host). Example: quality +3% with 6x latency can breach the evaluation latency SLO
and consume error budget through timeouts; the promotion decision must list that trade-off. Version identity is in logs
(`config_version`, model, prompt version, rag fingerprint) and never in metric labels.

## 8. Remaining limitations
No startup log of the active AI version/timeout chain; no soak/long-run test; no real SIGTERM test; in-process metrics only;
no automatic reconciliation job between metadata and Chroma (consistency is enforced at write time and checked by test).
