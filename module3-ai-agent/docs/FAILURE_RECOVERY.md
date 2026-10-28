# Failure & Recovery (Phase 20)

Every row is covered by a mock-based test in `tests/reliability/` unless marked NOT VERIFIED. "Recovery" means the next
request succeeds without restarting the API.

| Dependency | Failure | Detection | Response | Cleanup | Recovery | Test |
|---|---|---|---|---|---|---|
| Ollama | refused | `ConnectError` | `OLLAMA_UNAVAILABLE` 503, no retry | none | next request | test_ollama_failure, test_dependency_recovery |
| Ollama | timeout | `TimeoutException` | `OLLAMA_TIMEOUT`, no retry | none | next request | same |
| Ollama | 502/503/504 | status | one bounded retry, then error | none | next request | test_ollama_failure |
| Ollama | connection reset | `RemoteProtocolError` etc. | one bounded retry | none | next request | test_ollama_failure |
| Ollama | model missing (404) | status | `OLLAMA_MODEL_NOT_FOUND`, no retry; `/ready` 503 | none | after `ollama pull` | test_ollama_failure |
| Ollama | malformed / unexpected body | parse/shape | `OLLAMA_INVALID_RESPONSE` | none | next request | test_ollama_failure |
| Ollama | failure after retrieval OK | exception | controlled error, no fabricated score | none | next request | test_pipeline_failure |
| Ollama | stream failure | exception | one `error` event then `done`, no traceback | slot released | new request | test_concurrency_sse, test_dependency_recovery |
| Request budget | exhausted | deadline | `OLLAMA_TIMEOUT` before calling Ollama | none | new request | test_ollama_failure |
| ChromaDB | query error (several types) | exception | `CHROMA_ERROR` 5xx, no sources | none | next request | test_pipeline_failure |
| ChromaDB | write fails after partial write | exception | upload error | partial vectors deleted | re-upload | test_pipeline_failure |
| ChromaDB | write ok but count mismatch | post-write verification | `INDEXING_ERROR` | deleted | re-upload | test_pipeline_failure |
| ChromaDB | delete fails | exception | 5xx, data intact | none | retry delete | test_pipeline_failure |
| ChromaDB | cleanup also fails | exception | original error returned, cleanup error logged | manual (log has doc id) | re-upload / delete | test_pipeline_failure |
| Embedding | unavailable | `AppError` | upload error, nothing indexed | none | re-upload | test_pipeline_failure |
| PDF parser | exception / invalid / empty / corrupt | exception / validation | 4xx or controlled 5xx | temp dir removed | new upload | test_pipeline_failure |
| Filesystem | temp dir cannot be created (e.g. disk full) | `OSError` | catch-all 500, no internals | none | after space freed | test_pipeline_failure |
| Filesystem | free space below `MIN_FREE_DISK_MB` | `/ready` storage check | `/ready` 503 (`/health/live` still 200) | none | after space freed | test_readiness_recovery |
| Concurrency | duplicate deletes / parallel evals | | no corruption, other docs intact | | | test_concurrency_sse |
| Client | SSE cancel / disconnect | cancellation | LLM slot released | slot | immediate | test_concurrency_sse, test_concurrency_limits |
| Process | SIGTERM in Docker | | | | | NOT VERIFIED (no Docker) |
| Ollama / Chroma | real outage on real services | | | | | NOT VERIFIED (mocks only) |
| Disk | actually full disk | | | | | NOT VERIFIED (simulated by error injection) |

Data consistency: metadata view (`GET /api/documents/{id}`) and the vector store agree after upload and delete (tested);
a document is only reported `indexed` after the chunk count is verified.

Operational procedures: OPERATIONS_RUNBOOK.md section 5.
