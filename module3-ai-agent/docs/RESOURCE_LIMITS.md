# Resource Limits (Phase 22)

Single source of truth for every bound in Module 3. The defaults in the table are **checked against
`app/core/config.py` by a test** (`tests/performance/test_resource_limits_doc.py`), so this file cannot silently drift.
All values are environment variables (see `.env.example`); invalid values stop the app at startup.

| Setting | Default | Scope | Behaviour when exceeded |
|---|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | 30 | upload | rejected before processing (streamed in 1 MB reads, never fully buffered by the endpoint) |
| `MAX_PDF_PAGES` | 500 | upload | rejected, client error |
| `MAX_EXTRACTED_CHARS` | 2000000 | upload | rejected, client error |
| `MAX_CHUNKS_PER_DOCUMENT` | 3000 | upload | rejected, client error |
| `PDF_PROCESSING_TIMEOUT_S` | 60.0 | upload | cooperative deadline checked between pages |
| `MAX_CONCURRENT_UPLOADS` | 2 | upload | extra uploads are rejected (not queued); 0 = unlimited |
| `MAX_REQUEST_BODY_BYTES` | 1048576 | JSON bodies | 413-style rejection |
| `MAX_QUESTION_CHARS` | 2000 | evaluation/RAG input | 422 |
| `MAX_ANSWER_CHARS` | 10000 | evaluation input | 422 |
| `TOP_K` | 5 | RAG | default number of retrieved chunks |
| `MAX_TOP_K` | 20 | RAG | requests above it are rejected (and `TOP_K` must be <= it) |
| `MAX_CONTEXT_CHARS` | 12000 | RAG | context is truncated at chunk boundaries |
| `EMBEDDING_BATCH_SIZE` | 64 | embedding | documents are embedded in sequential batches of at most this many chunks |
| `MAX_CONCURRENT_EMBEDDINGS` | 4 | embedding | extra callers wait `EMBEDDING_QUEUE_TIMEOUT_S`, then `503 SERVICE_OVERLOADED`; 0 = unlimited |
| `EMBEDDING_QUEUE_TIMEOUT_S` | 30.0 | embedding | see above |
| `LLM_MAX_CONCURRENCY` | 4 | LLM | extra callers wait `LLM_QUEUE_TIMEOUT_S`, then `503 SERVICE_OVERLOADED`; 0 = unlimited |
| `LLM_QUEUE_TIMEOUT_S` | 30.0 | LLM | see above |
| `MAX_MATRIX_QUESTIONS` | 50 | exam matrix | request rejected with 422 when the rows add up to more |
| `OLLAMA_TIMEOUT` | 120 | LLM / embedding call | call fails with a timeout error |
| `OLLAMA_NUM_PREDICT` | 1024 | LLM output tokens | model stops generating (config key `ollama_num_predict`) |
| `MAX_LLM_OUTPUT_CHARS` | 60000 | LLM output | output rejected |
| `REQUEST_TIMEOUT_S` | 300.0 | whole request | must be >= `OLLAMA_TIMEOUT` (validated) |
| `STREAM_MAX_DURATION_S` | 300.0 | SSE | stream ends with an error event |
| `RATE_LIMIT_PER_MINUTE` | 120 | expensive endpoints, per client | `429 RATE_LIMITED`; 0 = disabled (in-process, single instance) |
| `MIN_FREE_DISK_MB` | 100 | readiness | `/ready` reports not ready |

## Notes

- Two layers of backpressure exist and are independent: the LLM gate and the embedding gate. A question-generation
  request holds an embedding slot only while embedding (the query), never while waiting for the LLM.
- The semaphores are per process. With several Uvicorn workers the limits multiply by the worker count.
  The documented deployment is a single process (see `docs/SCALABILITY.md`).
- `MAX_CONCURRENT_UPLOADS` bounds PDF CPU work; a 500-page document is the worst case allowed by `MAX_PDF_PAGES`.
- Docker: limits are passed through the environment, there are no separate container-level overrides in
  `docker-compose.yml`. No CPU/memory limits are set on the containers (NOT VERIFIED what is appropriate: needs a
  real Ollama host to measure).
