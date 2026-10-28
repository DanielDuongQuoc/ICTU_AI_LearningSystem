# Error matrix

Generated from `app/core/exceptions.py::ERROR_HTTP_STATUS` and checked by `tests/test_phase15.py`.
Every error body is `{"error": {"code", "message", "request_id"}}`. On the streaming endpoint
(`/api/evaluations/evaluate/stream`) request/dependency errors are sent as an SSE `error` event
(`code`, `message`) followed by `done` with HTTP 200.

| HTTP | Error code | Situation |
|---:|---|---|
| 400 | `EMBEDDING_EMPTY_TEXT` | Text to embed is empty |
| 400 | `EMPTY_DOCUMENT_ID` | document_id is empty |
| 400 | `EMPTY_EXPECTED_ANSWER` | expected_answer is empty |
| 400 | `EMPTY_FILE` | Uploaded file is empty |
| 400 | `EMPTY_QUERY` | RAG question is empty |
| 400 | `EMPTY_QUESTION` | Evaluation question is empty |
| 400 | `EMPTY_STUDENT_ANSWER` | student_answer is empty |
| 400 | `HTTP_ERROR` | Other HTTP error raised by the framework |
| 400 | `INPUT_TOO_LONG` | question > MAX_QUESTION_CHARS or answer > MAX_ANSWER_CHARS |
| 400 | `INVALID_DIFFICULTY` | difficulty not easy/medium/hard |
| 400 | `INVALID_DOCUMENT_ID` | document_id is not a valid UUID (question/evaluation endpoints) |
| 400 | `INVALID_FILE` | Upload is not a valid PDF upload (extension / content type / signature) |
| 400 | `INVALID_NUMBER_OF_QUESTIONS` | number_of_questions outside 1-20 |
| 400 | `INVALID_PDF` | File cannot be opened as a PDF (corrupted / not a PDF) |
| 400 | `INVALID_QUESTION_TYPE` | question_type not supported |
| 400 | `INVALID_SCORE` | max_score invalid (request) or model score outside 0..max_score (502 when produced by the model) |
| 400 | `INVALID_TOP_K` | top_k < 1 or > MAX_TOP_K |
| 404 | `DOCUMENT_NOT_FOUND` | No chunks stored for this document_id |
| 404 | `FILE_NOT_FOUND` | Internal file reference missing |
| 404 | `NOT_FOUND` | Unknown route |
| 405 | `METHOD_NOT_ALLOWED` | HTTP method not allowed on this route |
| 413 | `FILE_TOO_LARGE` | Upload exceeds MAX_UPLOAD_SIZE_MB |
| 422 | `INSUFFICIENT_CONTEXT` | Retrieved context is not enough; the LLM is NOT called |
| 422 | `NO_EXTRACTABLE_TEXT` | PDF has no extractable text (e.g. scanned images, no OCR) |
| 422 | `VALIDATION_ERROR` | Request body/params fail schema validation (FastAPI/Pydantic) |
| 500 | `APPLICATION_CONFIGURATION_ERROR` | Invalid configuration (startup failure; not an HTTP response) |
| 500 | `EVALUATION_ERROR` | Unexpected failure during evaluation |
| 500 | `INDEXING_ERROR` | Failed to store chunks in ChromaDB |
| 500 | `INTERNAL_SERVER_ERROR` | Unhandled internal error (details only in server logs) |
| 500 | `PROCESSING_ERROR` | Unexpected failure while processing the PDF |
| 500 | `QUESTION_GENERATION_ERROR` | Unexpected failure in question generation |
| 500 | `RAG_RETRIEVAL_ERROR` | Unexpected failure during retrieval |
| 500 | `SOURCE_INTEGRITY_ERROR` | Sources do not belong to the requested document (blocked) |
| 502 | `EMBEDDING_ERROR` | Embedding step failed during retrieval |
| 502 | `EMBEDDING_INVALID_RESPONSE` | Ollama returned a malformed embedding |
| 502 | `EVALUATION_INVALID_RESPONSE` | Model evaluation JSON invalid / inconsistent |
| 502 | `LLM_EMPTY_RESPONSE` | Model returned empty output |
| 502 | `LLM_INVALID_JSON` | Model output is not valid JSON (after one retry) |
| 502 | `OLLAMA_GENERATION_ERROR` | Ollama reported a generation error |
| 502 | `OLLAMA_INVALID_RESPONSE` | Ollama returned a malformed response |
| 502 | `QUESTION_GENERATION_INVALID_RESPONSE` | Model produced no valid questions |
| 503 | `CHROMA_ERROR` | ChromaDB operation failed / unavailable |
| 503 | `EMBEDDING_MODEL_NOT_FOUND` | Embedding model is not pulled in Ollama |
| 503 | `EMBEDDING_SERVICE_UNAVAILABLE` | Ollama embedding endpoint unreachable |
| 503 | `OLLAMA_MODEL_NOT_FOUND` | Generation model is not pulled in Ollama |
| 503 | `OLLAMA_UNAVAILABLE` | Ollama is unreachable (connection refused / DNS) |
| 503 | `SERVICE_OVERLOADED` | No LLM concurrency slot became free within LLM_QUEUE_TIMEOUT_S (backpressure; retry shortly) |
| 504 | `EMBEDDING_TIMEOUT` | Embedding request timed out |
| 504 | `OLLAMA_TIMEOUT` | Ollama did not answer within OLLAMA_TIMEOUT |
| 413 | `DOCUMENT_LIMIT_EXCEEDED` | PDF exceeds the page / extracted-text / chunk limit |
| 504 | `PDF_PROCESSING_TIMEOUT` | PDF extraction exceeded PDF_PROCESSING_TIMEOUT_S |
| 429 | `RATE_LIMITED` | Per-client rate limit exceeded (see Retry-After) |
| 502 | `LLM_OUTPUT_REJECTED` | Model output rejected (too large / leaked prompt) |
| 413 | `REQUEST_TOO_LARGE` | JSON request body exceeds MAX_REQUEST_BODY_BYTES |
| 401 | `ADMIN_AUTH_REQUIRED` | Missing or wrong X-Admin-Token on /api/ai-versions (admin plane) |
| 503 | `ADMIN_DISABLED` | ADMIN_API_TOKEN is not configured, so the admin plane is off |
| 404 | `AI_VERSION_NOT_FOUND` | Unknown AI version id |
| 422 | `AI_VERSION_INVALID` | AI version manifest or request failed validation |
| 409 | `AI_VERSION_DUPLICATE` | AI version id or fingerprint already exists |
| 409 | `AI_VERSION_INVALID_TRANSITION` | Status transition not allowed by the lifecycle |
| 409 | `AI_VERSION_CONFLICT` | Concurrent change detected (stale revision or lock held) |
| 409 | `AI_VERSION_GATE_BLOCKED` | Promotion gate verdict is not PASS (or evidence missing) |
| 409 | `AI_VERSION_PREFLIGHT_FAILED` | Model/prompt/config availability check failed before activation |
| 500 | `AI_VERSION_STORE_ERROR` | AI version store could not be read or written
 |

Notes:
- `INVALID_SCORE` is 400 for a bad `max_score` in the request and 502 when the *model* returns an out-of-range score.
- The generic internal error code is `INTERNAL_SERVER_ERROR` (there is no `INTERNAL_ERROR`).
- A status given by a code can be overridden per raise (`AppError(status_code=...)`); the table is the default.
- Validation errors on JSON endpoints use `VALIDATION_ERROR` (422) with the same envelope as every other error;
  the domain checks (empty text, bad ids, bad `top_k`, ...) use the 400 codes above.
