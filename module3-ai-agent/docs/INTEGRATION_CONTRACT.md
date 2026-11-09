# Module 3 — Integration Contract

Module 3's public HTTP API, as consumed by the C# backend (Module 1 / Module 2). The API is **frozen**:
paths, request/response schemas, status codes and error-code strings do not change without a new
contract version. The machine-readable source of truth is `GET /openapi.json` (Swagger UI at `/docs`);
`tests/test_api_contract.py` pins it.

Base URL: `http://<host>:8003` locally, the Render URL in production. JSON (`application/json`) unless
stated. There is no `/api/v1` prefix and no authentication (see [SECURITY.md](SECURITY.md)).

The AI provider behind the API is the **Google Gemini API** (generation + embeddings). Earlier versions
used Ollama; a few response fields and error codes keep their historical `ollama`/`OLLAMA_*` names for
backward compatibility (see [Legacy names](#legacy-names)).

## Endpoints

| Method & path | Purpose | Success |
|---|---|---|
| `GET /health` | Dependency status, always HTTP 200 | `HealthCheckResponse` |
| `GET /health/live` | Liveness (no dependency checked) | `{"status": "ok"}` |
| `GET /health/ready`, `GET /ready` | Readiness: 200 when all dependencies ok, else 503 (same body as `/health`) | `HealthCheckResponse` |
| `POST /api/documents/upload` | Multipart upload (field `file`, `.pdf` or `.docx`) → extract, chunk, embed, index | `DocumentUploadResponse` |
| `GET /api/documents/{document_id}` | Document info (404 `DOCUMENT_NOT_FOUND` if absent) | `DocumentInfoResponse` |
| `DELETE /api/documents/{document_id}` | Delete the document's chunks | `DocumentDeleteResponse` |
| `POST /api/documents/{document_id}/search` | Semantic search inside one document `{query, top_k?}` | `SearchResponse` |
| `POST /api/rag/retrieve` | Retrieve chunks + context, no LLM call | `RAGRetrievalResponse` |
| `POST /api/rag/answer` | Grounded answer + sources | `RAGAnswerResponse` |
| `POST /api/questions/generate` | Generate questions from a document | `QuestionGenerationResponse` |
| `POST /api/questions/generate-matrix` | Exam matrix (several type/difficulty rows) | `MatrixGenerationResponse` |
| `POST /api/evaluations/evaluate` | Score a student answer | `EvaluationResponse` |
| `POST /api/evaluations/evaluate/stream` | Same evaluation as Server-Sent Events | `text/event-stream` |
| `POST /api/rubrics/generate` | Draft an analytic rubric (criteria x levels) for a question | `RubricGenerateResponse` |
| `POST /api/rubrics/validate` | Check a rubric against the code rules (no AI call) | `RubricValidationResponse` |
| `POST /api/grading/grade` | Grade one submission with an approved rubric (homework PASS/FAIL, in-class points) | `GradeResult` |
| `POST /api/grading/batch` | Grade several submissions of one question | `GradeBatchResponse` |
| `GET /metrics` | Prometheus text metrics (internal, not in OpenAPI) | text |
| `GET /docs`, `GET /openapi.json` | API documentation | |

Every response carries `X-Request-ID` (echoes a valid client value matching
`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, otherwise a generated UUID), plus `X-AI-Config-Version` and
`X-AI-Config-Fingerprint` (identify the model/prompt configuration that produced the answer).

## Documents

`POST /api/documents/upload` (multipart, field `file`, max `MAX_UPLOAD_SIZE_MB`, default 30 MB):

```json
{"document_id": "<uuid4>", "filename": "lecture.pdf", "pages": 12, "chunks": 34, "status": "indexed"}
```

- Module 3 generates `document_id`; it does not accept a caller-supplied id. The caller must store it.
  Uploading the same file twice creates two independent documents.
- `.docx`: OOXML content is verified (legacy `.doc` and macro files are rejected with 400 `INVALID_FILE`).
  Word has no stored page numbers, so `page` is a logical page (explicit page break, or ~3000 chars).
- Scanned/image-only files have no text → 422 `NO_EXTRACTABLE_TEXT`.
- Limits: `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS_PER_DOCUMENT` → 413 `DOCUMENT_LIMIT_EXCEEDED`;
  `PDF_PROCESSING_TIMEOUT_S` → 504; more than `MAX_CONCURRENT_UPLOADS` in progress → 503 `SERVICE_OVERLOADED`.
- The document id never changes: upload response, document info, retrieval sources, question
  `source_chunks` and evaluation sources all refer to the same id.

## RAG

`POST /api/rag/retrieve` and `POST /api/rag/answer`, body `{"document_id", "question", "top_k"?}`
(`top_k` 1..`MAX_TOP_K`, default `TOP_K`). Sources carry `chunk_id`, `document_id`, `page`, `source`, `score`
and come from the vector store, never from the model. When nothing relevant is found the response has
`insufficient_context: true` and the LLM is not called. Retrieval is always filtered by `document_id`.

## Questions

`POST /api/questions/generate`

```json
{"document_id": "<uuid>", "number_of_questions": 5, "difficulty": "medium", "question_type": "short_answer"}
```

`number_of_questions` 1-20; `difficulty` easy|medium|hard; `question_type`
short_answer|multiple_choice|true_false|essay. Response:
`{"document_id", "questions": [{"question_id", "question", "expected_answer", "difficulty", "type", "source_chunks", "options"?, "correct_answer"?}]}`.

Reference material is optional (additive fields, the body above keeps working):

- `document_ids`: up to 5 documents of the same course, retrieved together; the response adds `document_ids`.
- `course_context`: `{"course_name", "course_code"?, "clos": [{"code", "description"}], "chapters": [], "language": "vi"}`.
  Without any document the questions are generated from this structured data only, carry `"grounded": false`
  and `source_chunks: []`, and must be reviewed by the lecturer. Sending none of `document_id`, `document_ids`,
  `course_context` → 400 `EMPTY_DOCUMENT_ID`. `generate-matrix` accepts the same three fields.

`POST /api/questions/generate-matrix`

```json
{"document_id": "<uuid>", "subject": "Databases 101",
 "items": [{"question_type": "multiple_choice", "difficulty": "medium", "count": 5},
           {"question_type": "essay", "difficulty": "hard", "count": 2}]}
```

`items` 1-12 rows, `count` 1-20 per row, total <= `MAX_MATRIX_QUESTIONS` (50). `subject` (optional, <= 200
chars) is echoed back and not sent to the model. Response:
`{"document_id", "subject", "requested_total", "generated_total", "complete", "items": [{"index", "question_type", "difficulty", "requested", "generated", "status", "error_code"?}], "questions": [{"matrix_item", ...question fields}]}`.
Rows run sequentially (one LLM call each); nothing is padded to reach a count; each row reports
`ok` / `partial` / `failed`. If the first row fails the whole call returns that error; later row failures
are reported inside the 200 response.

## Evaluation

`POST /api/evaluations/evaluate`

```json
{"document_id": "<uuid>", "question": "...", "student_answer": "...", "expected_answer": "...", "max_score": 10}
```

Response:

```json
{"score": 8.0, "max_score": 10.0, "percentage": 80.0, "correct": true,
 "feedback": "...", "strengths": ["..."], "missing_points": ["..."], "errors": [],
 "sources": [{"page": 1, "chunk_id": "...", "score": 0.82}]}
```

`percentage`, `max_score` and `sources` are computed by the backend; a model score outside
`0..max_score` is rejected. Module 3 only scores answers — it has no cheating/fraud output.

### Streaming

`POST /api/evaluations/evaluate/stream` (same body). `text/event-stream` with `Cache-Control: no-cache`,
`Connection: keep-alive`, `X-Accel-Buffering: no`. Each frame is `data: <json>\n\n` with `type`:

| type | meaning |
|---|---|
| `status` | progress message |
| `token` | incremental model text (display only) |
| `result` | final validated evaluation (same fields as above), at most once |
| `error` | `{code, message}`; followed by `done` |
| `done` | always the last event, exactly once |

Request-field problems are reported as `error` events with HTTP 200. Through a C#/proxy layer: read with
`HttpCompletionOption.ResponseHeadersRead`, flush per chunk, disable response compression/proxy buffering
for this route, and keep read timeouts above `LLM_TIMEOUT` (120 s) — no keep-alive comments are sent.

## Rubrics and grading

Two assessment contexts decide how a submission is graded. The caller never chooses the mode:

| `assessment_context` | Who decides | Result | `status` |
|---|---|---|---|
| `homework` | AI / Grading Agent, automatically | `result`: `PASS` or `FAIL` + feedback (no points) | `final`, or `needs_review` when a review flag is raised |
| `in_class` | The lecturer | `suggested_total` + points per criterion + reasons/evidence | always `pending_review` |

`POST /api/rubrics/generate`

```json
{"assessment_context": "homework", "question": "...", "expected_answer": "...", "max_score": 10,
 "document_ids": ["<uuid>"], "course_context": {"course_name": "Cơ sở dữ liệu", "clos": [{"code": "CLO2", "description": "..."}]},
 "language": "vi"}
```

`document_ids` (0-5) and `course_context` are optional; without documents the rubric has `grounded=false`.
Response `{"rubric": Rubric, "review_applied", "review_issues": [...], "model", "prompt_version"}`. The rubric is a
`draft`; the lecturer edits it, may check it with `POST /api/rubrics/validate` (`{"rubric", "assessment_context"?}` →
`{"valid", "violations": [{"rule", "criterion_id", "message"}]}`) and stores it with `status: "approved"`.

`Rubric`: `{"max_score", "criteria": [{"criterion_id", "name", "description", "max_points", "required",
"key_points", "levels": [{"label", "points", "descriptor"}], "clo"?, "bloom_level"?, "source_chunks"}],
"pass_rule": {"all_required_met": true, "min_percentage": 50}, "grounded", "status"}`. Rules: R1 points sum to
`max_score`; R2 >= 3 levels, strictly decreasing, top = `max_points`, bottom = 0; R4 no duplicate criteria;
R6 grounded criteria carry `source_chunks`; homework needs a pass rule that can fail.

`POST /api/grading/grade`

```json
{"assessment_context": "in_class", "submission_kind": "typed", "question": "...", "rubric": {"...": "approved rubric"},
 "answer": "...", "expected_answer": "...", "submission_ref": "SV001", "language": "vi"}
```

`submission_kind`: `typed` | `handwritten_transcript` | `code`. `submission_ref` is an anonymous id (no names).
Response (`GradeResult`): `{"assessment_context", "grading_mode", "status", "result"?, "unmet_required",
"suggested_total"?, "max_score"?, "criteria": [{"criterion_id", "name", "required", "level", "met", "points"?,
"max_points"?, "reason", "evidence"}], "feedback", "flags", "warnings", "trace": {"model", "prompt_version",
"request_id", "grounded"}}`. Points, totals and PASS/FAIL are computed by the backend from the rubric; the model
only picks one level label per criterion. Review flags: `injection_suspected`, `no_valid_files`, `unreadable`,
`truncated`, `not_executed`, `grading_error`; warnings: `empty_or_off_topic`, `ungrounded_rubric`.

`POST /api/grading/batch`: `{"assessment_context", "question", "rubric", "expected_answer"?, "document_ids"?,
"submissions": [{"submission_ref", "answer", "submission_kind"?}]}` (max `MAX_BATCH_SUBMISSIONS`) →
`{"total", "succeeded", "failed", "results": [{"submission_ref", "ok", "result"?, "error"?}]}`.

## Health

```json
{"status": "ok", "ollama": true, "chromadb": true, "version": "0.14.0",
 "services": {"api": "ok", "chroma": "ok", "ollama": "ok", "model": "ok", "storage": "ok", "embedding_index": "ok"}}
```

`status` is `ok` or `degraded`. `ollama` / `services.ollama` = the Gemini API is reachable with the
configured key; `services.model` = `GEMINI_MODEL` is available. A missing `GEMINI_API_KEY` shows as
`degraded` (the process still runs).

## Errors

Every error body is `{"error": {"code": "...", "message": "...", "request_id": "..."}}` (never a stack trace).
Validation failures are 422 `VALIDATION_ERROR`; rate limit 429 `RATE_LIMITED` with `Retry-After`.

| HTTP | Error code | Situation |
|---:|---|---|
| 400 | `EMBEDDING_EMPTY_TEXT` | Text to embed is empty |
| 400 | `EMPTY_DOCUMENT_ID` | document_id is empty |
| 400 | `EMPTY_EXPECTED_ANSWER` | expected_answer is empty |
| 400 | `EMPTY_FILE` | Uploaded file is empty |
| 400 | `EMPTY_QUERY` | RAG question is empty |
| 400 | `EMPTY_QUESTION` | Evaluation question is empty |
| 400 | `EMPTY_STUDENT_ANSWER` | student_answer is empty |
| 400 | `EMPTY_SUBMISSION` | Submission to grade is empty |
| 400 | `HTTP_ERROR` | Other HTTP error raised by the framework |
| 400 | `INPUT_TOO_LONG` | question > MAX_QUESTION_CHARS or answer > MAX_ANSWER_CHARS |
| 400 | `INVALID_DIFFICULTY` | difficulty not easy/medium/hard |
| 400 | `INVALID_DOCUMENT_ID` | document_id is not a valid UUID (question/evaluation endpoints) |
| 400 | `INVALID_FILE` | Not an accepted PDF/DOCX upload (extension / content type / signature) |
| 400 | `INVALID_NUMBER_OF_QUESTIONS` | number_of_questions outside 1-20 |
| 400 | `INVALID_PDF` | File cannot be opened (corrupted) |
| 400 | `INVALID_QUESTION_TYPE` | question_type not supported |
| 400 | `INVALID_SCORE` | max_score invalid (502 when the model produced the bad score) |
| 400 | `INVALID_TOP_K` | top_k < 1 or > MAX_TOP_K |
| 404 | `DOCUMENT_NOT_FOUND` | No chunks stored for this document_id |
| 404 | `FILE_NOT_FOUND` | Internal file reference missing |
| 404 | `NOT_FOUND` | Unknown route |
| 405 | `METHOD_NOT_ALLOWED` | HTTP method not allowed on this route |
| 413 | `FILE_TOO_LARGE` | Upload exceeds MAX_UPLOAD_SIZE_MB |
| 413 | `DOCUMENT_LIMIT_EXCEEDED` | Page / extracted-text / chunk limit exceeded |
| 413 | `REQUEST_TOO_LARGE` | JSON body exceeds MAX_REQUEST_BODY_BYTES |
| 422 | `INSUFFICIENT_CONTEXT` | Retrieved context is not enough; the LLM is NOT called |
| 422 | `NO_EXTRACTABLE_TEXT` | No extractable text (e.g. scanned images, no OCR) |
| 422 | `RUBRIC_INVALID` | Rubric violates the rubric rules (generation after one repair, or grading input) |
| 422 | `RUBRIC_NOT_APPROVED` | Grading requested with a rubric whose status is not `approved` |
| 422 | `VALIDATION_ERROR` | Request body/params fail schema validation |
| 429 | `RATE_LIMITED` | Per-client rate limit exceeded (see Retry-After) |
| 500 | `APPLICATION_CONFIGURATION_ERROR` | Invalid configuration (startup failure; not an HTTP response) |
| 500 | `EVALUATION_ERROR` | Unexpected failure during evaluation |
| 500 | `GRADING_ERROR` | Unexpected failure during rubric-based grading |
| 500 | `INDEXING_ERROR` | Failed to store chunks (or embedding-model mismatch, re-index needed) |
| 500 | `INTERNAL_SERVER_ERROR` | Unhandled internal error (details only in server logs) |
| 500 | `PROCESSING_ERROR` | Unexpected failure while processing the file |
| 500 | `QUESTION_GENERATION_ERROR` | Unexpected failure in question generation |
| 500 | `RAG_RETRIEVAL_ERROR` | Unexpected failure during retrieval |
| 500 | `SOURCE_INTEGRITY_ERROR` | Sources do not belong to the requested document (blocked) |
| 502 | `EMBEDDING_ERROR` | Embedding step failed during retrieval |
| 502 | `EMBEDDING_INVALID_RESPONSE` | Gemini returned a malformed embedding or an unexpected HTTP status |
| 502 | `EVALUATION_INVALID_RESPONSE` | Model evaluation JSON invalid / inconsistent |
| 502 | `GRADING_INVALID_RESPONSE` | Model grading output invalid (unknown criterion/level) after one retry |
| 502 | `LLM_EMPTY_RESPONSE` | Model returned empty output |
| 502 | `LLM_INVALID_JSON` | Model output is not valid JSON (after one retry) |
| 502 | `LLM_OUTPUT_REJECTED` | Model output rejected (too large / leaked prompt) |
| 502 | `OLLAMA_GENERATION_ERROR` | Gemini returned an error status (incl. persistent 429 quota errors) |
| 502 | `OLLAMA_INVALID_RESPONSE` | Gemini returned a malformed response |
| 502 | `QUESTION_GENERATION_INVALID_RESPONSE` | Model produced no valid questions |
| 502 | `RUBRIC_GENERATION_ERROR` | Rubric generation failed upstream |
| 503 | `CHROMA_ERROR` | Vector store operation failed / unavailable |
| 503 | `EMBEDDING_MODEL_NOT_FOUND` | GEMINI_EMBEDDING_MODEL does not exist |
| 503 | `EMBEDDING_SERVICE_UNAVAILABLE` | Gemini embedding endpoint unreachable or API key missing |
| 503 | `OLLAMA_MODEL_NOT_FOUND` | GEMINI_MODEL does not exist |
| 503 | `OLLAMA_UNAVAILABLE` | Gemini API unreachable or GEMINI_API_KEY not configured |
| 503 | `SERVICE_OVERLOADED` | No LLM/embedding/upload slot free in time (retry shortly) |
| 504 | `EMBEDDING_TIMEOUT` | Embedding request timed out |
| 504 | `OLLAMA_TIMEOUT` | Gemini did not answer within LLM_TIMEOUT (or the request budget ran out) |
| 504 | `PDF_PROCESSING_TIMEOUT` | Extraction exceeded PDF_PROCESSING_TIMEOUT_S |

On the streaming endpoint these errors arrive as an SSE `error` event followed by `done` (HTTP 200).

## Legacy names

Kept unchanged so existing clients keep working; they now refer to the Gemini provider:

- `/health` fields `ollama` and `services.ollama` (provider reachable) and `services.model`.
- Error codes `OLLAMA_UNAVAILABLE`, `OLLAMA_TIMEOUT`, `OLLAMA_MODEL_NOT_FOUND`, `OLLAMA_INVALID_RESPONSE`,
  `OLLAMA_GENERATION_ERROR`.

The disabled-by-default admin endpoints `/api/ai-versions*` were removed (never part of the client contract).
