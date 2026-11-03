# Security

Scope: Module 3 is an internal service called server-to-server by the C# backend. **There is no
authentication or authorisation on the API**; restrict network access (private network, reverse proxy,
firewall) and do not rely on CORS as protection. On Render the service URL is public — treat
`document_id`s (random UUID4) as the only access barrier and keep them private in the backend.

## Secrets

- The only secret is `GEMINI_API_KEY`. Set it in `.env` (local, git-ignored and docker-ignored) or in the
  platform secret store (Render: `sync: false`). `.env.example` contains no value.
- The key is sent only as the `x-goog-api-key` header (never in URLs), excluded from `Settings` repr, and
  never logged. Log lines and error bodies never contain prompts, document text, student answers or model output.
- `tests/test_security.py` scans every tracked file for Google API key patterns (`AIza...`, `AQ....`) and
  fails the suite if one is committed. If a key leaks: revoke it in Google AI Studio, create a new one,
  update the secret store.

## CORS

Disabled by default. `CORS_ALLOWED_ORIGINS=https://a.example,https://b.example` enables it for those
origins only; credentials are never allowed; `*` is rejected when `APP_ENV=production`.

## Rate limits and backpressure

- `RATE_LIMIT_PER_MINUTE` (default 120): per-client sliding window on the expensive routes (upload, RAG,
  questions, evaluations incl. SSE) → 429 `RATE_LIMITED` + `Retry-After`. In-process (single instance,
  resets on restart), keyed by TCP peer address (`X-Forwarded-For` is not trusted, so clients behind one
  proxy share a bucket).
- `LLM_MAX_CONCURRENCY` / `MAX_CONCURRENT_EMBEDDINGS` cap simultaneous Gemini calls; callers wait up to
  the queue timeout, then get 503 `SERVICE_OVERLOADED`. This also protects the Gemini quota.
- `REQUEST_TIMEOUT_S` is a per-request budget: no model call or retry starts after it is spent.

## Upload and input limits

| Area | Control |
|---|---|
| Upload type | `.pdf` (`%PDF` signature) or `.docx` (verified OOXML; `.doc`/macro files rejected) |
| Size | streamed cap `MAX_UPLOAD_SIZE_MB` (413), empty file rejected |
| Expansion | `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS_PER_DOCUMENT`, DOCX decompression cap |
| Time | `PDF_PROCESSING_TIMEOUT_S` (cooperative, per page) |
| Parallel uploads | `MAX_CONCURRENT_UPLOADS` (rejected, not queued) |
| Paths | client filename sanitised and never used as a path; temp files named from a server UUID |
| JSON body | `MAX_REQUEST_BODY_BYTES` (413 `REQUEST_TOO_LARGE`) |
| Text fields | `MAX_QUESTION_CHARS`, `MAX_ANSWER_CHARS`, `top_k <= MAX_TOP_K`, 1-20 questions, `0 < max_score <= 100` |

## AI-specific controls

- Document text, questions and student answers are delimited untrusted data in the user message; the
  system prompt is fixed. No context → `INSUFFICIENT_CONTEXT` without calling the model.
- Model output is parsed strictly (no NaN/Infinity, duplicate keys or deep nesting), size-capped
  (`MAX_LLM_OUTPUT_CHARS`), schema-validated, and rejected if it repeats the system prompt.
- Sources, ids, `percentage` and `max_score` are backend-derived; model-supplied metadata is ignored.
- SSE: `error → done` on failure, wall-clock cap `STREAM_MAX_DURATION_S`, upstream closed on disconnect.

## HTTP and container

- Responses carry `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`
  and a restrictive CSP (except `/docs`).
- Docker image: non-root user, no `.env` baked in; compose adds `no-new-privileges` and `cap_drop: [ALL]`.
- Production config validation: no `*` CORS origin, no `LOG_LEVEL=DEBUG`.

## Known limitations

- No authN/authZ, no per-user identity, no multi-tenant separation beyond `document_id` filtering.
- Rate limiting is not distributed. The PDF timeout cannot interrupt a single pathological page.
- Prompt-injection defence is structural + validation; wording of a real model can still be influenced,
  but scores, sources and percentages cannot be forged.
- Run `pip-audit -r requirements.txt` periodically (CI has an advisory, non-blocking job).
