# Security

Scope: Module 3 as an internal service behind a trusted network. **There is no authentication or authorisation on
the data API** (upload / RAG / questions / evaluations), and the admin plane (`/api/ai-versions`) is protected only by
a single shared `X-Admin-Token`. There is no role-based access control and no per-user identity. Restrict network
access (reverse proxy / firewall / compose network) and do not expose port 8003 publicly. CORS is *not*
authentication.

Related: [THREAT_MODEL.md](THREAT_MODEL.md), [DATA_PRIVACY.md](DATA_PRIVACY.md), [PHASE19_REPORT.md](PHASE19_REPORT.md),
[ERROR_MATRIX.md](ERROR_MATRIX.md). Security tests live in `tests/security/` (and earlier `tests/test_security_isolation.py`).

## Trust boundaries

| Trusted (backend decides) | Untrusted (treated as data, never as instruction) |
|---|---|
| application code, fixed system prompts, configuration | uploaded PDF bytes, extracted text, chunks / retrieved context |
| `document_id`, `chunk_id`, `page`, similarity score, `sources`, `percentage`, `max_score`, `question_id` | question, student answer, expected answer, filenames, ids |
| validated score (range / finite / type) | **everything the LLM returns** (JSON, score, feedback, metadata) |

## Controls

| Area | Control |
|---|---|
| Secrets | none required; `.env` git/docker-ignored; `.env.example` has no secret values; `ADMIN_API_TOKEN` empty = admin plane disabled, min 16 chars when set; token compared in constant time, never logged |
| Upload | `.pdf` + MIME + `%PDF` signature, streamed size cap (`MAX_UPLOAD_SIZE_MB`), empty/malformed/truncated PDF rejected, `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS_PER_DOCUMENT`, `PDF_PROCESSING_TIMEOUT_S` (cooperative, checked per page), `MAX_CONCURRENT_UPLOADS` (reject, never queue) |
| Paths | client filename sanitised and never a path; temp file named from a server UUID; ids are only Chroma metadata filter values, never filesystem paths |
| Input limits | question/query <= `MAX_QUESTION_CHARS`, answers <= `MAX_ANSWER_CHARS`, `top_k` <= `MAX_TOP_K` (also for `/documents/{id}/search`), 1-20 questions, `0 < max_score <= 100`, JSON body <= `MAX_REQUEST_BODY_BYTES` (413) |
| Rate limiting | per-client sliding window on every expensive route incl. the SSE stream (`RATE_LIMIT_PER_MINUTE`, 0 = off) -> 429 + `Retry-After`; plus the existing LLM concurrency gate |
| Prompt injection | PDF text, chunks, question and answers are delimited data in the user message; delimiter forgery neutralised; system prompt is fixed and never contains them |
| Context-only | no sufficient context -> `INSUFFICIENT_CONTEXT`, **no LLM call** |
| Source / score integrity | all metadata and scores are backend-derived; LLM-supplied metadata is dropped/ignored; non-finite, out-of-range, wrong-type or missing scores are rejected |
| LLM output | parsed with a strict parser (rejects NaN/Infinity, duplicate keys, deep nesting), size-capped (`MAX_LLM_OUTPUT_CHARS`), Pydantic-validated, free text clamped, **verbatim system-prompt text rejected** (`LLM_OUTPUT_REJECTED`, 8-word shingle match) |
| SSE | `error -> done` contract on failure; wall-clock cap `STREAM_MAX_DURATION_S`; output cap; upstream generator closed on disconnect / cancellation |
| Isolation | every retrieval is filtered by `document_id`; deletion removes only that document's vectors |
| Errors | one JSON schema, no stack traces / paths / prompts; validation errors never echo submitted values |
| Logging | request id, endpoint, status, latency, error code; never bodies, answers, prompts, PDFs, LLM text, tokens |
| HTTP | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, CSP `default-src 'none'` (not on `/docs`); CORS off unless `CORS_ALLOWED_ORIGINS` set; credentials never allowed |
| Admin plane | token required; client cannot send `status`; promotion only through the Phase 18 quality gate; 10 bad tokens / 5 min locks the client out (429) |
| Config | validated at startup (fail fast): limits > 0; in production no `*` CORS origin and no `LOG_LEVEL=DEBUG` |
| Container | non-root user, `no-new-privileges`, `cap_drop: [ALL]`, no `.env` baked in, Ollama published on loopback only, writable dirs limited to chroma / ai_versions / uploads; `API_BIND` can restrict the API to loopback |

## Dependency security (what was and was not checked)

`pip-audit` was run once (Phase 19) against `requirements.txt` with `--no-deps`: **transitive dependencies were NOT
audited.** Findings and actions are recorded in [PHASE19_REPORT.md](PHASE19_REPORT.md). An advisory, non-blocking `dependency-audit` job was added to `.github/workflows/ci.yml` (needs internet; direct
requirements only; NOT executed locally as a workflow). Run `pip install pip-audit && pip-audit -r requirements.txt`
periodically; it needs internet access.

## Known limitations (not fixed, not verified)

- No authentication/authorisation on the data API; anyone who can reach the API can read any document whose
  (random UUID4) id they know. Admin plane = one shared token, no roles, no per-user audit identity beyond the
  self-reported `X-Admin-Actor` label.
- Rate limiting is in-process: single instance only, resets on restart, keyed by TCP peer address
  (`X-Forwarded-For` deliberately not trusted, so clients behind one proxy share a bucket). Not distributed.
- The PDF timeout is cooperative (between pages); a single pathological page inside PyMuPDF/MuPDF cannot be
  interrupted. Memory use of the parser is not capped beyond the upload/page/text limits.
- Prompt-injection defence is prompt structure + validation + output checks. A real model may still be influenced
  in wording; score range, sources and percentage cannot be forged. Real-Qwen behaviour is NOT verified (no Ollama
  in the development environment). The prompt-leak check only catches verbatim (>= 8 consecutive words) leakage.
- Chroma 0.5.23 has advisories with no fixed version listed by the audit tool (see report).
- `ollama/ollama:latest` and `python:3.11-slim` float unless pinned; no image scanning was run (Docker unavailable).
- No external penetration test, WAF, TLS termination (reverse proxy responsibility) or real malicious-PDF corpus.
