# Phase 19 report - AI security, privacy, abuse prevention and safety hardening

**Status: PARTIAL.** All code, tests and documentation items were completed and the offline regression passes
(878 passed). It is PARTIAL (not PASS) because several things could not be verified in this environment: real-model
behaviour (no Ollama), Docker build/runtime (no Docker), a full dependency audit (transitive deps; Chroma advisories
without fixes), and the data API still has **no authentication/authorisation** (inherited, documented, not fixed).
This is not a penetration test and does not make the module "secure" or "production-ready".

Environment: Windows, Python 3.11.9, venv `.venv`, no git repo, no Docker, no Ollama, no Module 1/2. Internet was
available (pip, pip-audit).

## A. Security audit summary

Severity is my own assessment, with the reason given.

| Severity | Count | Items |
|---|---|---|
| Critical | 0 | none found (no remote code path from input to code/file execution was identified) |
| High | 2 | H1 (fixed), H2 (open, inherited) |
| Medium | 8 | M1-M8 (fixed except M8 partly) |
| Low | 3 | L1-L3 (fixed) |
| Informational | 3 | I1-I3 |

## B. Issues found

| ID | Severity | Component | Description / impact | Fix | Verification | Status |
|---|---|---|---|---|---|---|
| H1 | High | PDF upload | No cap on pages, extracted text, chunks or processing time: one valid-looking PDF can exhaust CPU/RAM/Chroma (DoS; reachable by anyone with API access). | `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS_PER_DOCUMENT`, `PDF_PROCESSING_TIMEOUT_S`, `MAX_CONCURRENT_UPLOADS`; new codes `DOCUMENT_LIMIT_EXCEEDED` 413, `PDF_PROCESSING_TIMEOUT` 504 | `test_upload_security.py` | Fixed. Residual: timeout is cooperative per page |
| H2 | High | Data API | No authn/authz: any caller can read/delete any document by id. Inherited from earlier phases. | Not fixed (needs an identity provider / gateway; out of scope). Documented in SECURITY.md, THREAT_MODEL.md | n/a | **Open** |
| M1 | Medium | `/documents/{id}/search` | `top_k` hard-coded <= 100 (not `MAX_TOP_K`), query length unbounded | uses `MAX_TOP_K`, `MAX_QUESTION_CHARS` | `test_top_k_is_validated_everywhere`, `test_question_and_query_length_capped` | Fixed |
| M2 | Medium | JSON parsing | `safe_parse_json` accepted NaN/Infinity and duplicate keys | strict parser | `test_safe_parse_json_rejects_*` | Fixed |
| M3 | Medium | LLM output | no size cap on raw/streamed output, no wall-clock cap on stream | `MAX_LLM_OUTPUT_CHARS`, `STREAM_MAX_DURATION_S`, `LLM_OUTPUT_REJECTED` 502 | `test_sse_security.py`, `test_llm_output_size_cap` | Fixed |
| M4 | Medium | AI output | model could echo the system prompt; response text fields unbounded | verbatim 8-word shingle check (RAG answers, questions, evaluation, stream) + clamps | `test_verbatim_system_prompt_*`, `test_oversized_feedback_is_clamped_*` | Fixed. Residual: only verbatim leakage detected |
| M5 | Medium | HTTP | no JSON body size cap | `MAX_REQUEST_BODY_BYTES` (413; declared length answered without reading the body, streamed bodies cut off) | `test_oversized_json_body_*`, `test_oversized_declared_content_length_*` | Fixed. (First implementation returned 400 because FastAPI wraps body errors; found by the new test and corrected) |
| M6 | Medium | API abuse | no rate limiting | in-process sliding window on all expensive routes incl. SSE, `RATE_LIMITED` 429 + `Retry-After` | `test_rate_limit_*`, `test_forged_forwarded_for_*` | Fixed for single instance only |
| M7 | Medium | Admin plane | no token minimum length, no brute-force throttle, no failure log | >= 16 chars, lockout after 10 failures/5 min (429), failure logged without the value | `test_brute_force_is_locked_out_*`, `test_admin_token_never_logged_*` | Fixed. Residual: single shared token, no roles |
| M8 | Medium | Dependencies | `pip-audit` (direct deps, `--no-deps`): 19 findings in 4 packages | upgraded `python-multipart` 0.0.20 -> 0.0.32 and `anyio` 4.7.0 -> 4.15.1 (full suite re-run, `pip check` clean) | see section D | **Partly fixed**: `chromadb 0.5.23` (3 advisories, no fix version listed) and `pytest 8.3.4` (fix needs 9.x, not trialled; pytest-asyncio 0.24 compatibility unknown) not changed |
| L1 | Low | HTTP | no security headers | nosniff, DENY, no-referrer, CSP `default-src 'none'` (not on /docs) | `test_security_headers_*`, `test_docs_not_broken_by_csp` | Fixed |
| L2 | Low | Config | no production guards | startup validation: positive limits, no `*` CORS / DEBUG logging in production | `test_insecure_configuration_is_rejected` | Fixed |
| L3 | Low | Docker | no `cap_drop`; `ai_versions` dir not a volume / not created for the non-root user; `.gitignore` missed `data/ai_versions` | `cap_drop: [ALL]`, `API_BIND`, `ai_versions_data` volume, `mkdir`+`chown` in Dockerfile, `.gitignore` entry | `test_compose_hardening`, `test_dockerfile_*`, `test_gitignore_*` (static checks only) | Fixed (not run in Docker) |
| I1 | Info | Image | pytest etc. are installed in the runtime image (single `requirements.txt`) | not changed | - | Open |
| I2 | Info | Image | `python:3.11-slim`, `ollama/ollama:latest` float | `OLLAMA_IMAGE` can pin; not pinned by default | - | Open |
| I3 | Info | Compose | API port published on all interfaces by default | `API_BIND` added (default unchanged to keep behaviour) | - | Open (by design) |

Already adequate and reused (not rewritten): extension/MIME/signature/size checks, filename sanitisation, UUID temp
names, prompt delimiting, backend-derived sources/percentage, strict score validation, uniform error schema,
request logging without bodies, CORS off by default, non-root container, LLM concurrency gate, SSE `aclosing`
cleanup.

## C. Files changed

| File | Change | Reason |
|---|---|---|
| `app/core/config.py`, `.env.example` | new limits + validators | H1, M3, M5, M6, M7, L2 |
| `app/core/exceptions.py`, `app/core/openapi.py`, `app/core/error_handlers.py`, `docs/ERROR_MATRIX.md` | 5 error codes, 429, 413 mapping | new rejection paths |
| `app/services/pdf_service.py`, `app/services/document_service.py` | page/text/chunk/time limits | H1 |
| `app/core/rate_limit.py` (new) | limiter, upload gate | M6 |
| `app/core/output_guard.py` (new) | size, leak, clamp | M3, M4 |
| `app/core/middleware.py` | body cap, security headers | M5, L1 |
| `app/api/documents.py`, `rag.py`, `questions.py`, `evaluations.py`, `ai_versions.py` | rate limits, search limits, admin lockout | M1, M6, M7 |
| `app/utils/json_utils.py`, `app/services/llm_service.py`, `rag_service.py`, `question_service.py`, `evaluation_service.py` | strict parsing, output guards, stream caps | M2-M4 |
| `Dockerfile`, `docker-compose.yml`, `.gitignore` | hardening | L3 |
| `requirements.txt` | `python-multipart==0.0.32`, `anyio==4.15.1` | M8 |
| `../.github/workflows/ci.yml` | advisory `dependency-audit` job (internet, non-blocking, direct deps only) | M8 |
| `scripts/release_check.py` | require new docs | documentation gate |
| `tests/conftest.py` | autouse reset of the global limiter | rate limiter is process state; without it 79 earlier tests got 429 |
| `tests/security/*` (new, 5 files + conftest) | 168 tests | Phase 19 |
| `docs/SECURITY.md`, `DATA_PRIVACY.md`, `THREAT_MODEL.md`, `PHASE19_REPORT.md` | docs | Phase 19 |

## D. Tests

| Command | Result | Environment |
|---|---|---|
| `pytest -q` (whole suite: Phases 1-18 + `tests/security`), after all changes, with upgraded dependencies | **878 passed**, 9 deselected (integration, need Ollama), 5 warnings, 158 s | Windows, Python 3.11.9, fakes for Ollama, in-memory Chroma |
| `pytest tests/security` | 168 passed, 32 s | same |
| `ruff check --select E9,F app scripts tests` | All checks passed | same |
| `python scripts/release_check.py` | 10/10 PASS | same |
| `pip check` after the upgrade | No broken requirements | same |
| `pip-audit -r requirements.txt --no-deps --disable-pip` (before upgrade; run once from a temporary install, since removed) | 19 findings in 4 packages (python-multipart, chromadb, pytest, anyio); the exact per-package split was not recorded | internet |

The baseline before Phase 19 was 710 passed; 710 + 168 = 878. Runtime of the full suite varied between 63 s and
257 s across runs on this machine (unrelated to the changes; not investigated further).

**Not re-run after the final dependency pins:** `pip-audit` (so the claim "python-multipart/anyio findings resolved"
rests on the advisory fix versions reported by the tool, not on a second scan).

### Security test matrix (all executed above)

| Area | Attack/Test | Expected | Test |
|---|---|---|---|
| Upload | oversized file | 413 | `test_oversized_upload_rejected` |
| Upload | invalid / truncated / junk / empty / wrong ext / wrong MIME | 4xx, no trace | `test_invalid_uploads_rejected_safely` |
| Upload | too many pages / too much text / too many chunks / timeout | 413 / 413 / 413 / 504 | `test_too_many_pages_*`, `*_text_*`, `*_chunks_*`, `*_timeout_*` |
| Upload | dangerous / unicode / long filenames, collision | never a path, distinct ids | `test_dangerous_filenames_*`, `test_same_filename_*` |
| Upload | concurrent upload flood | 503 then recovers | `test_upload_gate_*` |
| Path | `../`, `..\`, `/etc/passwd`, `C:\Windows`, null byte, encoded | 400/404/422 on all routes | `test_path_traversal.py` |
| Chroma | filter-injection ids (`{"$ne":..}`) | 422 | `test_chroma_filter_cannot_be_injected_via_id` |
| RAG/PDF | 15 injection strings (EN/VI/ZH/ES/RU, role, score, page, delimiter) in the PDF | stay in user data, never in system prompt | `test_poisoned_document_stays_data_*` |
| Evaluation | same strings as student answer | data only | `test_student_answer_injection_*` |
| Poisoning | document says "full marks / page 999" | score 8 (from validated model output), page 1 | `test_poisoned_document_cannot_change_*` |
| Source | fake page/chunk_id/document_id from LLM (eval, answer, questions) | dropped / rejected | `test_llm_fabricated_*` |
| Score | -1, >max, NaN, +-Inf, str, bool, None, list, dict, missing | rejected (>= 400) | `test_invalid_scores_*`, `test_missing_score_*` |
| Output | wrong-typed fields, prompt leak, oversized feedback | rejected / clamped | `test_wrong_typed_*`, `test_verbatim_*`, `test_oversized_feedback_*` |
| Parser | NaN, Infinity, duplicate keys, junk, deep nesting | rejected | `test_safe_parse_json_*` |
| API | huge / zero / negative / string `top_k`; long question; huge `number_of_questions`; bad `max_score` | 400/422, no LLM call | `test_resource_and_error_security.py` |
| API | oversized JSON body (declared + counted) | 413 | `test_oversized_*body*` |
| API | rate limit, all expensive routes incl. SSE, forged X-Forwarded-For | 429 + Retry-After | `test_rate_limit_*` |
| SSE | endless / oversized / broken / leaking stream | `error -> done`, upstream closed | `test_sse_security.py` |
| SSE | client disconnect, task cancellation | upstream generator closed | `test_client_disconnect_*`, `test_task_cancellation_*` |
| Error | unexpected exception, validation, malformed JSON | generic schema, no echo, no paths | `test_unexpected_exception_*`, `test_validation_error_*` |
| Logs | answer/document/question canaries, admin token | not in logs | `test_logs_never_contain_*`, `test_admin_token_never_logged_*` |
| Secrets | pattern scan of source/docs/tests/config; `.env.example`; ignore files | none found | `test_no_hardcoded_secrets_*` (heuristic) |
| Docker | non-root, no `.env` copy, cap_drop, loopback Ollama | present (static) | `test_dockerfile_*`, `test_compose_hardening` |
| Chroma | A/B isolation, delete only own vectors, deleted doc | isolated / 404 / no LLM call | `test_every_read_path_*`, `test_deleting_*` |
| Version | no/wrong token, brute force, injected `status`, unknown promote, traversal ids | 401 / 429 / 422 / 404 | admin tests in `test_isolation_secrets_admin.py` + Phase 18 gate tests |
| Config | zero/negative limits, short token, prod `*` CORS / DEBUG | rejected at startup | `test_insecure_configuration_is_rejected` |

## E. Security limitations - NOT VERIFIED

- External penetration testing, WAF, TLS termination.
- Real malicious-PDF corpus / fuzzing of PyMuPDF; memory ceilings of the parser; pathological single-page stall.
- Behaviour of the real Qwen model against injections (no Ollama); the prompt-leak guard only catches verbatim text.
- Docker image build, container hardening at runtime, image CVE scan (no Docker).
- Distributed rate limiting (limiter is single-process, per TCP peer).
- Transitive dependency audit; Chroma 0.5.23 and pytest advisories remain; no second `pip-audit` after the pin
  changes; the new CI job was not executed.
- No authentication/authorisation on the data API; admin plane has a single shared token, no RBAC (see
  SECURITY.md / THREAT_MODEL.md). Secret scan is regex-based, not gitleaks, and there is no git history to scan.
- The secret-scan test ignores matches whose value contains words such as "test"/"example"; a real secret
  containing those words would not be flagged.

Stopping here as requested: Phase 20 is not started.
