# Phase 18 Report - AI Governance, Versioning, Quality-Based Promotion & Rollback

**Status: PARTIAL.** The governance machinery is implemented and tested offline (710 passed). Nothing
model-dependent could be verified: no Ollama, no Docker, no real baseline exists. See "Not verified".

## 1. Audit summary (before coding)

* Phases 1-17 were green (659 tests). Prompts were constants in `app/core/prompts.py` with version labels but
  no registry/fingerprint; model, top_k, threshold and context limits were read from `Settings` at service
  construction; no versioning, rollback or audit log existed; no authentication existed.
* Phase 17 supplied datasets, evaluators, metrics and `scripts/quality_gate.py`; `evaluation/baselines/` is
  **empty** (no model-backed baseline).
* The Docker image contains only `app/`, so gate logic had to live in `app/`; evaluation stays host-side.
* Reused: Phase 17 evaluators/metrics/datasets, `AppError`/error matrix, request-ID middleware, settings
  pattern, router pattern. Nothing duplicated.

## 2. Files

Created: `app/core/prompt_registry.py`, `runtime_config.py`, `execution_trace.py`;
`app/schemas/ai_version.py`; `app/repositories/ai_version_repository.py`;
`app/services/promotion_gate.py`, `ai_version_service.py`; `app/api/ai_versions.py`;
`scripts/ai_version_cli.py`; `tests/test_phase18_governance.py` (51 tests); five docs
(`AI_VERSION_GOVERNANCE`, `PROMPT_VERSIONING`, `QUALITY_GATES`, `MODEL_PROMOTION_ROLLBACK`, this report).

Modified (minimal seams): `config.py` (3 settings), `exceptions.py` (10 codes), `openapi.py`, `middleware.py`
(per-request pin + trace headers), `llm_service.py` / `rag_service.py` / `question_service.py` /
`evaluation_service.py` (read effective config; explicit constructor values still win, so Phase 17 tests are
unchanged), `main.py` (router + startup load), `.env.example`, `docs/ERROR_MATRIX.md`,
`scripts/release_check.py` (required docs), `.github/workflows/ci.yml` (prompt-lock + snapshot step).
Test/tooling fixes: the secret regex in `release_check.py` and `tests/test_docker_config.py` used `\s*`, which
spans newlines and falsely flagged an empty `KEY=` line; now `[ \t]*`.

## 3. Architecture

File store (`AI_VERSIONS_DIR`: immutable manifests, atomic `state.json` with revision counter, append-only
`audit.jsonl`, `evidence/`), lifecycle `draft -> validated -> approved -> active` with `inconclusive`,
`rejected`, `superseded`, `rolled_back`. Promote/rollback = preflight (PASS evidence hash intact, Ollama
reachable, both models installed, prompt content unchanged, embedding/chunking equal environment, manifest
fingerprint intact) + static smoke test inside one lock-protected transaction with the audit event; stale
revision => 409. Embedding/chunking changes are refused at creation. Requests pin the effective config at start
and expose `X-AI-Config-Version/-Fingerprint`; services log `app.ai_trace` lines. Details in the docs.

## 4. API (new, admin token `X-Admin-Token`; 503 if `ADMIN_API_TOKEN` unset)

`GET/POST /api/ai-versions`, `GET /api/ai-versions/{id}`, `GET /active`, `GET /audit`,
`POST /{id}/evaluate | approve | reject | promote`, `POST /rollback`. Existing endpoints unchanged
(only two extra response headers). No request body can carry a status.

## 5. Verification actually run (Windows, Python 3.11.9, Ollama mocked)

| Command | Result |
|---|---|
| `pytest -q --no-header -p no:cacheprovider -o log_cli=false` (full) | **710 passed, 9 deselected** (659 prior + 51 new) |
| `pytest tests/test_phase18_governance.py` | 51 passed |
| `ruff check --select E9,F app scripts tests` | All checks passed |
| `python scripts/release_check.py` | all 10 checks PASS |
| `python scripts/ai_version_cli.py show-env` | prints the environment snapshot |

During the work three regressions appeared (new settings flagged by the secret regex, a hardcoded model name in
a schema description, undocumented error codes/OpenAPI descriptions); all were fixed and the suite re-run.

New tests cover: gate PASS/BLOCKED/INCONCLUSIVE/FAIL_* paths and tolerance direction; fingerprint stability;
prompt immutability, lock and lint; no status in create body, secret-text rejection; embedding/chunking refusal;
cannot approve/promote without PASS; evidence tampering; promote, supersede, rollback, rollback to `.env`,
rollback refused for unapproved target and for a deleted model (state/revision unchanged), Ollama unreachable,
prompt drift; two concurrent promotions (exactly one wins, other `AI_VERSION_CONFLICT`); held lock; aborted
transaction persists nothing; restart/worker refresh; tampered manifest => degraded, config kept; token absent
from logs/manifests/audit; API auth, end-to-end flow with trace header and audit; offline CLI run cannot
validate a version.

## 6. Not verified / limits (honest list)

* **No real-model evaluation was run** (no Ollama). `--mode ollama`, the `ollama-quality` workflow, and
  the real-model gate PASS path are exercised only with synthetic report fixtures. **NOT VERIFIED.**
* **No baseline exists**; default policy has no floors, so the first real promotion needs an operator-defined
  `AI_GATE_POLICY_FILE`. Thresholds are therefore *not* calibrated to a measured baseline.
* **No statistical significance** is computed; tolerance is a fixed margin; datasets are synthetic-labelled.
* Smoke test is static (no live LLM call); live behaviour after promotion is **NOT VERIFIED**.
* Concurrency tested in one process (asyncio + lock file); multi-process/multi-host races **NOT VERIFIED**.
* The `app.ai_trace` log content (chunk ids, latency) is implemented but the log lines themselves are not
  asserted by a test; the response headers are.
* Docker: the image/compose were not built; `AI_VERSIONS_DIR` has no volume mount added to compose, so inside a
  container the store is ephemeral unless a volume is configured. **NOT VERIFIED / TODO.**
* Auth is one shared token, no roles, no separation of duties; `X-Admin-Actor` is a label. Keep the admin API
  off public networks. README and `docs/SECURITY.md` were not updated.
* No v2 prompt ships; the registry is exercised with test-only versions. Embedding/chunking migration
  (re-embedding into a new collection) is **not implemented** by design. No automatic rollback by design.
* Ollama tags are mutable; model weights cannot be pinned or restored by this system.

## 7. Next steps (need your decision)

1. On a host with Ollama: create a draft from `show-env`, run `evaluate --mode ollama`, store the first real
   evidence, and set measured floors in `AI_GATE_POLICY_FILE`.
2. Add a compose volume for `AI_VERSIONS_DIR` and include it in backup scripts.
3. Update README/SECURITY with the admin-token note.

Stopped here as instructed; Phase 19 not started.
