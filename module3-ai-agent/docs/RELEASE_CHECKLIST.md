# Release Checklist (Phase 21)

Statuses: `PASS`, `FAIL`, `BLOCKED`, `NOT VERIFIED`, `N/A`. A `NOT VERIFIED` item is never treated as PASS.
Re-run `python -m scripts.validate_release --manifest <path>` to refresh the automated rows.

| Item | Status | Evidence / reason |
|---|---|---|
| Git clean | NOT VERIFIED | The folder is not a git repository, so no commit id or clean-tree check exists. |
| Tests pass | PASS | `validate_release`: functional 710 passed, 9 deselected. |
| Security tests pass | PASS | 168 passed (`tests/security`). |
| AI quality gate | PARTIAL | Deterministic checks pass. Model-dependent quality is NOT VERIFIED (no live Ollama). |
| Reliability gate | PASS | 78 passed (`tests/reliability`), offline fakes only. |
| Docker build | NOT VERIFIED | Docker daemon not running. `docker compose config -q` is valid. |
| Smoke test | PASS (offline) | In-process `run_smoke` with fakes. Live Ollama run is NOT VERIFIED. |
| OpenAPI verified | PASS | Contract-freeze tests in `tests/release`. Known extras: `/search`, `/api/ai-versions*`. |
| Config verified | PASS | Production validation tests, including pinned model tags. |
| Ollama verified | NOT VERIFIED | Ollama is not installed on this machine. |
| ChromaDB verified | PASS (test data) | Persistence across process restarts and embedding-compat tests. |
| Backup verified | PASS (test data) | DR drill and `tests/release`. Production data NOT VERIFIED. |
| Restore verified | PASS (test data) | DR drill: restore, query and compare. Production data NOT VERIFIED. |
| Rollback verified | PARTIAL | AI version rollback verified at runtime (mock transport). Docker image rollback NOT VERIFIED. |
| Documentation updated | PASS | `release_check` requires all Phase 21 docs. |
| Release manifest generated | PASS | `docs/release-manifest-phase21.json`. Git commit is recorded as unavailable. |
