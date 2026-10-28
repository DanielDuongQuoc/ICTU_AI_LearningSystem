# Release process

Version: `app/version.py` (`MODULE_VERSION`), shown by `/health` and `/openapi.json`. Module 3 is pre-1.0
(`0.x`); do not call it 1.0.0 until it has run against the real Ollama/Docker stack and Module 1/2 contracts
are confirmed.

## Release checklist
Automated (`python scripts/release_check.py [--run-tests] [--docker]`): `.env` ignored by git and docker, no
secrets / complete / valid `.env.example`, pinned requirements, Dockerfile non-root + no `.env` copy, compose
(`ollama` host, healthcheck, Chroma volume, Ollama on loopback), OpenAPI paths + descriptions, version
consistency, documentation present, backup/restore/smoke scripts present, no temporary artifacts, optional
`docker compose config`, optional pytest.

Manual (needs Docker + Ollama; not automated):
- [ ] `pip install -r requirements.txt` in a fresh venv, `pytest -q` green
- [ ] `pytest -m integration` against real Ollama
- [ ] `docker compose build` and `docker compose up -d`; `docker compose ps` healthy
- [ ] `GET /health/ready` = 200 after pulling both models
- [ ] `python scripts/smoke_test.py` all PASS
- [ ] `docker compose restart api`, then `GET /api/documents/{id}` still works (persistence)
- [ ] `python scripts/benchmark.py` / `scripts/evaluate_rag.py` compared with the previous release
- [ ] backup taken (see BACKUP_AND_RECOVERY.md) **before** deploying
- [ ] CHANGELOG-style notes written (API/behaviour changes, new env vars)

## Deploy
1. Back up Chroma. 2. Tag the build (`docker build -t module3-api:<version> .`; keep the previous image).
3. `docker compose up -d` with the new image. 4. `GET /health/ready`, then the smoke test.

## Rollback
```
issue detected -> stop rollout -> redeploy previous image/tag -> (restore Chroma backup only if data is bad)
              -> GET /health/ready -> smoke test
```
- Re-deploy the previous image: `docker compose up -d` with the old tag (e.g. `image:` override or rebuild the
  previous source revision).
- **Never** delete the `chroma_data` volume as part of a rollback. Only restore a backup if the data itself is
  damaged; `restore.py` keeps the replaced data next to the target.
- Data compatibility: this release changes no stored Chroma schema, so images are interchangeable. Any future
  schema change must stay backward-compatible or ship a documented migration.
- API changes in 0.14.0 that a rollback would undo: `INPUT_TOO_LONG` length limits, `GET /ready` alias,
  version string `0.14.0` (previously `1.0.0`).

## Known gaps before calling a release "production"
Docker build/up, real-Ollama integration tests, benchmarks and a full Module 1 + Module 2 end-to-end run have
not been executed in the development environment (see HANDOVER.md).


---

## Release record - 0.14.0 (validated 2026-10-24)

| Item | Status |
|---|---|
| Version | 0.14.0 (pre-1.0) |
| Date | 2026-10-24 |
| Git commit | n/a (not a git repository) |
| Python | 3.11.9 |
| Dependencies | pinned; clean-venv install OK |
| Docker | NOT VERIFIED (daemon unavailable); compose config valid |
| Tests | offline suite green (see FINAL_RELEASE_REPORT.md) |
| Integration (real Ollama) | NOT RUN |
| Module 1 / Module 2 E2E | PENDING (contracts unconfirmed) |
| Decision | NOT APPROVED - see docs/FINAL_RELEASE_REPORT.md |

### Release candidate checklist
Legend: PASS / FAIL / NOT APPLICABLE / NOT VERIFIED.

| Item | Status |
|---|---|
| Clean installation | PASS (fresh venv from requirements.txt) |
| Dependency installation | PASS |
| Configuration validation | PASS (19 invalid-env cases + startup refusal tested) |
| Unit tests | PASS |
| Integration tests (real Ollama) | NOT VERIFIED |
| Contract tests | PASS |
| E2E tests | NOT VERIFIED (contract E2E only; no Module 1/2) |
| API validation | PASS |
| OpenAPI validation | PASS |
| RAG regression | NOT VERIFIED (needs Ollama; no earlier baseline recorded) |
| Question generation | PASS offline / real model NOT VERIFIED |
| Evaluation | PASS offline / real model NOT VERIFIED |
| Streaming | PASS offline / real Ollama NOT VERIFIED |
| Ollama | NOT VERIFIED |
| ChromaDB | PASS (insert/query/delete/restart/empty/missing, tested) |
| Docker build | NOT VERIFIED |
| Docker Compose | PASS for `config`; `up` NOT VERIFIED |
| Persistence | PASS for Chroma directory reopen; Docker volume restart NOT VERIFIED |
| Backup | PASS (unit-tested); on Docker volume NOT VERIFIED |
| Restore | PASS (unit-tested); on Docker volume NOT VERIFIED |
| Security audit | PASS offline (no blocker); real-Qwen injection NOT VERIFIED |
| Performance smoke test | NOT VERIFIED (concurrency offline PASS; latency NOT RUN) |
| CI | PASS for local re-run of its steps; GitHub run NOT VERIFIED |
| Documentation | PASS |
| Git audit | NOT VERIFIED (no git); file-level cleanliness PASS |
