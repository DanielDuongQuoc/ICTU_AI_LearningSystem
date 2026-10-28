# Deployment Runbook (Phase 21)

Single-instance Docker Compose deployment (`api` + `ollama`). Builds on DEPLOYMENT.md, RELEASE.md and OPERATIONS_RUNBOOK.md.
Steps marked **NOT VERIFIED** were not executed on a real host (no Docker/Ollama on the development machine).
Expected downtime of a deployment = API restart time: **NOT VERIFIED / not measured**. There is no zero-downtime mechanism.

## 1. Pre-deployment
1. Release candidate: `python -m scripts.validate_release --manifest release-manifest.json` -> verdict must not be `BLOCKED`.
   Read every `NOT VERIFIED`/`PARTIAL` line and accept it consciously.
2. Verify the artifact: manifest `release_id`, `application_version`, `ai.llm_model`/`embedding_model` (pinned tags),
   `build.requirements_sha256`; image built from exactly this source (`docker build`, NOT VERIFIED here).
3. Config: production `.env` (from `.env.example`): `APP_ENV=production`, pinned `OLLAMA_MODEL` / `OLLAMA_EMBEDDING_MODEL`,
   no `*` in `CORS_ALLOWED_ORIGINS`, `ADMIN_API_TOKEN` (>=16 chars) or empty, `OLLAMA_IMAGE=<tested tag>`, `API_BIND=127.0.0.1` behind a proxy.
   The app refuses to start on invalid config.
4. Ollama: models present (`docker compose exec ollama ollama list`); the API never pulls models. Pull with the exact tags first.
5. Disk: free space above `MIN_FREE_DISK_MB` on the Chroma volume (readiness enforces it).
6. **Back up** (DISASTER_RECOVERY.md): stop `api`, `scripts/backup.py`, copy `ai_versions_data`. Record the archive name + sha256.
7. Embedding model unchanged vs the running release? If it changes this is a migration (RELEASE_ENGINEERING.md), not a deploy.

## 2. Deployment
```
docker compose pull            # only if OLLAMA_IMAGE changed
docker compose build api
docker compose up -d           # recreates api; named volumes are reused
```
Readiness, not "running", decides success: `curl -fsS http://127.0.0.1:8003/ready` must return 200
(`services`: api, chroma, ollama, model, storage, embedding_index all `ok`). `/health/live` only proves the process is up.
Ollama slow to start: compose waits for the Ollama *service* health; a missing model shows as `model: error` in `/ready`.

## 3. Post-deployment
1. `python scripts/smoke_test.py --base-url http://127.0.0.1:8003` (needs live Ollama): health, ready, upload, retrieve, questions, evaluate, stream, delete.
2. Logs: `docker compose logs api --tail 200` - no `ERROR`, `request completed` lines carry the expected `config_version`.
3. Metrics: `python -m scripts.check_reliability --url http://127.0.0.1:8003/metrics` (INCONCLUSIVE until real traffic exists).
4. AI version: `python scripts/ai_version_cli.py show-env`; confirm the active version id is the intended one.
5. Retrieval on a **known** `document_id` returns that document only.
6. Restart check: `docker compose restart api` -> `/ready` 200 and the same retrieval still works (data persisted). NOT VERIFIED in Docker here; tested with separate processes (`tests/release/test_dr_and_restart.py`).

## 4. Rollback
Trigger: smoke test fails, `/ready` not 200 within a few minutes, 5xx share or timeouts clearly above the pre-release level,
`INDEXING_ERROR`/`embedding_index: error`, source-integrity or score-integrity problem.
| Problem | Action |
|---|---|
| Bad AI version only | `rollback` to the previous approved version (MODEL_PROMOTION_ROLLBACK.md); no redeploy |
| Bad application release | redeploy the previous image/commit with the same volumes and `.env` |
| Data damaged / embedding model changed | stop `api`, `scripts/restore.py --archive <pre-release backup> --target <chroma dir> --yes`, start the **previous** release |
Never `docker compose down -v` (deletes the volumes). Verify: `/ready` 200, smoke test, previous `release_id`/`config_version` in logs.

## 5. Escalation
Failed restore, suspected data loss, source leakage between documents: SEV-1 (INCIDENT_RESPONSE.md).
