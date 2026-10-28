# Release Engineering (Phase 21)

Reuses the Phase 14/15/16/18/19/20 mechanisms; this page only describes what Phase 21 added.

## Deployment model
**Single instance, restart deployment** (Docker Compose: `api` + `ollama`). Expected downtime during a release =
the API restart time. **NOT VERIFIED / not measured**: no Docker on the development machine. There is no blue-green,
no rolling update and no zero-downtime claim. State (Chroma, AI-version registry, uploads) lives in named volumes
(`chroma_data`, `ai_versions_data`, `uploads_data`), so replacing the container does not touch data.

## Versions (three different things)
| Identity | Source | Example |
|---|---|---|
| Application version | `app/version.py` (`MODULE_VERSION`, pre-1.0 convention kept) | `0.14.0` |
| AI configuration | Phase 18 `version_id` + `config_fingerprint` (models, prompts, RAG, chunking) | `env-bootstrap` or `ai-...` |
| Source | git commit (`unavailable` when the folder is not a git repository, as today) | |
Rules: breaking API change -> bump MAJOR (still `0.x`: bump MINOR and document), compatible feature -> MINOR, fix -> PATCH.
The OpenAPI `info.version` must equal `MODULE_VERSION` (tested).

## Release manifest (`python -m scripts.release_manifest`)
`release_id = <app version>+<12 hex of sha256(version, git commit, requirements, Dockerfile, AI config fingerprint)>`.
Same inputs -> same id; the timestamp is recorded but excluded. Contains models, prompt versions, config/RAG fingerprints,
`model_tags_pinned`, Python version, dependency/Dockerfile hashes, base images and **gate results, all `NOT VERIFIED`
unless `scripts/validate_release.py` filled them in**. Secrets are never read; a guard rejects secret-looking values.

## Gate aggregator (`python -m scripts.validate_release [--manifest PATH]`)
Gates: `static_checks`, `functional`, `security` (Phase 19), `reliability` (Phase 20), `release_engineering` (Phase 21),
`ai_quality` (deterministic only: PARTIAL at best offline), `docker` (NOT VERIFIED without Docker).
Verdict: **BLOCKED** if any gate FAILS (a failing security gate blocks even when quality passes); **GO** only when all gates
are PASS; otherwise **GO WITH LIMITATIONS**. NOT VERIFIED / PARTIAL is never promoted. A quality override is not
implemented here: the Phase 18 governance rules apply and nothing in these scripts bypasses them.

## Model immutability
* `APP_ENV=production` refuses `OLLAMA_MODEL` / `OLLAMA_EMBEDDING_MODEL` that are untagged or `:latest` (startup fails).
  Example pins: `qwen2.5:7b`, `nomic-embed-text:v1.5` - confirm with `ollama list` on the host.
* A tag can still be re-pushed by the registry; for byte-level pinning record the digest from `ollama list` in the release notes (NOT automated).
* `OLLAMA_IMAGE` defaults to `ollama/ollama:latest` in `docker-compose.yml`: **production must set `OLLAMA_IMAGE` to a tested tag** (documented risk, not forced).
* The Dockerfile base is pinned to `python:3.11.9-slim`. Not digest-pinned; the build itself is NOT VERIFIED here.

## Vector compatibility (embedding model)
Chunks written from Phase 21 on carry `embedding_model` in their metadata. Then:
* indexing with a different embedding model is **refused** (`INDEXING_ERROR`, "re-index is required");
* `/ready` -> 503 with `services.embedding_index = "error"` when stored vectors do not match the configured model (sampled: first 50 chunks);
* Phase 18 already blocks promoting an AI version with another embedding model.
Changing the embedding model is therefore a **migration**, not a config tweak: create a new collection (`CHROMA_COLLECTION_NAME`),
re-upload documents, switch, keep the old collection/backup for rollback. Chunks indexed *before* Phase 21 have no tag:
their compatibility **cannot be verified** automatically (the check treats them as unknown, not as mismatched).

## Rollback
* **AI version**: `rollback` (Phase 18) re-activates a previously approved version; verified that the next Ollama request
  really carries the rolled-back model, manifests/audit survive and ChromaDB is untouched.
* **Application**: redeploy the previous image/commit with the same volumes and `.env`; data is untouched because it is in volumes.
  The code-level rollback itself (Docker) is NOT VERIFIED here; the data-preservation half is covered by the restart and
  restore tests. If the failed release changed the embedding model, restoring the matching backup/collection is part of the rollback.

## Git / repository hygiene
`.gitignore` and `.dockerignore` exclude `.env`, `data/chroma/*`, `uploads/*`, `backups/`, caches (tested). **This folder is not a
git repository**, so "git clean", the commit id and accidental-commit checks are NOT VERIFIED. `scripts/` is excluded from the image
by `.dockerignore`: run backup/restore from a host checkout against the volume (see DISASTER_RECOVERY.md).
