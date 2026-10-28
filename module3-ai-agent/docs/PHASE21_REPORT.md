# Phase 21 Report: Production Deployment, Release Engineering, DR and Go-Live Validation

## A. Release status

**GO WITH LIMITATIONS.** This is not "production ready". Several gates are PARTIAL or NOT VERIFIED (section L). Release verdict is computed by `python -m scripts.validate_release`. Any FAIL gives BLOCKED.

## B. Build

- Git commit: NOT VERIFIED (not a git repository).
- Application version, AI config fingerprint and release id: see `docs/release-manifest-phase21.json`.
- Docker: base image pinned to `python:3.11.9-slim`. Image build NOT VERIFIED (daemon not running). `compose config` is valid.
- Dependencies: existing pins reused, nothing upgraded.

## C. Functional

710 passed, 9 deselected (`validate_release`). Full run: functional 710 + security 168 + reliability 78 + release 53.

## D. Security

PASS: 168 tests. A new hardcoded-secret scan finding was caused by test literals I added. I removed them. External penetration test: NOT VERIFIED.

## E. AI Quality

PARTIAL. Deterministic checks pass. Real model quality (Qwen via Ollama) NOT VERIFIED.

## F. Reliability

PASS (78 tests) using fakes. Ollama restart is tested via fakes only. A real Ollama restart is NOT VERIFIED.

## G. Deployment

- Offline smoke test passes (in-process, fakes).
- Restart persistence passes using separate processes.
- Live Docker, compose startup and live Ollama are NOT VERIFIED.
- Deployment model: single instance, restart deployment. Downtime NOT VERIFIED. No zero-downtime claim.

## H. Backup

`scripts/backup.py` and `scripts/restore.py` already existed and were reused. Tests cover checksum, corrupted-backup rejection, `--yes` requirement and no overwrite. Verified on test data only.

## I. Disaster Recovery

| Metric | Target | Observed | Verified |
|---|---|---|---|
| RPO | See `docs/DISASTER_RECOVERY.md` | N/A | NOT VERIFIED |
| RTO | See `docs/DISASTER_RECOVERY.md` | about 1.85 s (restore + first query, test data, local disk) | NOT VERIFIED for production |

The drill (`python -m scripts.dr_drill`) passed on test data: ingest, backup, destroy, restore, query, compare.

## J. Rollback

- AI version rollback: PASS. The model sent to Ollama was checked at runtime.
- Application rollback: data preservation tested. Docker image rollback NOT VERIFIED.

## K. Known limitations

- Production now rejects untagged or `latest` models. The default `OLLAMA_IMAGE` is still `ollama/ollama:latest`, so it is documented but not forced.
- Backup/restore scripts run from a host checkout (`scripts/` is excluded from the image).
- Added embedding-model tracking: indexing refuses to mix incompatible vectors, and `/ready` reports a mismatch. Legacy untagged chunks are treated as compatible.
- `/api/documents/{id}/search` and `/api/ai-versions*` exist but are not in the prompt's list.
- Pre-existing flaky security assertion: a random chunk uuid could contain "999". Fixed in `test_poisoned_document_cannot_change_score_or_sources`. Similar `"999" not in r.text` assertions in the same file could flake rarely.
- CI job `release-validation` was added. It has not been run on GitHub.

## L. NOT VERIFIED

- Git cleanliness and commit id.
- Docker image build and container runtime.
- Live Ollama: model availability, real quality, restart.
- Production backup, restore, RPO and RTO.
- Docker-level application rollback.
- CI execution on GitHub.
- External penetration test and production traffic SLOs.
