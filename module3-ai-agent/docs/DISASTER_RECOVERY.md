# Disaster Recovery (Phase 21)

Extends BACKUP_AND_RECOVERY.md (Phase 14). Scripts: `scripts/backup.py`, `scripts/restore.py`, drill: `scripts/dr_drill.py`.

## What to back up
| Data | Where | Frequency (proposal) | Retention (proposal) |
|---|---|---|---|
| ChromaDB | volume `chroma_data` (`/app/data/chroma`) | daily + before every release | 14 daily, 4 weekly (manual: `rotate_backups.py` is explicit, nothing auto-deletes) |
| AI version registry, manifests, audit | volume `ai_versions_data` | before every promote/rollback and release | same |
| Configuration | `.env` (secrets!) | store in the operator's secret manager, never in git | n/a |
| Release manifest | `release-manifest.json` per release | per release | keep all |
| Uploads | volume `uploads_data` (transient temp copies) | not required: source PDFs belong to Module 1/2 | n/a |
Not backed up: logs (rotate instead), temp files, secrets in git. Encryption at rest of archives is **not** done by the scripts:
store backups on an encrypted volume/bucket (operator responsibility, NOT VERIFIED).

## Backup (consistent only with writers stopped)
`docker compose stop api` -> `python scripts/backup.py --source <path to chroma volume> --dest-dir backups` -> `docker compose start api`.
Output: `chroma-backup-<timestamp>.tar.gz` + `.sha256`. Refuses a missing/empty source, never overwrites, verifies the archive file count.
The AI registry has no script yet: copy the `ai_versions_data` volume directory the same way (manual, NOT automated).

## Restore (destructive, explicit)
`docker compose stop api` -> `python scripts/restore.py --archive <file> --target <chroma dir> --yes`.
Without `--yes` it refuses; checksum mismatch or unsafe archive members abort before anything is replaced; the existing target is
**moved aside** to `<target>.pre-restore-<timestamp>` (never deleted). Then `docker compose start api`, check `/ready`, run
`python scripts/smoke_test.py` (needs Ollama) or a retrieval on a known `document_id`.

## Drill (safe, test data only)
`python -m scripts.dr_drill` creates a disposable directory, ingests a generated PDF (fake embeddings), backs up, destroys the
*test* persistence, restores, queries again and compares chunk count, source, embedding model and top chunk. Each phase is a
separate process. It refuses the configured `CHROMA_PERSIST_DIRECTORY` and non-empty directories. Test:
`tests/release/test_dr_and_restart.py` (also: corrupted backup rejected, restore needs confirmation, previous data kept aside).

## RPO / RTO
| | Target (proposal) | Observed | Verified |
|---|---|---|---|
| RPO | <= 24 h (daily backup; data created after the last backup is lost) | not measurable: depends on the real schedule | NOT VERIFIED (no scheduler configured by this repository) |
| RTO | <= 60 min for restore + restart + readiness (excludes re-pulling Ollama models) | drill on test data: see PHASE21_REPORT.md (seconds, tiny collection, local disk) | NOT VERIFIED for production-size data, Docker volumes and model reload |
Re-pulling models needs network and disk (`qwen2.5:7b` is several GB): keep them in the `ollama_data` volume and include that
in the RTO planning. Observed drill timings are **not** an RTO claim.

## Limits
Restore of the AI registry, Docker volume restore, restore on another host, and encrypted backups are **NOT VERIFIED**.
