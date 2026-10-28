# Backup and recovery

## What is persistent
| Data | Location | Back up? |
|---|---|---|
| ChromaDB (chunks, embeddings, document metadata) | `data/chroma` / volume `chroma_data` | **Yes** |
| Ollama models | volume `ollama_data` | No - re-pull (`ollama pull ...`); large |
| Uploaded PDFs | `uploads/` / volume `uploads_data` | No - temporary files; the source PDFs belong to Module 1 / the uploader |
There is no SQL database. Losing Chroma means re-uploading the PDFs (document ids change, because Module 3
generates them - consumers must re-map ids).

## Consistency warning
Chroma stores SQLite + index files. Copying them while the API writes can produce a broken snapshot. **Stop the
API before backing up** (`docker compose stop api`); `scripts/backup.py` cannot detect a running writer and only
warns. A "hot" copy is not guaranteed to be restorable.

## Backup
```bash
docker compose stop api
# Docker volume -> host directory (read-only mount):
docker run --rm -v module3-ai-agent_chroma_data:/src:ro -v "$PWD/chroma_export":/dst alpine cp -a /src/. /dst/
python scripts/backup.py --source chroma_export --dest-dir backups
# local run: python scripts/backup.py            (defaults: CHROMA_PERSIST_DIRECTORY -> ./backups)
docker compose start api
```
(The volume name is `<compose project>_chroma_data`; check `docker volume ls`.) The script writes
`backups/chroma-backup-YYYYmmdd-HHMMSS.tar.gz` + `.sha256`, verifies the archive, never overwrites an
existing file and never deletes anything. Keep copies off the host.

## Restore
```bash
docker compose stop api
python scripts/restore.py --archive backups/chroma-backup-<stamp>.tar.gz --target chroma_restore --yes
docker run --rm -v "$PWD/chroma_restore":/src:ro -v module3-ai-agent_chroma_data:/dst alpine \
    sh -c "rm -rf /dst/* && cp -a /src/. /dst/"
docker compose start api && curl http://localhost:8003/health/ready
```
`restore.py` refuses to run without `--yes`, verifies the checksum when the `.sha256` file exists, rejects
archives with path-traversal or link members, and **moves** the existing target to
`<target>.pre-restore-<timestamp>` instead of deleting it. For a local (non-Docker) run restore directly
into `data/chroma`.

After a restore run `python scripts/smoke_test.py` (or at least `GET /api/documents/{id}` for a known id).

## Verification status
The archive/restore logic is covered by unit tests (`tests/test_phase14.py`). The Docker-volume commands above
were **not executed** in the Phase 14 environment (no Docker daemon); try them on a non-production stack first.

## Backup retention (Phase 16)

`python scripts/rotate_backups.py --keep 7` is a dry run; add `--apply` to delete. It only touches
`chroma-backup-*.tar.gz` (+ `.sha256`), always keeps the newest N *valid* archives, and never
deletes an archive whose checksum is missing or wrong (reported as CORRUPT). See
[MAINTENANCE.md](MAINTENANCE.md) for the restore drill. Backup and restore were exercised against
real ChromaDB persistence in an isolated temporary directory by `tests/test_operations_scripts.py`
(backup, delete data, restore, count and retrieve). That is **not** a Docker-volume drill (NOT VERIFIED).