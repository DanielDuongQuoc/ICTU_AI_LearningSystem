# Maintenance Procedure (Phase 16)

Applies to routine changes. None of the steps below has been exercised on a Docker host or with a
live Ollama from this repository's development machine (**NOT VERIFIED** there); the offline parts
are covered by the test suite.

## 0. Always
1. Stop writers and take a backup first: `docker compose stop api`, `python scripts/backup.py`, start again.
2. Make the change on a copy / staging environment when possible.
3. Run the regression suite and `python scripts/release_check.py` before and after.
4. Keep the previous image tag and previous `.env` for rollback.

## 1. Routine backup and retention
```text
python scripts/backup.py                         # checksummed .tar.gz, never overwrites
python scripts/rotate_backups.py --keep 7        # dry run: shows what would be deleted
python scripts/rotate_backups.py --keep 7 --apply
```
Retention keeps the newest N *valid* archives; checksum-mismatched archives are reported and left in
place. Do a restore drill into an isolated directory periodically (`scripts/restore.py --target <new dir> --yes`),
then check document count and one retrieval. Never restore over production without the pre-restore copy.

## 2. Updating dependencies
1. Change one package (or one tightly coupled group) in `requirements.txt` with an exact pin. No blanket upgrades; no major-version jump without reading its changelog.
2. Fresh virtual environment: `pip install -r requirements.txt`, then `pytest -q`, `ruff check --select E9,F app scripts tests`, `scripts/release_check.py`.
3. For `chromadb`: also verify an existing persisted directory still opens (copy a backup, open it, count documents) because on-disk format can change between versions.
4. Vulnerability scanning: run a scanner such as `pip-audit` where available. It has **not** been run in this repository (not installed, and it needs network access); treat dependency vulnerability status as **unknown** until it is.
5. Rebuild the image, redeploy, check `/ready`, run `scripts/smoke_test.py`.

## 3. Updating the Ollama model (generation)
1. `ollama pull <new-model>` on the Ollama host, pin `OLLAMA_IMAGE` to a tested tag.
2. Set `OLLAMA_MODEL`, restart the API.
3. Re-run: `scripts/benchmark.py`, a handful of evaluations/question generations (score range, JSON validity, Vietnamese/Chinese/English text), and a stream. Compare latency with the previous run; change `OLLAMA_TIMEOUT` only if measured.
4. Roll back by restoring the previous `OLLAMA_MODEL` value.

## 4. Embedding model: do **not** change casually
Vectors are model-specific. Mixing vectors from two embedding models in one collection gives
meaningless similarity scores. Phase 16 does not change the embedding model. If it must change in
future:
1. Back up the Chroma directory.
2. Create a **new** collection name (`CHROMA_COLLECTION_NAME`) or a new directory; do not overwrite the old one.
3. Re-ingest every document with the new model (new `document_id`s unless a migration plan preserves them, which is a separate design task).
4. Run `scripts/evaluate_rag.py` and the isolation tests against the new collection; compare with the baseline.
5. Switch traffic only after validation; keep the old collection until rollback is no longer needed.

## 5. Data migrations
There are no schema migrations in this module (ChromaDB metadata is written at ingestion time).
If a future change alters stored metadata, it needs its own migration/re-ingestion plan and tests.

## 6. Deploying a new version
1. Build (`docker compose build`), keep the old image tag.
2. `docker compose up -d`, then `/health/live`, `/ready`, `scripts/smoke_test.py`.
3. `docker compose restart` and verify a known document still retrieves (persistence).
4. Watch `/metrics` / `scripts/ops_check.py` for 5xx, p95 and `module3_llm_rejected_total`.

## 7. Rollback
Redeploy the previous image tag and `.env`. If the data directory was written by a different
`chromadb` version, restore the pre-change backup into a fresh directory instead of reusing it.

## 8. Tuning knobs (change one at a time, with measurements)
`LLM_MAX_CONCURRENCY`, `LLM_QUEUE_TIMEOUT_S`, `OLLAMA_TIMEOUT`, `SLOW_REQUEST_THRESHOLD_MS`,
`LOG_LEVEL`, `TOP_K`, `SIMILARITY_THRESHOLD`. Do not raise timeouts to hide an unresolved cause.
