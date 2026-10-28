# Maintenance Plan (Phase 22)

All frequencies are **recommendations**, not an existing production policy. Procedures for backup, dependency and model
updates already exist in `docs/MAINTENANCE.md` (Phase 16), `docs/DEPLOYMENT_RUNBOOK.md` and `docs/DISASTER_RECOVERY.md`;
this plan only schedules them and adds the performance items.

## Weekly
- `/health`, `/ready`: all services ok including `embedding_index` and `storage`.
- `/metrics`: error rate, `module3_llm_rejected_total`, embedding `rejected` events, p95 per stage (`scripts.check_reliability`).
- Disk: Chroma volume, uploads volume (should be ~empty), Docker log volume (capped at 5 x 10 MB per container), backup directory.
- Logs: scan for `SERVICE_OVERLOADED` bursts (limits too low or real overload).

## Monthly
- Dependency audit: `pip check`, `pip list --outdated`. Pinned versions are *not* auto-upgraded (see below).
- Performance: `python -m scripts.benchmark_scale --output <new file>` then
  `python -m scripts.perf_regression docs/benchmarks/<baseline>.json <new file>`; run `scripts/benchmark.py` against the
  real Ollama on the target host when available (baseline for that host does not exist yet: NOT VERIFIED).
- Vector store growth: chunk count and storage size vs the capacity model (`docs/CAPACITY_PLANNING.md`).
- Backups: restore one into a scratch directory (`scripts/dr_drill.py` for the procedure on test data).
- Technical debt list below.

## Per release
Run `python -m scripts.validate_release --manifest <path>` (functional, security, reliability, release tests, static
checks) plus: Phase 17 quality gate with a real model (NOT VERIFIED without Ollama), performance regression gate,
OpenAPI freeze test, compatibility check of AI version / embedding model (Phase 18/21 guards).

## Dependency audit (this phase; `pip check`: no broken requirements)

| Package | Pinned | Newest seen | Assessment |
|---|---|---|---|
| fastapi | 0.115.5 | 0.162.0 | large gap; upgrade needs full regression and OpenAPI diff |
| uvicorn | 0.32.1 | 0.54.0 | large gap; same |
| pydantic / pydantic-settings | 2.10.1 / 2.6.1 | 2.14.1 / 2.16.1 | minor-version upgrades; validate config tests |
| PyMuPDF | 1.24.13 | 2.0 | **major**: do not upgrade without PDF extraction/page-metadata tests |
| chromadb | 0.5.23 | 1.5.9 | **major**: stored data format and API may change; needs migration test with backup/restore |
| httpx | 0.28.0 | 0.28.1 | patch; low risk |
| pytest / pytest-asyncio | 8.3.4 / 0.24.0 | 9.1.1 / 1.4.0 | major test-tool jumps; separate change |

No dependency was upgraded in Phase 22: the version gaps are large and each needs its own regression run. Python is
pinned to 3.11.9 in the Dockerfile. Ollama image is `latest` unless `OLLAMA_IMAGE` is set (documented risk). The newest
versions were read from `pip list --outdated` on 2026-10-24 and were not verified for compatibility or security
advisories (no vulnerability scanner was run: NOT VERIFIED).

## Technical-debt audit (found, not rewritten)

| Item | Impact | Status |
|---|---|---|
| Chunk selection was quadratic | performance | **fixed** |
| Unbounded embedding batch/concurrency | reliability | **fixed** |
| Unbounded Docker logs | disk | **fixed** (compose cap) |
| No `TODO`/`FIXME`/`print()` in `app/` | | clean (grep) |
| `chunk_service._overlap_units` uses `insert(0)` | negligible (small lists) | left; no evidence of cost |
| Rate limiter, gates are per-process | scaling | documented, by design |
| Three late `"999" not in r.text` assertions in `tests/security/test_ai_integrity.py` can match a random uuid | rare test flake | left; low impact |
| `OLLAMA_IMAGE` default `latest` | reproducibility | documented, not forced |
| Real-model quality/latency benchmarks missing | verification gap | needs Ollama host |
