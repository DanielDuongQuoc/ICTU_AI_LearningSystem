# Phase 22 Report: Capacity Planning, Performance, Scalability and Maintainability

## 1. Executive Summary

Phase 22 found and fixed one real performance bottleneck (quadratic chunking), bounded embedding work (batch size and
concurrency), capped unbounded Docker logs, added an offline scalability benchmark and a performance-regression gate that
keeps failure classes apart, and documented limits, capacity, scaling and maintenance. **Everything that needs Ollama
(embedding latency, LLM latency, end-to-end throughput, GPU, cold/warm start, real RAG quality) is NOT VERIFIED:
Ollama is not installed on this machine.** No capacity or throughput claim for production is made.

## 2. Repository Audit (Phase 1-21)

Reused, not duplicated: LLM concurrency gate and queue timeout (Phase 16), upload/page/chunk/char limits and rate limit
(Phase 19), per-stage metrics and SLO harness (Phase 20), `scripts/benchmark.py` (Phase 15, needs Ollama), backup/restore
and DR drill (Phase 21), AI version governance (Phase 18). SLOs in `docs/SLO_SLA.md` are still proposals with no
production data. Gaps found: embeddings were unbounded (one request for all chunks, unlimited parallel queries); chunk
selection was O(k^2); Docker logs had no rotation; no offline growth benchmark; no regression gate.

## 3. Baseline

`scripts/benchmark_scale.py`, synthetic PDFs of 5, 25, 100 pages, real PyMuPDF/chunker/ChromaDB, hash embeddings.
Raw data: `docs/benchmarks/scale_before_chunking_fix.json`. Hardware specs not recorded; git commit unavailable (not a
git repository).

## 4. Bottlenecks

Largest Ollama-independent stage was chunking: 1500 ms for 100 pages (11x extraction, 3.4x Chroma insert). LLM and
embedding-model bottlenecks are expected but NOT VERIFIED.

## 5. Performance Optimization

| Optimization | Before | After | Why | Tradeoff | Test |
|---|---|---|---|---|---|
| Incremental length in chunk selection | 1500 ms / 100 pages | 325 ms (-78%) | removed per-unit string rebuild | none; output identical | differential test vs previous algorithm, 32 seeded cases + scaling guard |
| Embedding batches | one request with all chunks | batches of `EMBEDDING_BATCH_SIZE` (64) | bound request size/memory | more round trips for big documents (cost NOT VERIFIED) | batch order/count tests |

## 6. Resource Optimization

Docker json-file logs capped at 5 x 10 MB per container (compose; `docker compose config` valid, runtime NOT VERIFIED).
Limits consolidated in `docs/RESOURCE_LIMITS.md`, with a test that fails if documented defaults differ from code.

## 7. Concurrency

New `MAX_CONCURRENT_EMBEDDINGS` (default 4), `EMBEDDING_QUEUE_TIMEOUT_S` (30), `EMBEDDING_BATCH_SIZE` (64), all
validated at startup and in `.env.example`. Overload gives `503 SERVICE_OVERLOADED` after a bounded wait. Tests: peak
in-flight equals the limit, rejection then recovery, unlimited mode, bursts of 10/20/60 retrievals, LLM overload burst
(mix of 200/503, no traceback, recovers). These use fakes behind the real API.

## 8. Capacity Planning

See `docs/CAPACITY_PLANNING.md`. MEASURED: ~8.7 ms/page for extract+chunk+insert (embeddings excluded), ~60 KB/page
Python peak, ~7.2 KB/chunk Chroma storage at 64 dims. ESTIMATE: ~10 KB/chunk at 768 dims. Documents/day, requests/minute,
concurrent users: NOT VERIFIED (needs embedding and LLM timings).

## 9. ChromaDB Scalability

Chunks 583 -> 1043 (3 -> 23 documents): retrieval p95 stayed 5-7 ms for the base document, flat. 0 wrong-document hits in
every query (250 hits per size), delete removed exactly the document's chunks with 0 orphans and left others intact.
Behaviour at large scale NOT VERIFIED.

## 10. Ollama Performance

NOT VERIFIED (not installed): cold start, warm inference, concurrency, `num_predict`/context effects, CPU/GPU use.
**GPU CAPACITY NOT VERIFIED.** No model was changed or recommended.

## 11. AI Quality vs Performance

INSUFFICIENT DATA: no Hit@K, Precision@K or answer-quality comparison was possible without a real embedding model. The
only changes are output-identical (chunking) or bounds that do not alter results below the limits (batching). No RAG
parameter was tuned.

## 12. Technical Debt

See `docs/MAINTENANCE_PLAN.md`. No TODO/FIXME/print in `app/`. Open: three rare-flake `"999" in r.text` assertions in
security tests; per-process gates; `OLLAMA_IMAGE` default `latest`.

## 13. Dependency Maintenance

`pip check` clean. Large version gaps (chromadb 0.5.23 vs 1.5.9, PyMuPDF 1.24.13 vs 2.0, fastapi, uvicorn, pytest);
nothing upgraded because each needs its own regression and, for chromadb, a data-migration test. No vulnerability scan
was run (NOT VERIFIED).

## 14. Tests

`python -m scripts.validate_release` (this phase, final run): static checks 10 PASS; functional 772 passed, 9 deselected
(710 before + 62 new: `tests/performance/`); security 168 passed; reliability 78 passed; release tests 53 passed;
AI quality PARTIAL; Docker PARTIAL. Total passed = 1071. `scripts.release_check`: all 10 checks PASS. Verdict:
GO WITH LIMITATIONS (unchanged from Phase 21 reasoning, same NOT VERIFIED items). Not run: Phase-by-phase live checks of
upload/RAG/question generation/evaluation/SSE against real Ollama (offline fakes only), Docker build/runtime.

## 15. Benchmark Results

| Metric (100-page doc) | Before | After | Change |
|---|---:|---:|---:|
| Extraction | 136.7 ms | 103.2 ms | -24.6% (unchanged code; noise) |
| Chunking | 1499.6 ms | 325.0 ms | -78.3% |
| Chroma insert | 438.8 ms | 438.4 ms | -0.1% |
| Retrieval p95 (50 queries) | 13.4 ms | 13.8 ms | +2.9% |

`scripts.perf_regression` verdict: PASS at a 20% threshold. Single runs; embeddings are `fake-hash-64`. Run metadata
(run id, timestamp, dataset version, configured models, chunk/RAG config, Python/platform) is stored in each file.

## 16. Limitations

Synthetic text only; single machine, specs unrecorded; one run per size; per-document stage rows have 1 sample; memory is
tracemalloc not RSS; chunking benchmark text lacks sentence punctuation (worst case).

## 17. NOT VERIFIED Items

Embedding and LLM latency; end-to-end p50/p95/p99, throughput, error and timeout rate under real load; CPU/RAM/GPU/VRAM
of Ollama; cold vs warm start; real Ollama restart under load; real PDF corpus variety; Chroma at 100k+ chunks; effect of
`EMBEDDING_BATCH_SIZE`/gate values on real latency; RAG quality tradeoffs; Docker build/runtime and log-cap behaviour;
deployed-host disk usage; vulnerability scan; git commit/clean tree; CI on GitHub; any production traffic or SLO.

## 18. Recommendations

1. On a host with Ollama, run `scripts/benchmark.py` and record embedding/LLM latency; then fill the capacity model.
2. Tune `LLM_MAX_CONCURRENCY`, `MAX_CONCURRENT_EMBEDDINGS`, `EMBEDDING_BATCH_SIZE` from those numbers, one at a time.
3. Keep one API process; scale Ollama vertically first.
4. Pin `OLLAMA_IMAGE`. Plan chromadb/PyMuPDF upgrades as separate, backup-protected changes.
5. Commit a baseline `benchmark_scale` result per environment and run the regression gate monthly.

## 19. Final Acceptance Status

```
PHASE 22 STATUS
Repository audit:      PASS
Performance:           PARTIAL  (offline stages measured and one bottleneck fixed; Ollama stages NOT VERIFIED)
Scalability:           PARTIAL  (bounds + small-scale Chroma growth; large scale and real load NOT VERIFIED)
Capacity planning:     PARTIAL  (model written; key inputs NOT VERIFIED, estimates labelled)
Resource optimization: PASS     (embedding bounds, log cap, limits doc with drift test; Docker runtime NOT VERIFIED)
AI performance:        NOT VERIFIED (no Ollama)
Regression:            PASS     (1071 passed; AI quality and Docker gates PARTIAL)
Documentation:         PASS
```

Files changed: `app/core/config.py`, `app/core/concurrency.py`, `app/core/metrics.py`,
`app/services/embedding_service.py`, `app/services/chunk_service.py`, `.env.example`, `docker-compose.yml`.
Files added: `scripts/benchmark_scale.py`, `scripts/perf_regression.py`, `tests/performance/` (2 test files),
`docs/PERFORMANCE.md`, `CAPACITY_PLANNING.md`, `SCALABILITY.md`, `RESOURCE_LIMITS.md`, `MAINTENANCE_PLAN.md`,
`PHASE22_REPORT.md`, `docs/benchmarks/*.json`, `docs/release-manifest-phase22.json`.
Not 100% complete: see section 17.
