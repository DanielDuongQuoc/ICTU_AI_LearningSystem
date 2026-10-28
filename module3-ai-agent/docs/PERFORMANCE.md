# Performance (Phase 22)

> **Scope of the numbers.** Everything marked MEASURED was produced by `python -m scripts.benchmark_scale` on one
> Windows developer machine (specs not recorded), with the real PyMuPDF, the real chunker and a real ChromaDB in a temp
> directory. Embeddings were a deterministic hash embedder (`fake-hash-64`). **Ollama is not installed here**, so
> embedding-model latency, LLM latency, cold start, GPU behaviour and end-to-end request latency are NOT VERIFIED.
> Raw results: `docs/benchmarks/`.

## Pipeline and where time goes

```
PDF -> extraction -> chunking -> embedding -> Chroma insert -> retrieval -> context -> LLM -> parsing -> evaluation
```

Runtime per-stage timings already exist (Phase 20 `module3_stage_duration_seconds`, stages `pdf_extraction`,
`chunking`, `embedding`, `index_write`, `chroma_query`, `llm_generate`, ...). `scripts/benchmark.py` (Phase 15) measures
the same stages against a live Ollama; `scripts/benchmark_scale.py` (Phase 22) measures the Ollama-independent stages
and growth without needing a model.

### Measured stage costs (100-page synthetic PDF, 448 chunks, after the chunking fix)

| Stage | Time | Per unit | Status |
|---|---:|---:|---|
| PDF extraction | 103 ms | ~1.0 ms/page | MEASURED (varied 0.98-1.4 ms/page between runs) |
| Chunking | 325 ms | ~3.2 ms/page | MEASURED |
| Embedding | n/a | n/a | NOT VERIFIED (needs Ollama). Hash embedder: 59 ms for 448 chunks, not representative |
| Chroma insert | 438 ms | ~1.0 ms/chunk | MEASURED |
| Retrieval (Chroma query, top_k=5) | p50 7.7 ms, p95 13.8 ms (50 queries) | | MEASURED, excludes query embedding |
| Delete document | 46 ms (23 chunks) | | MEASURED |
| LLM generation / evaluation / SSE | n/a | | NOT VERIFIED |

Method limits: single run per size, 1 sample for per-document stage timings (the regression gate treats such rows with
`--min-samples`), synthetic text without sentence punctuation (this is a worst case for the chunker: every page is split
into word units). Real documents will chunk faster per character; this was not measured.

## Bottleneck found and fixed

| | |
|---|---|
| Optimization | Chunk selection no longer rebuilds the candidate string for every unit |
| Before | `_join_units(selected + [unit])` per unit: O(k^2) per chunk. 100 pages: **1500 ms** |
| After | length is tracked incrementally (same separator rules). 100 pages: **325 ms** (-78%) |
| Why | chunking was the largest Ollama-independent stage, 11x extraction and 3.4x the Chroma insert |
| Tradeoff | none observed; logic is identical, not an approximation |
| Test | `tests/performance/test_phase22.py::test_optimised_chunking_matches_previous_algorithm` runs 32 seeded random documents x 4 size/overlap settings against a copy of the previous algorithm and requires identical text and page metadata; plus a linear-scaling guard. All existing chunking tests pass |

Before/after files: `docs/benchmarks/scale_before_chunking_fix.json`, `scale_after_chunking_fix.json`. Comparing them
with the gate: chunking -78.3%, extraction -24.6% (noise: extraction was not changed), insert -0.1%, retrieval +2.9%,
verdict PASS. The 100-page numbers come from one run each, so small deltas are noise.

## Embedding

| | |
|---|---|
| Before | one `/api/embed` request containing every chunk of a document (up to 3000 chunks); query embeddings unbounded in parallel |
| After | sequential batches of `EMBEDDING_BATCH_SIZE` (default 64) and at most `MAX_CONCURRENT_EMBEDDINGS` (default 4) calls at once; excess callers get `503 SERVICE_OVERLOADED` after `EMBEDDING_QUEUE_TIMEOUT_S` |
| Why | bounds request size and memory on the Ollama side and prevents a burst of queries from piling up behind the model |
| Tradeoff | a very large document needs more round trips. The sequential overhead per extra request is unmeasured (NOT VERIFIED); the default of 64 is a conservative choice, not a tuned optimum |
| Test | order/count preserved across batches, single request for small inputs, observed in-flight peak == limit, overload rejection then recovery |

Duplicate embedding: chunks are embedded once at upload; queries are embedded once per request. No embedding cache was
added: no measured repeat rate justifies it and a cache risks cross-document or stale-model contamination.

## Concurrency under load (offline, fake LLM/embedding behind the real API)

`tests/performance/test_phase22.py`: bursts of 10, 20 and 60 simultaneous `/api/rag/retrieve` calls all succeed with
unique request ids; 6 simultaneous evaluations against a gate of 1 slot produce a mix of 200 and clean 503, no
tracebacks, and the next request succeeds. Existing Phase 16/19/20 tests cover upload limits, SSE disconnect, slow and
unavailable Ollama, and recovery. These are behaviour checks; they say nothing about real throughput.

## Regression gate

`python -m scripts.perf_regression BASELINE.json CURRENT.json [--threshold 20]` compares two `benchmark_scale` runs.
Threshold default is `PERFORMANCE_REGRESSION_THRESHOLD_PERCENT=20`. Verdicts and exit codes:

| Verdict | Exit | Meaning |
|---|---:|---|
| PASS | 0 | all comparable metrics within threshold |
| PERFORMANCE REGRESSION | 1 | a metric slowed more than the threshold (never hidden by missing data elsewhere) |
| INSUFFICIENT DATA | 2 | missing baseline/metric, too few samples, or **runs not comparable** (dataset, embedding, LLM, prompt, chunk or RAG config differ) |
| INFRASTRUCTURE FAILURE | 3 | file missing or unreadable |

Functional failures come from pytest and quality regressions from `scripts.quality_gate`; they are intentionally not
mixed into this verdict. Absolute values under `--noise-floor-ms` (1 ms) are ignored. Timing on a shared CI runner is
noisy; compare runs from the same machine. No committed machine-independent baseline exists: the files in
`docs/benchmarks/` are from the machine above.

## Quality vs performance

| Configuration | Latency | Hit@K | Precision@K | Answer quality |
|---|---|---|---|---|
| baseline (TOP_K 5, chunk 800/150) | retrieval p95 13.8 ms (Chroma only) | INSUFFICIENT DATA | INSUFFICIENT DATA | INSUFFICIENT DATA |
| optimized | identical retrieval settings; chunk output byte-identical | unchanged by construction | unchanged by construction | unchanged by construction |

No retrieval-parameter tuning (top_k, threshold, chunk size, overlap, context size) was performed. A fair comparison
needs Phase 17 quality metrics with a real embedding model, which cannot run here. The only optimizations made are
output-identical (chunking) or bounds that do not change results below the limits (embedding batching).

## Known limitations

- No end-to-end or Ollama measurements (NOT VERIFIED). No CPU/GPU/VRAM profiling, no cold/warm model comparison.
- PDF corpus is synthetic text only: no scanned, image-heavy or unusual-font PDFs.
- Memory figure is Python allocations for extract+chunk (tracemalloc), not process RSS, and excludes Chroma.
