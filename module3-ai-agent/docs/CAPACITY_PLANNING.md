# Capacity Planning (Phase 22)

Labels: **MEASURED** = produced by a script in this repository on the machine named in `docs/PERFORMANCE.md`;
**ESTIMATE** = arithmetic on measured values or on stated assumptions; **NOT VERIFIED** = no data.
There is no production traffic, no Ollama on the benchmark machine, and no GPU data. Nothing here is a production fact.

## Current measured capacity (Ollama-independent stages only)

| Item | Value | Label |
|---|---|---|
| Extraction + chunking + Chroma insert, 100 pages | ~0.87 s (103 + 325 + 438 ms), embeddings excluded | MEASURED |
| Same, per page | ~8.7 ms | MEASURED (100-page run) |
| Chroma query p95 at ~1000 chunks | 5-14 ms, flat while chunks grew 583 -> 1043 | MEASURED |
| Python memory, extract + chunk | ~60 KB/page (tracemalloc peak) | MEASURED |
| Chroma store, 64-dim vectors | ~7.2 KB/chunk (3 -> 23 documents: 5.2 -> 8.5 MB for +460 chunks) | MEASURED |
| Chunks per page (synthetic dense text) | ~4.5 | MEASURED (real PDFs vary) |
| Document delete | 46 ms; 0 orphan chunks; other documents untouched | MEASURED |
| Embedding latency per chunk / batch | unknown | NOT VERIFIED |
| LLM latency (generate, evaluate, SSE first token) | unknown | NOT VERIFIED |
| Requests per minute / concurrent users end to end | unknown | NOT VERIFIED |

## Model for planning (ESTIMATE)

```
ingest_time(doc)   = pages x 8.7 ms  +  chunks x t_embed          chunks ~ 4.5 x pages;  t_embed = NOT VERIFIED
storage(doc)       = chunks x S                                   S ~ 7.2 KB (64-dim, measured)
                                                                  S ~ 10 KB if the embedding model has 768 dims (estimate:
                                                                  +2.8 KB raw float32 per chunk; HNSW overhead not measured)
python_memory(doc) = pages x 60 KB (peak, extract + chunk, one document)
```

Worked example (ESTIMATE): 100-page document -> ~450 chunks -> ~3-4.5 MB in Chroma and ~6 MB Python peak, ~0.9 s CPU
before embedding. At the upload limit (`MAX_PDF_PAGES` 500) these scale linearly to ~2250 chunks (under
`MAX_CHUNKS_PER_DOCUMENT` 3000 for this text density) and ~4.4 s CPU before embedding.

Documents per day, pages per day and embeddings per day cannot be derived honestly without `t_embed`. To measure it:
run `python scripts/benchmark.py` on the target host with the pinned models; then
`docs/` figures above can be completed by plugging `t_embed` into the model.

## Concurrency recommendations (defaults, not tuned)

| Knob | Default | Rationale | Label |
|---|---|---|---|
| `LLM_MAX_CONCURRENCY` | 4 | prevents a pile-up behind Ollama | untuned; depends on hardware (NOT VERIFIED) |
| `MAX_CONCURRENT_EMBEDDINGS` | 4 | same for embeddings | untuned (NOT VERIFIED) |
| `EMBEDDING_BATCH_SIZE` | 64 | bounds one embed request | untuned (NOT VERIFIED) |
| `MAX_CONCURRENT_UPLOADS` | 2 | bounds CPU/memory of PDF work | untuned |
| `UVICORN` workers | 1 | the gates and rate limiter are per process | by design (see SCALABILITY.md) |

A local model on CPU usually serves one generation at a time efficiently; raising `LLM_MAX_CONCURRENCY` mainly adds
queueing latency. Treat that as a hypothesis to test, not a result.

## Resource requirements

| Resource | Requirement | Label |
|---|---|---|
| CPU | API + PDF + Chroma: modest (see above). Ollama inference dominates | CPU sizing for Ollama NOT VERIFIED |
| RAM | API process small (tens of MB per document in flight); Ollama RAM = model size (qwen2.5:7b is multi-GB) | model RAM NOT VERIFIED here |
| GPU | optional; compose has a commented GPU block | **GPU CAPACITY NOT VERIFIED** (no GPU assumed) |
| Disk | Chroma growth above; Ollama models (multi-GB) in `ollama_data`; backups (full copies) | see Disk growth |

## Disk growth

| Store | Growth | Control |
|---|---|---|
| ChromaDB | ~7-10 KB per chunk (ESTIMATE above) | delete document removes its chunks (MEASURED, no orphans); no compaction scheduled |
| Uploads | temp PDFs are removed after processing (`rmtree` in the upload handler); `uploads_data` volume should stay near empty | verify with `du` on the volume (NOT VERIFIED on a deployed host) |
| Logs | stdout only. **Docker json-file logs were unbounded; Phase 22 caps them at 5 x 10 MB per container in compose** | compose `logging` block; not exercised because the Docker daemon is unavailable (NOT VERIFIED), `docker compose config` validates |
| Backups | one full tar.gz per backup | `scripts/rotate_backups.py` exists (Phase 16); retention policy is a recommendation, not enforced |
| Benchmark artifacts | small JSON files in `docs/benchmarks/` | keep only baselines |
| AI versions | small JSON | negligible |

## Retention (recommendation, nothing is deleted automatically)

Documents and vectors: keep until the owner deletes them via the API; no automatic expiry exists. Logs: Docker cap
above. Backups: keep the last N full backups and one older monthly (N to be chosen by the operator). Benchmark results:
keep the latest baseline per configuration. No production data policy has been defined by the project; deciding one is
an open item.

## Known bottlenecks

1. Ollama inference (LLM) and embedding: not measured here, expected to dominate end-to-end latency.
2. Chunking (fixed this phase, still the largest Ollama-independent stage after Chroma insert).
3. Chroma insert at ~1 ms/chunk is the next largest Ollama-independent stage.
4. Single process: one Python event loop; CPU-bound PDF work runs in worker threads, bounded by `MAX_CONCURRENT_UPLOADS`.

## Scaling recommendations

See `docs/SCALABILITY.md`. In short: scale the Ollama host vertically first; keep one API process; do not add
infrastructure until measurements on the real model show a need.
