# Scalability (Phase 22)

Current deployment: **one API process + one Ollama + one embedded ChromaDB on local disk** (docker-compose, single host).
Nothing below has been load-tested against a real Ollama (NOT VERIFIED). No distributed component is proposed because
no measured need exists.

## Current architecture limits

| Limit | Why | Evidence |
|---|---|---|
| One API process | in-process LLM/embedding gates, upload gate and rate limiter are per process; a second process doubles every limit and splits the rate-limit buckets | code (`app/core/concurrency.py`, `rate_limit.py`) |
| Embedded Chroma on one disk | `PersistentClient` is not safe to share between several writer processes/hosts | ChromaDB design; not tested here |
| Ollama throughput | a local model serves few simultaneous generations well | NOT VERIFIED here |
| No queue/worker system | overload is shed (503) after a bounded wait instead of queued durably | by design (Phase 16) |

## Safe concurrency

Bounded everywhere (see `docs/RESOURCE_LIMITS.md`): LLM calls, embedding calls, uploads, request size, stream length,
per-client rate. Excess work is rejected with a clear status; offline tests confirm no crash, no traceback leak and
recovery after overload. The numeric values are defaults, not tuned.

## Bottleneck components (in expected order)

1. Ollama generation. 2. Ollama embedding. 3. Chroma insert (~1 ms/chunk MEASURED). 4. Chunking (3.2 ms/page MEASURED).
5. PDF extraction (~1 ms/page MEASURED).

## Growth

- **Documents/vectors:** Chroma query p95 stayed flat (5-14 ms) while chunks grew 583 -> 1043 (MEASURED, small). Behaviour
  at 100k+ chunks is NOT VERIFIED. Document isolation holds (a `where document_id` filter; 0 cross-document hits in
  every benchmark query and in the isolation tests).
- **Lifecycle:** upload -> process -> embed -> store -> retrieve -> delete verified offline; delete leaves 0 chunks for that
  document and does not touch others. Partial indexes from failed uploads are removed by the upload handler (existing
  logic, Phase 19). No separate orphan-cleanup job was added: no orphan source was found.
- **Embedding model changes** require a re-index (Phase 21 guard refuses to mix vectors).

## Vertical scaling (first choice)

More RAM/CPU for Ollama (and a GPU if latency requires it: NOT VERIFIED), faster disk for Chroma. Re-measure with
`scripts/benchmark.py` before and after, and gate with `scripts/perf_regression.py`.

## Future options (only if measurements justify; none is implemented)

- Separate Ollama host(s): change `OLLAMA_BASE_URL`; no code change needed. Several Ollama hosts would need a client-side
  balancer, which does not exist.
- Multiple API processes: requires moving the gates/rate limiter to shared state and Chroma to a client/server
  deployment. That is a design change, not a configuration change.
- Background ingestion queue for very large uploads: only if upload latency proves unacceptable on real documents.

## Explicitly not claimed

No horizontal scalability, no zero-downtime scaling, no throughput figure, no maximum number of users.
