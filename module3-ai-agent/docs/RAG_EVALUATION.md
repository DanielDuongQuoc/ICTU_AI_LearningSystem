# RAG evaluation

Script: `scripts/evaluate_rag.py` (Phase 10 script extended; the legacy `tests/data/rag_eval.json`
flow still works with `--dataset tests/data/rag_eval.json`). Metrics code:
[`app/utils/rag_metrics.py`](../app/utils/rag_metrics.py); suite: `scripts/rag_quality.py`.
Datasets and provenance: [`evaluation/README.md`](../evaluation/README.md). Overview:
[AI_QUALITY_EVALUATION.md](AI_QUALITY_EVALUATION.md).

## Ground truth

`rag_eval.jsonl` (v1.0.0, 39 cases, **synthetic, author-labelled, not human-reviewed**): each case names
one fixture document, a query and the `relevant_chunk_ids` that answer it (fixed ids `fixture_chunk_NNN`).
Categories: answerable 24, paraphrased 4, partial 3, false_premise 2, out_of_scope 4,
outside_knowledge_request 1, fabricate_source_request 1 (the last three groups, 6 cases, expect
abstention). Fixtures are indexed into a temporary ChromaDB under fresh UUIDs, using the application's
`VectorService`, `ChromaRepository` and `RAGService.retrieve` (document filter and similarity threshold included).

## Metrics (K = `--k`, default `TOP_K`; recorded in each report)

Computed over the 33 answerable cases:

* **Hit@K** - at least one relevant chunk in the top K.
* **Precision@K** - relevant in top K / K (denominator always K; with 1-3 relevant chunks the maximum
  is below 1, so compare Precision only against itself).
* **Recall@K** - unique relevant found / total relevant.
* **MRR** - mean of 1/rank of the first relevant chunk.
* **nDCG@K** - binary gain.
* **Duplicates:** a chunk id returned more than once counts once (tested).
* **Wrong-document results are never hits**; they are counted as `document_isolation_violations`.
* Retrieval latency (mean/max), empty-retrieval rate on answerable cases.

Abstention (6 abstain cases): `insufficient_context_detection_rate`, `unsupported_context_pass_rate`
(abstention cases that nonetheless reached generation), and `incorrect_abstention_rate` on answerable
cases. Reporting both directions prevents "improving" by refusing everything.

With `--mode ollama --answers` also: completeness (key-point token coverage >= 0.6), faithfulness /
answer-relevance **proxies** (token overlap with the retrieved context / query), final
`correct_abstention_rate`, `unsupported_answer_rate`, `incorrect_final_abstention_rate`, and
`answer_source_violations` (answer sources not in the retrieved set). These are proxies, not a verdict.

## Results obtained (offline, lexical embedder - NOT production quality)

K=5, dataset 1.0.0, mode `offline` (`lexical-test-embedding`, no LLM), no git commit available.
Reports: `evaluation/reports/`, `evaluation/reports/sweeps/`.

| metric | threshold 0.3 (production value) | threshold 0.1 (sweep) |
|---|---|---|
| Hit@5 | 0.5152 | 0.9697 |
| Precision@5 | 0.1030 | 0.2000 |
| Recall@5 | 0.5152 | 0.9697 |
| MRR | 0.4848 | 0.8939 |
| nDCG@5 | 0.4928 | 0.9138 |
| empty retrieval (answerable) | 0.4848 | 0.0000 |
| insufficient-context detection (6 cases) | 0.6667 | 0.0000 |
| incorrect abstention (answerable) | 0.4848 | 0.0000 |
| document-isolation / source violations | 0 / 0 | 0 / 0 |
| mean latency | ~3.9 ms | ~2.6 ms |

How to read it: the two columns show the **recall-vs-abstention trade-off** of `SIMILARITY_THRESHOLD`
(lenient = finds the right chunk but never abstains; strict = abstains but misses answers). With this
embedder cosine scores are lower than a real embedding model's, so 0.3 rejects half the answerable
queries; that says nothing about `nomic-embed-text`. The thresholds were **not** chosen to look good and
`SIMILARITY_THRESHOLD` was not changed. The only reliable offline results are the invariants:
**zero isolation and zero source-integrity violations** across all 39 cases at both thresholds.

Model-dependent results (real embeddings, `--answers`): **NOT VERIFIED** (Ollama unavailable).

## Interpreting results / limits

* Compare runs only if dataset version, K, threshold, embedding model and mode match (the gate checks).
* 33 answerable cases means one case moves Hit@K by 3 points; differences under ~10 points are noise.
* Chunk granularity: ground truth is one passage per chunk; production chunking (800/150) differs.
* Token-overlap proxies miss paraphrase and negation; use human review for faithfulness claims.
