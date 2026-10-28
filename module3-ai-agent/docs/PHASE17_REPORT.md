# Phase 17 Report

AI quality assurance, RAG evaluation and continuous quality improvement - Module 3.
Run on 2026-10-24, Python 3.11.9, Windows. Ollama and Docker were **not** available.

## 1. Initial Audit

- **Existing evaluation infrastructure:** `scripts/evaluate_rag.py` (Phase 10; Hit/Precision/Recall@K, needs
  real Ollama embeddings), `app/utils/rag_metrics.py`, `tests/test_rag_evaluation.py`,
  `scripts/benchmark.py`, `scripts/release_check.py`, offline fakes in `tests/fakes.py`
  (`LexicalEmbedding`, `RecordingLLM`). Prompt versions `v1` for RAG, question generation and evaluation.
  Source/isolation behaviour already unit-tested in Phases 5-7; kept untouched.
- **Existing datasets:** `tests/data/rag_eval.json` (Phase 10, phrase-based ground truth). Reused (still
  runs via `--dataset tests/data/rag_eval.json`, covered by existing tests); extended, not replaced.
- **Existing baselines:** none for real Ollama (Phase 10/15: NOT RUN). No baseline could be recorded
  before changes because the model stack is unavailable. Pre-phase regression state: 617 passed, 9
  integration deselected, ruff clean, `release_check.py` all PASS.

## 2. Dataset

- **Dataset versions:** `1.0.0` (all four files).
- **Number of cases:** RAG 39 (33 answerable, 6 abstention), question generation 12, answer evaluation 31
  (4 error-contract cases; 27 intended to be graded, 26 actually graded because `eval_024` was rejected
  by retrieval, see 5) over a 3-document / 24-passage corpus.
- **Data source:** 100 % synthetic, written by the project author; no personal data, no student work.
  Fixture ids (`fixture_doc_001`, `fixture_chunk_001`...) are mapped to fresh UUIDs in a temporary
  ChromaDB; production data is never read or written.
- **Ground-truth method:** author-assigned relevant chunks and score bands, written before any run
  (`label_source: synthetic_author`). **HUMAN REVIEW NOT PERFORMED.** One pre-evaluation correction:
  `eval_006` was removed from rank-consistency group `g_sup` (ambiguous ordering) - see `evaluation/README.md`.

## 3. RAG Evaluation

Offline mode (lexical test embedder, no LLM), K=5. **Not production-embedding quality.**

| metric | thr 0.3 (prod value) | thr 0.1 (sweep) |
|---|---|---|
| Hit@5 | 0.5152 | 0.9697 |
| Precision@5 | 0.1030 | 0.2000 |
| Recall@5 | 0.5152 | 0.9697 |
| MRR | 0.4848 | 0.8939 |
| nDCG@5 | 0.4928 | 0.9138 |
| insufficient-context detection | 0.6667 | 0.0 |
| incorrect abstention | 0.4848 | 0.0 |
| isolation / source violations | 0 / 0 | 0 / 0 |

- **Faithfulness:** NOT VERIFIED (needs `--mode ollama --answers`; implemented only as a lexical proxy).
- **Unsupported answer rate:** NOT VERIFIED at answer level. Retrieval-level `unsupported_context_pass_rate`
  (abstention cases that reached generation): 0.3333 at 0.3, 1.0 at 0.1 (lexical embedder).

## 4. Question Generation

Offline harness self-check (reference questions replayed as the "LLM"; validates checker + service validation only):

- **Answerability:** 0.8333 (proxy; false negatives on author-written references).
- **Groundedness:** 0.8333 (proxy).
- **Duplication:** 0.0.
- **Expected-answer correctness:** proxy only (same as groundedness); MCQ structure 1.0 (3 MCQ), source integrity 1.0.
- Model-dependent results: NOT VERIFIED.

## 5. Answer Evaluation

Offline harness self-check (the "LLM" returns the band midpoint; **perfect by construction**):

- **MAE:** 0.0 (normalised) - not grading quality. **RMSE:** 0.0 - not grading quality.
- **Correct-label accuracy:** 1.0 (25 labelled cases) - by construction.
- **Score consistency:** 0 rank violations; scores in `[0, max_score]` 1.0; percentage backend-computed 1.0.
- **Real service findings:** empty answer, over-long answer and cross-document probe return the right
  error codes without calling the LLM (3/4). `eval_030` was graded instead of rejected and `eval_024`
  was rejected - lexical-embedder artefacts, to be re-checked with the real embedder.
- **Contract finding:** `correct` is not tied to any score threshold (documented, not changed).
- Model-dependent grading quality: NOT VERIFIED.

## 6. Security

- **Prompt injection:** deterministic behaviour tests pass (student answer never used for retrieval and
  confined to the user-prompt data block; document injection never in the system prompt, creates no
  sources; an "obedient" model returning an impossible score is rejected). Real-model resistance: NOT VERIFIED.
- **Source integrity:** PASS (deterministic) - fake sources/page/chunk_id/document_id/score/percentage in LLM
  output are ignored for RAG answers, evaluations and generated questions; 0 violations across all datasets.
- **Document isolation:** PASS (deterministic) - 0 foreign chunks for 39 RAG cases at thresholds 0.0 and 0.3,
  and for the targeted cross-document query test.

## 7. Reproducibility

- **Git commit:** `unavailable` (the working copy is not a git repository).
- **Model:** offline runs `scripted-test-llm`; configured production model `qwen2.5:7b` - never run.
- **Embedding model:** offline `lexical-test-embedding`; configured `nomic-embed-text` - never run.
- **Prompt version:** `v1` (rag), `v1` (question generation), `v1` (evaluation).
- **Dataset version:** `1.0.0`. Each report carries run id, timestamp, fingerprint, top_k, threshold.
- Offline runs are deterministic; model runs would not be bit-for-bit reproducible.

## 8. Tests

- Total (default selection): **659**.
- Passed: **659** (617 earlier + 42 new in `tests/test_phase17_quality.py`).
- Failed: **0**.
- Skipped: **0**; 9 integration tests deselected by `pytest.ini` (need Ollama) - NOT RUN.
- `ruff check --select E9,F app scripts tests`: clean. `scripts/release_check.py`: all PASS.
- Quality gate: `--deterministic-only` exit 0 ("QUALITY: INSUFFICIENT EVIDENCE"); strict run exit 10
  (`INSUFFICIENT EVIDENCE`) - correct, because no model evidence exists.

## 9. Files Created

- `evaluation/datasets/{corpus.json, rag_eval.jsonl, question_generation_eval.jsonl, answer_evaluation_eval.jsonl}`
- `evaluation/{README.md, HUMAN_REVIEW_RUBRIC.md}`, `evaluation/baselines/README.md`
- `evaluation/reports/` (generated offline reports) and `evaluation/reports/sweeps/` (threshold-0.1 RAG report)
- `scripts/{eval_common.py, rag_quality.py, evaluate_question_generation.py, evaluate_answer_evaluation.py, quality_gate.py}`
- `tests/test_phase17_quality.py`
- `docs/{AI_QUALITY_EVALUATION.md, RAG_EVALUATION.md, QUESTION_GENERATION_EVALUATION.md, ANSWER_EVALUATION_QUALITY.md, PHASE17_REPORT.md}`

## 10. Files Modified

- `scripts/evaluate_rag.py` - `.jsonl` datasets dispatch to the new suite (legacy `.json` flow unchanged),
  new flags `--mode --out-dir --answers --min --threshold --validate-only`.
- `app/utils/rag_metrics.py` - added `reciprocal_rank`, `ndcg_at_k` (existing functions untouched).
- `.github/workflows/ci.yml` - always-on dataset validation + offline run + deterministic gate;
  opt-in `ollama-quality` job (`workflow_dispatch` input, self-hosted runner; **UNVERIFIED**).
- No production model, prompt, API contract, schema or business logic was changed.

## 11. Known Limitations

- Datasets are small, synthetic, English-only, author-labelled; no human review.
- No model-dependent result exists; every number above is a harness or pipeline measurement.
- Offline retrieval numbers come from a lexical embedder and say nothing about `nomic-embed-text`.
- No LLM judge; faithfulness/relevance/completeness are token-overlap proxies (4 of 12 reference
  questions already trip them).
- Semantic difficulty and question clarity are not measured automatically.
- Gate tolerance (0.05) and minimum case count (30) are provisional, not derived from baseline variance;
  `evaluation/baselines/` is empty.
- `correct` is not defined by a score threshold (documented; contract unchanged).
- Vietnamese-language behaviour is not evaluated.

## 12. Unverified Items

- All `--mode ollama` runs (RAG with real embeddings, answers/faithfulness, question generation, grading).
- Real-model prompt-injection resistance, prompt/secret disclosure refusal, abstention behaviour.
- Insufficient-context detection with the production embedder and `SIMILARITY_THRESHOLD=0.3`.
- Human review (`HUMAN REVIEW NOT PERFORMED`), inter-reviewer agreement.
- The opt-in CI job `ollama-quality` (needs a self-hosted runner with Ollama).
- Docker, Ollama-based integration tests, git commit identification, Module 1/2 integration.

## 13. Final Status

**COMPLETE WITH LIMITATIONS** - the evaluation infrastructure, datasets, deterministic safety checks,
quality gate, CI integration, documentation and report are in place and verified; AI model quality itself
is **not** verified.

PHASE 17 COMPLETE / VERIFIED WITH KNOWN LIMITATIONS
