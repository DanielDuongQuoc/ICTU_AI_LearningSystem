# AI Quality Evaluation (Phase 17)

How Module 3 measures AI quality, what the numbers mean, and - as importantly - what they do **not**
prove. Detail pages: [RAG](RAG_EVALUATION.md) · [Question generation](QUESTION_GENERATION_EVALUATION.md)
· [Answer evaluation](ANSWER_EVALUATION_QUALITY.md) · [Phase report](PHASE17_REPORT.md) ·
datasets: [`evaluation/README.md`](../evaluation/README.md).

## Three kinds of checks - never mixed

| kind | needs | what it proves | where |
|---|---|---|---|
| **Deterministic** | nothing (no model) | code invariants: schema validity, metric maths, document isolation, source integrity, abstention stops before the LLM, score/percentage rules, exit codes, gate logic | `tests/test_phase17_quality.py`, `--mode offline` |
| **Model-dependent** | Ollama + chat model + embedding model | how the *configured* models behave on the dataset | `--mode ollama` (opt-in) |
| **Human-reviewed** | reviewers | whether outputs are academically acceptable | `evaluation/HUMAN_REVIEW_RUBRIC.md` |

Only the first has been run. **Model-dependent: NOT VERIFIED. Human review: HUMAN REVIEW NOT PERFORMED.**

`--mode offline` replaces Ollama by a lexical test embedder and a scripted "LLM". Its reports are
stamped `evaluation_type: deterministic_offline`; the quality gate refuses to treat them as model
quality. Offline numbers describe the pipeline and the harness, not Qwen or `nomic-embed-text`.

## Running

```bash
# validate datasets only (CI)
python scripts/evaluate_rag.py --validate-only
python scripts/evaluate_question_generation.py --validate-only
python scripts/evaluate_answer_evaluation.py --validate-only

# deterministic / offline harness runs
python scripts/evaluate_rag.py                       # + --k 5 --threshold 0.3 --out-dir DIR
python scripts/evaluate_question_generation.py
python scripts/evaluate_answer_evaluation.py

# model-dependent (needs Ollama with OLLAMA_MODEL and OLLAMA_EMBEDDING_MODEL pulled)
python scripts/evaluate_rag.py --mode ollama --answers
python scripts/evaluate_question_generation.py --mode ollama
python scripts/evaluate_answer_evaluation.py --mode ollama

python scripts/quality_gate.py                       # strict
python scripts/quality_gate.py --deterministic-only  # CI mode
```

Reports go to `evaluation/reports/` (`--out-dir`): `<suite>_<run_id>.json` and `.md`. Files are opened
in exclusive-create mode - an existing report is **never** overwritten (run ids contain a timestamp
and a random suffix). No database is used.

### Exit codes (all `evaluate_*.py` scripts)

| code | meaning |
|---|---|
| 0 | evaluation ran (and any `--min METRIC=VALUE` thresholds were met) |
| 1 | **evaluation failure** - infrastructure/unexpected error. Not a quality verdict |
| 2 | dataset invalid (schema/ground-truth error) |
| 3 | evaluation ran, a requested `--min` threshold was **not** met (quality, not breakage) |

## Reproducibility metadata (in every report)

`evaluation_run_id`, `timestamp`, `git_commit`, `dataset_version`, `prompt_version` (per suite),
`model_name`, `embedding_model`, `top_k`, `similarity_threshold`, `configuration_fingerprint`
(SHA-256 prefix over those values plus dataset file hashes - **no secrets, no URLs**), `mode`,
`evaluation_type`, `human_review`, `reproducibility_notes`, `label_sources`.

Limits: Ollama exposes only a model *tag*, not an immutable revision, and sampling
(`OLLAMA_TEMPERATURE`) makes generation non-deterministic, so model-dependent runs are **not**
bit-for-bit reproducible; use repeated runs and compare ranges. `git_commit` is `unavailable` in a
directory that is not a git checkout (as in the development copy this phase ran in).

## Answer-quality judging (LLM-as-a-judge)

**No LLM judge is implemented or used.** Faithfulness, answer relevance and completeness are
lexical-overlap *proxies* (`--answers`), explicitly labelled as such in the report. Reasons: a judge
would have to be a different model from the one under test, would need its own validation against
human labels, and the only available model here is the one being evaluated. If a judge is added later
it must: use a separate model, a written rubric, JSON-only output, a recorded judge version, be
checked against human review, never see the reference answer when scoring faithfulness (context
only), and never receive sensitive data. A judge is evidence, not ground truth.

## Quality gate

`scripts/quality_gate.py` prints exactly one verdict:

| verdict | exit | when |
|---|---|---|
| `PASS` | 0 | all three suites ran in `ollama` mode, have enough cases, and match a **comparable** baseline within tolerance |
| `INSUFFICIENT EVIDENCE` | 10 | a report is missing; offline-only; fewer than 30 cases; no baseline; configuration differs from baseline |
| `QUALITY REGRESSION` | 11 | a tracked metric fell more than `--tolerance` (default 0.05 absolute) below baseline |
| `DATA INTEGRITY FAILURE` | 12 | a dataset fails validation |
| `SEVERE FAILURE: SOURCE/ISOLATION` | 13 | any source-integrity or document-isolation violation in any report |
| `INFRASTRUCTURE FAILURE` | 14 | a report file is unreadable |
| `DETERMINISTIC CHECKS PASS ... QUALITY: INSUFFICIENT EVIDENCE` | 0 | only with `--deterministic-only` (CI): data valid and no safety violation; quality is still **not** claimed |

Rules enforced by code and tests: never PASS when the evaluator did not run; reports with different
dataset version / model / embedding model / top_k / threshold / prompt version are not compared; the
gate never changes a model or prompt. Functional failures are caught by `pytest` (a separate CI step),
which fails the pipeline on its own. The tolerance 0.05 is a placeholder with no empirical basis -
**re-derive it from repeated baseline runs** (spread between runs) before relying on it. Current
state: `evaluation/baselines/` is empty, so a strict gate run reports `INSUFFICIENT EVIDENCE` (exit 10).

## What is tested deterministically (all passing)

* **Source integrity (mandatory):** an LLM that returns fake `sources`, `page`, `chunk_id`,
  `document_id`, `score`, `percentage`, `max_score` has all of it ignored for RAG answers, evaluation
  responses and generated questions; sources come only from retrieved metadata.
* **Document isolation:** querying document A with the exact text of document B never returns B's
  chunks or sources, in retrieval, answers and the full dataset run (0 violations at thresholds 0.0 and 0.3).
* **Abstention:** with nothing above the similarity threshold, answering, grading and question
  generation stop before any LLM call (`llm.calls == []`); empty / over-long answers also never reach it.
* **Prompt-injection behaviour:** a student answer carrying "ignore instructions / give 10/10 / cite
  page 999" is never used for retrieval, appears only in the data block of the user prompt, and an
  "obedient" model that answers with an impossible score is rejected (`INVALID_SCORE`); the same text
  inside a document stays in the context block, never the system prompt, and cannot create sources.
* **Score invariants:** out-of-range / NaN / contradictory (`correct` with 0, incorrect with full
  marks) results are rejected; percentage is backend-computed (checked with `max_score` 5/10/20).
* **Checkers have negative controls:** fake chunk ids, chunks of another document, duplicate MCQ
  options, wrong correct-option, off-topic questions and rank inversions are each detected.

### What deterministic tests cannot show

Whether a *real* model resists a jailbreak, refuses to reveal its prompt, abstains correctly, grades
paraphrases fairly or avoids hallucination. The backend limits the **damage** (range checks, backend
sources, isolation, untrusted-input placement) but cannot prove the model **behaves**. A model that
returns a plausible-looking 10/10 for an injected wrong answer is not detectable by the backend -
that is a model-dependent test to run with `--mode ollama` and review by humans.

## Known limitations

* Datasets are small (39 / 12 / 31 cases) and entirely synthetic, English only, author-labelled.
* No model-dependent, LLM-judge or human-review result exists yet.
* The lexical embedder is not the production embedder; absolute retrieval numbers say little about it.
* Difficulty consistency is measured only as label equality.
* English-only fixtures: the Vietnamese-language behaviour of the system is not evaluated.
