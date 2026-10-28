# Answer evaluation quality

Script: `scripts/evaluate_answer_evaluation.py`. Dataset: `evaluation/datasets/answer_evaluation_eval.jsonl`
(v1.0.0, 31 cases, synthetic, author-labelled, not human-reviewed). Overview:
[AI_QUALITY_EVALUATION.md](AI_QUALITY_EVALUATION.md). Service under test: `EvaluationService`
(contract unchanged).

## Rubric and ground truth

Each case has a **score band** in points of `max_score` written before any run (`max_score` is 10 for
most cases, 5 or 20 for a few to test normalisation):

| category (cases) | band on 10 | `expected_correct` |
|---|---|---|
| correct (4), paraphrase (3 + 2 with other max_score) | 8-10 | true |
| incomplete expected answer (2) | 7-10 | true (answer matches the passage, reference is thin) |
| partial (3) | 3-7 | false |
| unsupported claims (1: correct core + fabricated facts) | 5-9 | not asserted |
| wrong (4 + 1), contradictory (3), irrelevant (1), prompt injection (3) | 0-2 | false |
| empty answer, over-long answer (10 001 chars), insufficient context, cross-document probe (4) | no band; **expected error code** | - |

Cases are grouped (`consistency_group`, `rank`): within a group a better answer (lower rank) must not
score lower than a worse one.

**Tolerance:** the band itself. Bands are 2-3 points wide on a 10-point scale because two careful human
graders of short answers commonly differ by about that much. They were set by the author, not calibrated
against human graders, and are **not an academic standard**; do not read "within band" as "accurate
enough for grading students".

## Metrics

All score errors are normalised by `max_score` so cases are comparable.

* `mae_normalized_vs_band_midpoint`, `rmse_normalized_vs_band_midpoint` - distance to the band midpoint
  (a coarse reference, not a single true score).
* `mean_distance_outside_band_points`, `within_band_rate`.
* `correct_label_accuracy` over cases with a non-null `expected_correct` (25).
* `over_scoring_rate_wrong_answers` (above band for wrong/contradictory/irrelevant/injection) and
  `under_scoring_rate_paraphrases` (below band for paraphrases).
* `score_in_range_rate`, `backend_percentage_ok_rate` (percentage == round(score/max*100, 2)).
* `rank_consistency_violations` and listed findings.
* Error-contract cases: the right error code, and the LLM **not** called (`llm_called_on_error_case`).
* `correct`-vs-score disagreement list (see below).

## `correct` definition - recorded inconsistency, not changed

`correct` is produced by the model. The backend only rejects `correct` with score 0 and `not correct` with
full marks (`validate_llm_result`). Nothing ties `correct` to a score threshold, so `score 9, correct false`
and `score 1, correct true` are both accepted (tested in `test_documented_correct_vs_score_gap_is_real_and_unchanged`).
This was **not** altered. Controlled proposal for the owner (not implemented): either document `correct`
as an independent model verdict, or derive it as `percentage >= N` in the backend and drop it from the LLM
schema - a contract change that needs Module 2's agreement. Reports list any run where `correct` disagrees
with the percentage (correct below 50 %, or incorrect at 70 % or more).

## Results obtained (offline harness self-check)

Mode `offline`: the "LLM" returns the **band midpoint and expected label**, so score metrics are
trivially perfect and prove only that the pipeline and metrics work. They are **not** grading quality.

| metric | value |
|---|---|
| cases / graded | 31 / 26 |
| MAE, RMSE (normalised) | 0.0, 0.0 (by construction) |
| within band, label accuracy (25 labelled) | 1.0, 1.0 (by construction) |
| over-scoring, under-scoring | 0.0, 0.0 (by construction) |
| score in range, backend percentage OK | 1.0, 1.0 |
| rank-consistency violations | 0 (after removing the ambiguous `eval_006` from its group, see changelog) |
| sources outside the document | 0 |
| error-contract cases correct | 3 of 4: `EMPTY_STUDENT_ANSWER`, `INPUT_TOO_LONG`, cross-document probe `INSUFFICIENT_CONTEXT`; LLM never called |
| **failures** | `eval_030` (quantum question on an ML document) was *graded* instead of rejected; `eval_024` (answerable case) was rejected with `INSUFFICIENT_CONTEXT` |

The two failures are genuine outcomes of the **lexical test embedder** at threshold 0.3 (chance word
overlap above / real overlap below the cut-off) and are not representative of `nomic-embed-text`; they
show that insufficient-context detection depends on the embedder and threshold and must be verified
with `--mode ollama`. Nothing was tuned to remove them.

Model-dependent grading quality (MAE, within-band, label accuracy, over/under-scoring, injection
resistance of the real model): **NOT VERIFIED**. Human review: **HUMAN REVIEW NOT PERFORMED**.

## Score consistency checks (deterministic, passing)

Score always within `[0, max_score]`; percentage always backend-computed; contradictory verdicts and
non-finite scores rejected; sources always from retrieved metadata of the requested document; student
answer never used for retrieval; injection text confined to the user-prompt data block.

## Limits

31 cases (26 graded) is far too few for calibration; the gate needs >= 30 graded model-run cases plus
a baseline for a verdict. Injection cases test the harness around the model; whether the real model
resists them is model-dependent.
