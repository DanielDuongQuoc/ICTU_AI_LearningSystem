# Quality Gates (Phase 18)

`app/services/promotion_gate.py` turns Phase 17 evidence into one verdict. It is pure (no I/O, no model call),
so the decision is reproducible from the stored evidence files. It **judges** evidence; it never produces it.

## Evidence

Stored under `<AI_VERSIONS_DIR>/evidence/<version_id>/` by `scripts/ai_version_cli.py evaluate`:
`rag.json`, `qgen.json`, `answer_eval.json` (the Phase 17 reports, same evaluators and datasets),
`functional_tests.json` (the deterministic pytest run), `evidence_status.json` (infrastructure problems).
SHA-256 of each file is stored with the verdict; `approve` and `promote` re-check them, so edited evidence
cannot ride on an old PASS.

## Verdicts (evaluated in this order)

| Verdict | Meaning | Resulting status |
|---|---|---|
| `FAIL_SAFETY` | any `source_integrity_violations` / `document_isolation_violations` > 0 (even in tiny/offline runs) | rejected |
| `FAIL_INTEGRITY` | malformed report; report's model/embedding/top_k/threshold/prompt version ≠ the candidate; mixed dataset versions | rejected |
| `FAIL_FUNCTIONAL` | functional test run failed | rejected |
| `BLOCKED` | a suite or the functional result is missing, or the evaluation could not run (e.g. Ollama unreachable) | inconclusive |
| `INCONCLUSIVE` | report is `offline` (scripted stand-ins), fewer than `min_cases`, or no baseline **and** no floors, or baseline not comparable | inconclusive |
| `FAIL_QUALITY` | a tracked metric fell more than `tolerance` below the baseline, or an operator floor is violated | rejected |
| `PASS` | everything above satisfied on real-model data | validated |

Running the evaluator successfully is never enough: the offline harness always ends at `INCONCLUSIVE`
(or `BLOCKED` without the functional result).

## Policy and defaults

`GatePolicy` (optional JSON file via `AI_GATE_POLICY_FILE`, `extra=forbid`):

| Field | Default | Why |
|---|---|---|
| `min_cases` | 30 per suite | Phase 17 treats smaller sets as unable to support a quality conclusion |
| `tolerance` | 0.05 absolute | same margin as `scripts/quality_gate.py` |
| `floors` | none | the repository has **no model-backed baseline**, so any absolute number would be arbitrary; add floors after measuring a real one |
| `require_baseline` | true | without a baseline or floors the verdict is INCONCLUSIVE |
| `require_functional_tests` | true | |

Tracked metrics are the Phase 17 ones (`hit@5`, `mrr`, `insufficient_context_detection_rate`,
`incorrect_abstention_rate`; `answerability_rate`, `groundedness_rate`, `duplication_rate`;
`within_band_rate`, `correct_label_accuracy`, `over_scoring_rate_wrong_answers`,
`under_scoring_rate_paraphrases`) with their direction (higher/lower is better). No new metrics.

Example policy file for the first promotion (values are examples, set from your own measurements):

```json
{"floors": {"rag": {"hit@5": 0.7}, "qgen": {"answerability_rate": 0.7}, "answer_eval": {"within_band_rate": 0.6}}}
```

## Statistics: what a PASS does not mean

The tolerance is a fixed margin, **not a significance test**. With ~30-40 cases a rate has several points of
sampling noise, so PASS means "no regression larger than the margin was observed", never "equivalent".
Datasets are synthetic-author labelled (Phase 17); no human review was performed. No confidence intervals are
computed because the sample sizes would not justify them.

## Distinguishing failure kinds

Functional (`FAIL_FUNCTIONAL`), data/evidence integrity (`FAIL_INTEGRITY`, `FAIL_SAFETY`), quality regression
(`FAIL_QUALITY`), insufficient data (`INCONCLUSIVE`) and infrastructure/not-run (`BLOCKED`) are separate
verdicts with reasons stored on the version record and in the audit log.

## CI

CI (no Ollama) runs the deterministic suite including the gate unit tests and verifies the prompt lock. It
never evaluates or promotes a real version. Model-dependent evaluation is the opt-in `ollama-quality`
workflow / a host with Ollama (`python scripts/ai_version_cli.py evaluate <id> --mode ollama`) — **UNVERIFIED**
in this repository (no Ollama was available).
