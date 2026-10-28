# Human review rubric (questions and graded answers)

**Status: HUMAN REVIEW NOT PERFORMED.** No reviewer has scored any sample, so no claim of verified
academic quality is made anywhere in this repository. This file is the instrument to use when
reviewers become available.

## Scale (per criterion, per sample)

`2` = fully satisfied, `1` = partly, `0` = not satisfied, `N/A` = not applicable.

| criterion | applies to | question reviewers answer |
|---|---|---|
| Correctness | question, answer-grade | Is the expected answer / the awarded verdict factually right given the passage? |
| Groundedness | question, answer-grade | Is every claim in the question/feedback supported by the passage, with nothing from outside? |
| Completeness | question, answer-grade | Does it cover the passage's key points (question) / identify all missing points (grade)? |
| Clarity | question | Is it unambiguous and answerable in one reading? (multiple choice: distractors plausible, exactly one right) |
| Source accuracy | question, answer-grade | Do the returned source chunks/pages really contain the supporting text? |
| Fairness of scoring | answer-grade | Would a careful teacher give a score within +/-1 of this on the same scale, without penalising paraphrase or rewarding fabrication? |

## Procedure

1. Sample: at least 20 items per suite, stratified by category (include every failure-prone category:
   paraphrase, partial, contradictory, injection, insufficient-context). Use outputs from the real
   model run (`--mode ollama`) - never from the offline scripted run.
2. Two reviewers score **independently**, without seeing each other's sheet or the automatic metrics.
3. Compute agreement (percentage of identical scores and Cohen's kappa per criterion). Report it; if
   kappa < 0.4 the rubric, not the model, is the first thing to fix.
4. **Disagreements:** a differences of 1 point is averaged; a difference of 2 (0 vs 2) goes to a third
   reviewer or joint discussion. Record the final value, who decided, and why in the `resolution` column.
5. Cases both reviewers confirm get `label_source: "human_reviewed"` and a dataset version bump.

## Record sheet (copy per reviewer)

| case_id | criterion scores (C / G / Co / Cl / S / F) | free-text note | reviewer | date |
|---|---|---|---|---|
| | | | | |

Disagreement log:

| case_id | reviewer A | reviewer B | resolution | decided by |
|---|---|---|---|---|
| | | | | |

Results table (fill from the sheets - **currently empty, nothing was reviewed**):

| suite | items | mean score per criterion | agreement (%) | kappa |
|---|---|---|---|---|
| question generation | 0 | - | - | - |
| answer evaluation | 0 | - | - | - |
