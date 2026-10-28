# Question generation evaluation

Script: `scripts/evaluate_question_generation.py`. Dataset: `evaluation/datasets/question_generation_eval.jsonl`
(v1.0.0, 12 cases, synthetic, author-labelled, not human-reviewed). Overview:
[AI_QUALITY_EVALUATION.md](AI_QUALITY_EVALUATION.md). The response schema is unchanged.

## Cases

12 cases over 3 fixture documents: easy / medium / hard x `short_answer`, `multiple_choice`,
`true_false`, `essay`. Each lists `required_key_points` and one author-written `reference_questions`
entry (a good example, not the only acceptable output).

## Criteria and how each is measured

The real `QuestionGenerationService` runs (retrieval, prompt, LLM-output validation, deduplication,
backend source attribution). Each produced question is then checked:

| criterion | check | strength |
|---|---|---|
| Answerability | >= 50% of the question's content words occur in the document | proxy |
| Groundedness / expected-answer correctness | >= 60% of the expected answer's content words occur in the document (true/false: the question instead) | proxy - a fluent but wrong answer built from document words passes |
| Relevance | `required_key_points` all present in question / answer / options | proxy |
| Clarity | not measured automatically | human review |
| Duplication | near-duplicates among everything generated per document (service's own `is_duplicate`) | deterministic |
| Difficulty consistency | label equals the request. **Semantic difficulty is not measured** (length is deliberately not used) | label only |
| Source integrity | every `source_chunks` id exists in the case's own document; chunks of another document are flagged | deterministic |
| MCQ contract | exactly 4 options, all different (case-insensitive), `correct_answer` among them, correct option supported by the document | deterministic + proxy |

## Results obtained (offline harness self-check)

`--mode offline` replays the dataset's *own reference questions* as the "LLM" output with similarity
threshold 0.0 (the generic retrieval query has no lexical overlap with passages; at 0.3 all 12 cases
correctly end in `INSUFFICIENT_CONTEXT`, which was also observed). So these numbers validate the
checker and the service's validation, **not a model**:

| metric | value |
|---|---|
| cases generated / errors | 12 / 0 |
| answerability rate | 0.8333 |
| groundedness rate | 0.8333 |
| key-point coverage (mean) | 0.9722 |
| duplication rate | 0.0 |
| type / difficulty-label match | 1.0 / 1.0 |
| MCQ structure OK / correct option supported | 1.0 / 1.0 (n = 3 MCQ questions) |
| source integrity rate | 1.0 (0 foreign chunk ids) |

The 0.83 values are informative: author-written *good* references fail at least one proxy in 4 of 12
cases (`qgen_003` groundedness, `qgen_005` answerability, `qgen_007` both, `qgen_011` key-point coverage:
short or abstract wording such as true/false statements, and one reference answer lacking the key-point
word "reward"). That is the false-negative rate of the lexical proxies themselves; it bounds how far any
model-run proxy score can be trusted. The dataset was not edited to hide this.

Model-dependent results: **NOT VERIFIED**. Human review: **HUMAN REVIEW NOT PERFORMED**.

## Limits

12 cases (2 per rare type) cannot support statistical claims; the gate requires >= 30 for a quality
verdict, so this suite alone yields `INSUFFICIENT EVIDENCE`. Essay is supported by the implementation and
is covered. English only.
