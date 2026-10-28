# Evaluation assets (Phase 17)

Version-controlled datasets and reports used to measure AI quality of Module 3.
Full method, metrics and interpretation: [`docs/AI_QUALITY_EVALUATION.md`](../docs/AI_QUALITY_EVALUATION.md).

```
evaluation/
├── datasets/
│   ├── corpus.json                       # 3 synthetic documents, 8 passages each, fixed chunk ids
│   ├── rag_eval.jsonl                    # 39 retrieval / abstention cases
│   ├── question_generation_eval.jsonl    # 12 question-generation cases
│   └── answer_evaluation_eval.jsonl      # 31 grading cases
├── reports/                              # generated JSON + Markdown reports (never overwritten)
├── baselines/                            # reports promoted as baselines (EMPTY - none verified yet)
└── HUMAN_REVIEW_RUBRIC.md                # rubric + record sheet (HUMAN REVIEW NOT PERFORMED)
```

## Provenance - read this first

* **Dataset version `1.0.0`** (`corpus.json: dataset_version`). Bump it on ANY change to a dataset
  file; reports from different versions must not be compared.
* **All data is synthetic** (`label_source: "synthetic_author"`): passages and rubric bands were
  written by the project author. There is **no personal data and no student work**.
* **No case has been human-reviewed.** Ground truth is author-assigned. Cases that a second person
  has checked must be flagged `label_source: "human_reviewed"` (see the rubric file).
* `document_id` values such as `fixture_doc_001` and `chunk_id` values such as `fixture_chunk_001`
  are *fixture names*. At run time every fixture document is indexed in a **temporary ChromaDB under a
  fresh UUID**; nothing refers to, reads or writes the production ChromaDB.
* The datasets were written **before** any run, and were not tuned to improve a number. The only
  post-hoc change is recorded in the changelog below.

## Schemas

All files are JSON Lines (one object per line); `corpus.json` is a single JSON object.

### `corpus.json`
`dataset_version`, `provenance`, `provenance_note`, `language`, `documents[]` ->
`{document_id, title, passages[] -> {chunk_id, page, text}}`. Chunk ids are unique across the corpus.

### `rag_eval.jsonl`
| field | meaning |
|---|---|
| `case_id`, `document_id`, `query` | identity and the question asked of ONE document |
| `relevant_chunk_ids` | ground-truth chunks (empty only when `expect_abstain`) |
| `expected_key_points` | facts a complete answer contains (used by completeness proxy) |
| `category` | `answerable`, `paraphrased`, `partial`, `false_premise`, `out_of_scope`, `outside_knowledge_request`, `fabricate_source_request` |
| `expect_abstain` | the document cannot support an answer |
| `label_source` | `synthetic_author` or `human_reviewed` |

### `question_generation_eval.jsonl`
`case_id`, `document_id`, `question_type`, `difficulty`, `number_of_questions`,
`required_key_points`, `reference_questions[]` (author-written good questions: `question`,
`expected_answer`, `difficulty`, `type`, plus `options`/`correct_answer` for multiple choice),
`expected_properties`, `label_source`.

### `answer_evaluation_eval.jsonl`
`case_id`, `document_id`, `question`, `expected_answer`, `student_answer`, `max_score`,
`expected_score_min/max` (rubric band, in points of `max_score`; `null` for error-contract cases),
`expected_correct` (`true`/`false`/`null` = not asserted), `category`, `context_chunk_ids`,
`consistency_group` + `rank` (lower rank = better answer; scores must not increase with rank),
`expected_error_code` (empty answer, over-long answer, insufficient context),
`pad_student_answer_to_chars`, `label_source`.

Validate without running anything:

```bash
python scripts/evaluate_rag.py --validate-only
python scripts/evaluate_question_generation.py --validate-only
python scripts/evaluate_answer_evaluation.py --validate-only
```

## Changelog

* `1.0.0` - initial datasets. One correction before any model-quality run: `eval_006`
  (answer with a correct core plus fabricated claims, band 5-9) was removed from consistency group
  `g_sup` because its ordering relative to the partial answer `eval_003` (band 3-7) is a judgement,
  not a ground truth; the offline self-check exposed the inconsistency.
