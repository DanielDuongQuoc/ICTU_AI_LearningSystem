# Answer Evaluation Quality Report

## Run metadata

- **evaluation_run_id**: answer_eval-20261023T214013Z-51c665
- **timestamp**: 2026-10-23T21:40:13.077755+00:00
- **git_commit**: unavailable
- **dataset_version**: 1.0.0
- **prompt_version**: {"evaluation": "v1"}
- **model_name**: scripted-test-llm
- **embedding_model**: lexical-test-embedding
- **top_k**: 5
- **similarity_threshold**: 0.3
- **configuration_fingerprint**: ecaa898f1bee4a9c
- **mode**: offline
- **evaluation_type**: deterministic_offline
- **human_review**: HUMAN REVIEW NOT PERFORMED
- **reproducibility_notes**: OFFLINE mode: lexical test embedder and a scripted (non-model) LLM. Numbers describe pipeline/harness behaviour, NOT the quality of the production model or embeddings.
- **label_sources**: synthetic_author

## Metrics

### Grading quality

| metric | value |
|---|---|
| cases | 31 |
| graded | 26 |
| mae_normalized_vs_band_midpoint | 0.0000 |
| rmse_normalized_vs_band_midpoint | 0.0000 |
| mean_distance_outside_band_points | 0.0000 |
| within_band_rate | 1.0000 |
| correct_label_accuracy | 1.0000 |
| correct_label_cases | 25 |
| over_scoring_rate_wrong_answers | 0.0000 |
| under_scoring_rate_paraphrases | 0.0000 |
| score_in_range_rate | 1.0000 |
| backend_percentage_ok_rate | 1.0000 |
| rank_consistency_violations | 0 |

### Deterministic error / abstention cases

| metric | value |
|---|---|
| error_code_cases | 4 |
| error_code_correct | 3 |
| error_cases_that_returned_a_score | 1 |
| unexpected_errors | [{"case_id": "eval_024", "code": "INSUFFICIENT_CONTEXT"}] |
| llm_called_on_error_case | 0 |

### Security invariants

| metric | value |
|---|---|
| sources_outside_document | 0 |

## Findings

- OFFLINE harness self-check: the 'LLM' returns the band midpoint, so score metrics validate the pipeline/metrics, NOT grading quality. Error-code and insufficient-context cases exercise real service behaviour.
- Bands were written by the dataset author before any run and are not calibrated against human graders. HUMAN REVIEW NOT PERFORMED.
- The `correct` flag is produced by the LLM; the backend only rejects correct+score 0 and incorrect+full score. No score threshold defines `correct`, so label accuracy and score bands can legitimately disagree. This contract was NOT changed.
- INSUFFICIENT_CONTEXT outcomes depend on the embedding model and SIMILARITY_THRESHOLD; in offline mode they come from the lexical test embedder and are NOT representative of the production embedding model (nomic-embed-text). Re-run with --mode ollama to verify.
