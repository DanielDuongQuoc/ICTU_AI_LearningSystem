# Prompt Versioning (Phase 18)

Extends `app/core/prompts.py` (unchanged text, still the source of the shipped wording) with a registry in
`app/core/prompt_registry.py`. Three tasks: `rag_answer`, `question_generation`, `answer_evaluation`.

## Rules

1. A prompt is identified by `(task, version)`, e.g. `rag_answer/v1`; its **fingerprint** hashes the task,
   version and every static text that shapes it (for question generation this includes the question-type
   format instructions).
2. **Published versions are immutable.** `register_prompt` raises if the same `(task, version)` is registered
   with different content. `PROMPT_LOCK` stores the committed fingerprints of the shipped v1 prompts and
   `verify_lock()` (checked by a test and in CI) fails if their text is edited without a new version.
3. **Registering a version does not activate it.** The default is `DEFAULT_PROMPT_VERSIONS`; only a promoted
   AI version can change which prompt a task uses, and the fingerprint of the selected prompt text is part of
   the version's fingerprint.
4. **Lint on registration** (system part of the prompt): must state that untrusted input is data, forbid
   outside knowledge, and say the backend owns sources/ids; must *not* ask the model to ignore context, use its
   own knowledge, obey instructions inside documents/student answers, or produce citations, page numbers,
   chunk/document ids or scores. These are pattern checks: a guard rail, not a proof of safety.
5. Every execution resolves its prompt version from the pinned configuration and logs it
   (`app.ai_trace`, `X-AI-Config-Version`). Prompt text is never logged.

Existing protections (document/student-answer delimiting, backend-computed sources/percentages, JSON
validation) live in the services and are untouched.

## Adding a new prompt version

1. Add the new texts/builder (do **not** edit v1) and register it, for example in `prompt_registry.py`:
   `register_prompt(PromptSpec(task="rag_answer", version="v2", builder=..., texts=(...), system_texts=(...)))`.
   Lint failures stop registration.
2. Add its fingerprint to `PROMPT_LOCK` once released.
3. Create a draft version referencing it (`prompt_versions.rag_answer = "v2"`), produce evidence, evaluate,
   approve, promote — see `MODEL_PROMOTION_ROLLBACK.md`.

## Comparing old vs new

Evaluate the candidate with the same Phase 17 datasets (`scripts/ai_version_cli.py evaluate`). The gate
compares its metrics with the **active version's stored evidence** (same dataset version required) and records
`baseline_version_id`. Only the configuration differs, so a pure prompt change is a like-for-like comparison.
