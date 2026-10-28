# AI Version Governance (Phase 18)

Controls **which AI configuration serves requests** and makes every result traceable to it. Built on what
already existed: `app/core/prompts.py` (prompt text), Phase 17 evaluators/datasets (evidence), `Settings`
(`.env`), the request-ID middleware and the file-based backup scripts. No database, queue or MLOps platform
was added.

## Components

| Concern | Where |
|---|---|
| Prompt registry (id + version + content fingerprint, immutable, lint) | `app/core/prompt_registry.py` |
| Effective config = `.env` overlaid with the active version; per-request pinning | `app/core/runtime_config.py` |
| Manifest/request schemas (no `status` input, secret-text guard) | `app/schemas/ai_version.py` |
| Durable store: manifests, `state.json`, `audit.jsonl`, `evidence/` | `app/repositories/ai_version_repository.py` |
| Lifecycle, preflight, promote/rollback | `app/services/ai_version_service.py` |
| Pure promotion gate over Phase 17 reports | `app/services/promotion_gate.py` |
| Admin API | `app/api/ai_versions.py` |
| Evidence producer (host side) | `scripts/ai_version_cli.py` |
| Per-execution trace log (`app.ai_trace`) | `app/core/execution_trace.py` |

## What a manifest is

`<AI_VERSIONS_DIR>/manifests/<version_id>.json`, written once with exclusive-create and never rewritten:
`version_id`, `config_fingerprint`, `created_at/by`, `description`, `config`
(`llm{provider,model,temperature,num_predict}`, `embedding_model`, `prompt_versions`,
`rag{top_k,similarity_threshold,max_context_chars}`, `chunking{chunk_size,chunk_overlap}`),
`prompt_fingerprints`, `change_summary`, `previous_version_id`.

Everything that **changes** lives separately in `state.json` (status, evaluation verdict, approval,
activation, `active_version_id`, a `revision` counter). That is the separation of *proposed* (draft),
*validated/approved* and *active*.

**Fingerprint** = SHA-256 of canonical JSON (sorted keys, no whitespace) of the config **plus the content
fingerprint of every selected prompt**. Key order and whitespace do not matter; editing prompt text under an
unchanged version label does. No secret can enter: the schema has no credential fields and rejects
secret-looking text in descriptions, reasons and model names.

## Lifecycle

```
draft --evaluate--> validated   (gate PASS)
                --> inconclusive (gate INCONCLUSIVE / BLOCKED; re-evaluate after more evidence)
                --> rejected     (gate FAIL_*; terminal)
validated --approve--> approved --promote--> active
active --(other promote)--> superseded      active --(rollback)--> rolled_back
approved | superseded | rolled_back --rollback--> active
draft | inconclusive | validated | approved --reject--> rejected
```

`created_at → evaluate → approve → promote` are separate calls; none can be skipped, and no request body can
carry a status.

## What kind of change is it?

`change_summary` classifies a draft: `prompt:<task> vA->vB`, `llm_model`, `generation_params`,
`retrieval_params`. **Embedding model and chunking changes are refused at creation** (`AI_VERSION_INVALID`)
because they would make the live ChromaDB vectors inconsistent with queries. Changing them needs a
re-embedding/re-ingestion procedure into a *new* collection; that is not automated here (see
"Not implemented").

## Request-level traceability

* Every response carries `X-AI-Config-Version` and `X-AI-Config-Fingerprint` (first 16 hex chars).
* The configuration is **pinned when a request starts**; a promotion mid-request affects only later requests.
* RAG answer, question generation and evaluation emit one `app.ai_trace` log line with `request_id`,
  version id, config/RAG fingerprints, models, prompt version, outcome, latency and backend-owned source
  chunk ids. No document text, prompts, student answers or secrets.

`audit.jsonl` answers: who/what changed the active version, when, why (`reason`), from which request
(`request_id`), and which evidence/policy fingerprint justified it.

## Honest limits

* **Configuration reproducibility ≠ output reproducibility.** The same version gives the same settings, not
  the same tokens.
* **Ollama tags are mutable and models can be deleted.** A manifest stores `name:tag`, not weights. Preflight
  checks that the model is *installed*; it cannot prove it is the same weights.
* **Admin auth is a single shared token** (`ADMIN_API_TOKEN`, header `X-Admin-Token`). No users, roles or
  separation of duties; `X-Admin-Actor` is an unauthenticated label. Disabled (503) when unset. Keep the API
  off the public network.
* The store is a local directory (Docker: mount a volume, or the active version resets with the container).
  Multi-host deployments need a shared volume; locking is per-host file locking.
* Workers converge via a per-request `state.json` mtime check; there is no push notification.
