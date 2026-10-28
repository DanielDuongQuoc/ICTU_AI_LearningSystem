# Model Promotion & Rollback Runbook (Phase 18)

All admin calls need `X-Admin-Token: $ADMIN_API_TOKEN` (the API returns 503 `ADMIN_DISABLED` if the token is
unset, 401 if wrong). Optional `X-Admin-Actor: your.name` labels the audit log (unauthenticated).

## 0. Prerequisites

* `ADMIN_API_TOKEN` set (secret, not in Git); `AI_VERSIONS_DIR` on a persistent volume.
* The candidate model is already pulled in Ollama **by you**; the system never pulls or deletes models.

## 1. Create a draft

```
python scripts/ai_version_cli.py show-env        # JSON of the current .env configuration
curl -X POST $API/api/ai-versions -H "X-Admin-Token: $T" -H "Content-Type: application/json" \
  -d '{"description":"qwen candidate, top_k 4","config":{...}}'
```

The body has no `status`. `embedding_model` and `chunking` must equal the running environment (else 422 with
the re-embedding explanation). Duplicate configurations → 409.

## 2. Produce evidence (host with Ollama)

```
python scripts/ai_version_cli.py evaluate <version_id> --mode ollama
```

Runs the three Phase 17 evaluators under the candidate's settings plus the pytest suite, and stores the
evidence. If Ollama is down, the run is recorded as infrastructure error → the gate says `BLOCKED`.
`--mode offline` only checks the harness (gate: `INCONCLUSIVE`/`BLOCKED`).

## 3. Evaluate → approve → promote

```
POST /api/ai-versions/{id}/evaluate    {"baseline_version_id": null}   # default baseline: active version
POST /api/ai-versions/{id}/approve     {"reason": "..."}               # only after gate PASS (status validated)
POST /api/ai-versions/{id}/promote     {"reason": "..."}
GET  /api/ai-versions/active                                           # effective config + drift
```

`promote` refuses (409 `AI_VERSION_PREFLIGHT_FAILED`, state unchanged, audited as `promote_blocked`) unless:
a PASS with unchanged evidence is on record; Ollama is reachable and both models are installed; prompts are
registered with unchanged content; embedding/chunking equal the environment; the manifest fingerprint is
intact. A static smoke test runs inside the same transaction before the change is committed. The smoke test
does **not** call the LLM.

Concurrent promote/rollback calls: the operation records the store revision it read and commits only if it is
unchanged (else 409 `AI_VERSION_CONFLICT` — re-read and retry). Writers are serialised by a lock file.

## 4. Rollback

```
POST /api/ai-versions/rollback   {"target_version_id": "<id>", "reason": "..."}
POST /api/ai-versions/rollback   {"target_version_id": "env-bootstrap", "reason": "..."}   # back to .env config
```

Target must have been approved (status `approved`, `superseded` or `rolled_back`) with its evidence intact;
the same preflight and smoke test run. The previous active version becomes `rolled_back`; nothing is deleted
or overwritten and the audit log records who/why. On failure you get an error, the active version is
unchanged. **There is no automatic rollback**: a single failed request is not a signal, and no alert-driven
rollback policy exists. Roll back manually on monitoring evidence (Phase 16 metrics/runbook).

## 5. Reject / inspect

`POST /api/ai-versions/{id}/reject` (terminal) · `GET /api/ai-versions[/{id}]` ·
`GET /api/ai-versions/audit?version_id=` · `X-AI-Config-Version` response header.

## When things fail

| Symptom | Meaning / action |
|---|---|
| gate `BLOCKED` | evidence missing or evaluation could not run (Ollama?) — fix infra, re-run `evaluate` |
| gate `INCONCLUSIVE` | offline/too few cases/no baseline — add data or floors; never force-approve |
| gate `FAIL_*` | version is `rejected` with reasons; create a new draft |
| `AI_VERSION_PREFLIGHT_FAILED` | read the listed problems (missing model, prompt drift, env mismatch) |
| `/active` shows `drift` or `store_degraded` | manifest/state problem; the process keeps its previous configuration; restore the store from backup |
| need a different embedding model/chunking | not a promotion: re-ingest into a new collection, then change `.env` deliberately |

## Backup

Include `AI_VERSIONS_DIR` in backups (it holds manifests, state, audit and evidence). It is not part of the
ChromaDB backup.
