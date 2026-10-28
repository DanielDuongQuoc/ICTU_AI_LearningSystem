# Incident Response (Phase 16)

Proposed internal process. The severity levels are suggestions, **not an SLA**. No incident data is
sent to any external service by this module or its scripts.

```text
Detect -> Assess severity -> Contain -> Diagnose -> Recover -> Verify -> Document -> Prevent recurrence
```

## Severity (proposed)

| Level | Definition | Examples |
|---|---|---|
| SEV-1 | data loss, data exposure, or service down | Chroma data lost, document A's context returned for document B, secret in logs, API not serving |
| SEV-2 | main function unavailable or widely failing | all evaluations 5xx, Ollama down, `/ready` 503 |
| SEV-3 | degraded with a workaround, limited impact | slow p95, intermittent `SERVICE_OVERLOADED`, one stream type failing |
| SEV-4 | minor issue or improvement | cosmetic log issue, documentation gap |

## Steps

1. **Detect**: alert from `scripts/ops_check.py` or your monitor, a user report, or log review. Record the time and a failing `X-Request-ID`.
2. **Assess**: pick a severity. Anything touching isolation, data loss or leaked content is SEV-1 until proven otherwise.
3. **Contain**: stop the harm first (stop the API for suspected corruption or leakage; do not delete data; do not run `docker compose down -v`). For overload, rely on `LLM_MAX_CONCURRENCY` and shed load at the caller.
4. **Diagnose**: follow the matching case in [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md). Correlate by `request_id`; compare `/metrics` by route, error code and stage. Separate *verified* cause from *hypothesis* in the notes.
5. **Recover**: smallest safe action (restart one dependency, correct configuration, roll back, or restore a backup into an isolated directory first). Restoring replaces data: follow [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md) and keep the pre-restore copy.
6. **Verify**: `/ready` 200, `scripts/smoke_test.py`, a known document retrieves only itself, one evaluation and one stream complete.
7. **Document**: fill the template below.
8. **Prevent recurrence**: add a regression test for any code defect; update the runbook, thresholds or configuration.

## Incident record template

```text
ID / title:
Severity (and why):
Detected at / by:            (timestamps with timezone)
Mitigated at / Resolved at:
Impact:                      (who, which endpoints, how long, any data affected)
Timeline:
Cause - VERIFIED:            (evidence: logs, request_id, metrics)
Cause - HYPOTHESIS:          (clearly marked, not yet proven)
Actions taken:
Verification results:        (commands and outcomes)
Prevention / follow-ups:     (owner, due date, test added?)
Data sent outside the system: none (state explicitly if anything was shared)
```

## Escalation triggers
- Any SEV-1.
- Suspected cross-document leakage, or a secret/student text found in logs.
- Restore needed but no valid (checksum-verified) backup exists.
- The same SEV-2/3 repeating after the documented remediation.
