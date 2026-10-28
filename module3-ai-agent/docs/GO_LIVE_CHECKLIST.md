# Go-Live Checklist (Phase 21)

No owners are named: this repository defines no roles. Fill the **Responsible** column with the real person/role when
the team assigns one. Status values: PASS / FAIL / BLOCKED / NOT VERIFIED / N/A. Evidence is in PHASE21_REPORT.md.

| Area | Item | Status | Responsible | Evidence / note |
|---|---|---|---|---|
| Application | Full offline test suite green | see report | _unassigned_ | `validate_release` gate `functional` |
| Application | API contract frozen, OpenAPI matches | see report | _unassigned_ | `tests/release/test_release_gates_and_contract.py` |
| Application | Production config validated (pinned models, no CORS `*`, no DEBUG) | PASS (unit tests) | _unassigned_ | real production `.env` NOT VERIFIED |
| Infrastructure | Docker image builds | NOT VERIFIED | _unassigned_ | no Docker available |
| Infrastructure | Compose stack starts, volumes mapped | NOT VERIFIED | _unassigned_ | static audit only |
| Infrastructure | Ollama image/models pinned and pulled | NOT VERIFIED | _unassigned_ | `OLLAMA_IMAGE` defaults to latest |
| AI | Quality gate vs baseline with a live model | NOT VERIFIED | _unassigned_ | deterministic part only |
| AI | Active AI version identified, rollback target known | NOT VERIFIED on host | _unassigned_ | rollback tested with mocks |
| RAG | Embedding model matches stored vectors | PASS (tests) / legacy data unknown | _unassigned_ | `services.embedding_index` |
| Security | Phase 19 security suite green | see report | _unassigned_ | gate `security` |
| Security | Dependency audit, external pen-test | NOT VERIFIED | _unassigned_ | CI job is advisory only |
| Observability | `/metrics`, logs, runbook scenarios in place | PASS (code/tests) | _unassigned_ | no dashboards/alerts deployed |
| Backup | Backup of production data taken | NOT VERIFIED | _unassigned_ | drill used test data only |
| Recovery | Restore rehearsed on a staging copy | NOT VERIFIED | _unassigned_ | drill on test data passed |
| Rollback | Previous image + backup retained | NOT VERIFIED | _unassigned_ | |
| Documentation | Runbooks and checklists current | PASS | _unassigned_ | this folder |

Go-live is a **human decision**. Any FAIL/BLOCKED row stops it; each NOT VERIFIED row must be accepted explicitly or closed.
