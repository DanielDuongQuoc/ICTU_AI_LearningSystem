"""AI version governance service (Phase 18).

Lifecycle::

    draft --evaluate--> validated (gate PASS) | inconclusive (gate INCONCLUSIVE/BLOCKED) | rejected (FAIL_*)
    validated --approve--> approved --promote--> active
    active --(another promote)--> superseded        active --(rollback)--> rolled_back
    superseded | rolled_back | approved --rollback/promote--> active again (history is never rewritten)
    draft | inconclusive | validated | approved --reject--> rejected (terminal)

Rules enforced here (not in the router): clients cannot choose a status; ``approve`` needs a stored PASS
whose evidence hashes still match the files; ``promote``/``rollback`` run a preflight (models installed,
prompts registered with unchanged content, embedding/chunking equal to the running environment, manifest
fingerprint intact) plus a static smoke test BEFORE anything is persisted; every state change happens in one
repository transaction (lock + atomic replace + revision check) together with its audit event.

Limits that are documented rather than hidden: Ollama tags are mutable and models can be deleted, so a
manifest records the model NAME:TAG but cannot guarantee the same weights later; configuration
reproducibility is not token-exact output reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.core import prompt_registry as registry
from app.core import runtime_config as rc
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.repositories.ai_version_repository import AIVersionRepository
from app.schemas.ai_version import (
    ActiveVersionResponse, AIVersionCreate, AIVersionList, AIVersionRecord,
    AuditEvent, EvaluationSummary, SuiteSummary,
)
from app.services.promotion_gate import SUITES, GatePolicy, GateResult, evaluate_evidence

logger = logging.getLogger(__name__)

TRUSTED_BASELINE = {"approved", "active", "superseded", "rolled_back"}
ACTIVATABLE = {"approved"}
ROLLBACK_TARGETS = {"approved", "superseded", "rolled_back"}
_ACTOR_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,40}$")

ModelLister = Callable[[], Awaitable[Optional[list[str]]]]


def now() -> datetime:
    return datetime.now(timezone.utc)


def clean_actor(label: Optional[str]) -> str:
    """Unauthenticated, operator-supplied label (the admin token identifies the plane, not a person)."""
    return label if label and _ACTOR_RE.match(label) else "admin"


def load_policy(path: str = "") -> GatePolicy:
    if not path:
        return GatePolicy()
    try:
        return GatePolicy.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise AppError(ErrorCode.AI_VERSION_INVALID, f"Gate policy file is unusable: {type(exc).__name__}") from exc


def override_from_manifest(manifest: dict[str, Any]) -> rc.RuntimeOverride:
    cfg = manifest["config"]
    return rc.RuntimeOverride(
        version_id=manifest["version_id"], llm_model=cfg["llm"]["model"], temperature=cfg["llm"]["temperature"],
        num_predict=cfg["llm"]["num_predict"], top_k=cfg["rag"]["top_k"],
        similarity_threshold=cfg["rag"]["similarity_threshold"],
        max_context_chars=cfg["rag"]["max_context_chars"],
        prompt_versions=tuple(sorted(cfg["prompt_versions"].items())),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _default_lister() -> Optional[list[str]]:
    from app.services.ollama_checker import get_available_models

    return await get_available_models()


def _model_present(model: str, names: list[str]) -> bool:
    return model in names or (":" not in model and f"{model}:latest" in names)


class AIVersionService:
    def __init__(self, repo: AIVersionRepository, policy: Optional[GatePolicy] = None,
                 model_lister: Optional[ModelLister] = None) -> None:
        self.repo = repo
        self.policy = policy or GatePolicy()
        self._lister: ModelLister = model_lister or _default_lister
        self._seen_mtime: Optional[int] = None
        self.degraded: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ runtime activation
    def load_active(self) -> None:
        """(Re)apply the persisted active version to the process. Never raises; keeps the current
        configuration and records ``degraded`` if the store is unreadable or inconsistent."""
        with self._lock:
            mtime = self.repo.state_mtime_ns()
            try:
                state = self.repo.read_state()
                active_id = state.get("active_version_id")
                if not active_id:
                    rc.set_active(None)
                else:
                    manifest = self.repo.read_manifest(active_id)
                    if manifest is None:
                        raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, f"active manifest {active_id} is missing")
                    problems = self._static_problems(manifest)
                    if problems:
                        raise AppError(ErrorCode.AI_VERSION_STORE_ERROR, "; ".join(problems))
                    rc.set_active(override_from_manifest(manifest))
                self.degraded = None
            except AppError as exc:
                self.degraded = exc.message
                logger.error("AI version store problem (keeping current configuration): %s", exc.message)
            self._seen_mtime = mtime

    def refresh_if_changed(self) -> None:
        """Cheap per-request check so several workers converge after a promote/rollback."""
        if self.repo.state_mtime_ns() != self._seen_mtime:
            self.load_active()

    # ------------------------------------------------------------------ helpers
    def _manifest(self, version_id: str) -> dict[str, Any]:
        m = self.repo.read_manifest(version_id)
        if m is None:
            raise AppError(ErrorCode.AI_VERSION_NOT_FOUND, f"AI version '{version_id}' was not found.")
        return m

    @staticmethod
    def _vstate(state: dict[str, Any], version_id: str) -> dict[str, Any]:
        vs = state["versions"].get(version_id)
        if vs is None:
            raise AppError(ErrorCode.AI_VERSION_NOT_FOUND, f"AI version '{version_id}' has no state entry.")
        return vs

    @staticmethod
    def _require(vs: dict[str, Any], allowed: set[str], action: str) -> None:
        if vs["status"] not in allowed:
            raise AppError(ErrorCode.AI_VERSION_INVALID_TRANSITION,
                           f"Cannot {action} a version in status '{vs['status']}' (allowed: {sorted(allowed)}).")

    @staticmethod
    def _event(event: str, version_id: Optional[str], actor: str, request_id: Optional[str],
               from_status: Optional[str] = None, to_status: Optional[str] = None,
               reason: Optional[str] = None, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {"ts": now().isoformat(), "event": event, "version_id": version_id, "actor": actor,
                "request_id": request_id, "from_status": from_status, "to_status": to_status,
                "reason": reason, "details": details or {}}

    def _record(self, manifest: dict[str, Any], vs: dict[str, Any]) -> AIVersionRecord:
        return AIVersionRecord(
            version_id=manifest["version_id"], config_fingerprint=manifest["config_fingerprint"],
            status=vs["status"], status_reason=vs.get("status_reason"), description=manifest["description"],
            created_at=manifest["created_at"], created_by=manifest["created_by"], config=manifest["config"],
            prompt_fingerprints=manifest["prompt_fingerprints"], change_summary=manifest["change_summary"],
            previous_version_id=vs.get("previous_active_id") or manifest.get("previous_version_id"),
            evaluation=vs.get("evaluation"), approved_at=vs.get("approved_at"), approved_by=vs.get("approved_by"),
            activated_at=vs.get("activated_at"), updated_at=vs["updated_at"],
        )

    def _static_problems(self, manifest: dict[str, Any]) -> list[str]:
        """Checks that need no network: fingerprint intact, prompts registered & unchanged & lint-clean,
        embedding/chunking equal the live environment. Also serves as the post-activation smoke test."""
        problems: list[str] = []
        cfg = manifest["config"]
        if rc.config_fingerprint(cfg) != manifest["config_fingerprint"]:
            problems.append("manifest fingerprint does not match its configuration "
                            "(manifest edited, or a prompt's content changed under the same version label)")
        for task, version in cfg["prompt_versions"].items():
            if not registry.is_registered(task, version):
                problems.append(f"prompt {task}/{version} is not registered in this build")
                continue
            if registry.content_fingerprint(task, version) != manifest["prompt_fingerprints"].get(task):
                problems.append(f"prompt {task}/{version} content differs from the approved fingerprint")
            problems += [f"prompt {task}/{version}: {p}"
                         for p in registry.lint_prompt_texts(registry.get_spec(task, version).system_texts)]
        s = get_settings()
        if cfg["embedding_model"] != s.ollama_embedding_model:
            problems.append("embedding model differs from the running environment; switching embeddings needs a "
                            "re-embedding procedure and is not performed by a version promotion")
        if (cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"]) != (s.chunk_size, s.chunk_overlap):
            problems.append("chunking differs from the running environment; existing vectors were created with "
                            "other chunking and would be inconsistent")
        eff = rc.build_effective(override_from_manifest(manifest)).config_dict()
        for key in ("llm", "rag", "prompt_versions"):
            if eff[key] != cfg[key]:
                problems.append(f"effective {key} does not match the manifest")
        return problems

    async def _preflight(self, manifest: dict[str, Any]) -> list[str]:
        problems = self._static_problems(manifest)
        names = await self._lister()
        if names is None:
            problems.append("Ollama is unreachable, so the model cannot be verified (infrastructure; retry later)")
        else:
            cfg = manifest["config"]
            for label, model in (("LLM", cfg["llm"]["model"]), ("embedding", cfg["embedding_model"])):
                if not _model_present(model, names):
                    problems.append(f"{label} model '{model}' is not installed in Ollama")
        return problems

    def _evidence_hashes(self, version_id: str) -> dict[str, str]:
        d = self.repo.evidence_dir(version_id)
        names = [f"{s}.json" for s in SUITES] + ["functional_tests.json"]
        return {n: _sha256(d / n) for n in names if (d / n).is_file()}

    def _read_evidence(self, version_id: str) -> tuple[dict[str, dict], Optional[dict], list[str]]:
        d = self.repo.evidence_dir(version_id)
        reports: dict[str, dict] = {}
        errors: list[str] = []
        functional = None

        def load(path: Path) -> Optional[dict]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
            except (OSError, ValueError):
                errors.append(f"evidence file {path.name} is unreadable")
                return None

        for s in SUITES:
            if (d / f"{s}.json").is_file():
                data = load(d / f"{s}.json")
                if data is not None:
                    reports[s] = data
        if (d / "functional_tests.json").is_file():
            functional = load(d / "functional_tests.json")
        status = d / "evidence_status.json"
        if status.is_file():
            data = load(status)
            if data:
                errors += [str(e) for e in data.get("infrastructure_errors", [])]
        return reports, functional, errors

    # ------------------------------------------------------------------ create / read
    def create(self, body: AIVersionCreate, actor: str, request_id: Optional[str]) -> AIVersionRecord:
        cfg = body.config
        cfg_dict = cfg.model_dump()
        for task, version in cfg.prompt_versions.items():
            if not registry.is_registered(task, version):
                raise AppError(ErrorCode.AI_VERSION_INVALID,
                               f"prompt {task}/{version} is not registered; available: "
                               f"{registry.available_versions(task)}")
        s = get_settings()
        if cfg.embedding_model != s.ollama_embedding_model:
            raise AppError(ErrorCode.AI_VERSION_INVALID,
                           "embedding_model must equal the running environment. Changing the embedding model makes "
                           "existing ChromaDB vectors incompatible; it requires re-embedding into a NEW collection "
                           "and is outside runtime version promotion (see docs/AI_VERSION_GOVERNANCE.md).")
        if (cfg.chunking.chunk_size, cfg.chunking.chunk_overlap) != (s.chunk_size, s.chunk_overlap):
            raise AppError(ErrorCode.AI_VERSION_INVALID,
                           "chunking must equal the running environment; changing chunking requires re-ingesting "
                           "documents and is outside runtime version promotion.")
        fp = rc.config_fingerprint(cfg_dict)
        reference = rc.current().config_dict()
        version_id = f"v{now():%Y%m%d%H%M%S}-{fp[:8]}"
        with self.repo.transaction() as tx:
            for existing_id in self.repo.manifest_ids():
                m = self.repo.read_manifest(existing_id)
                if m and m.get("config_fingerprint") == fp:
                    raise AppError(ErrorCode.AI_VERSION_DUPLICATE,
                                   f"An identical configuration already exists as '{existing_id}'.")
            manifest = {
                "schema_version": 1, "version_id": version_id, "config_fingerprint": fp,
                "description": body.description, "created_at": now().isoformat(), "created_by": actor,
                "config": cfg_dict,
                "prompt_fingerprints": {t: registry.content_fingerprint(t, v) for t, v in cfg.prompt_versions.items()},
                "change_summary": self._change_summary(reference, cfg_dict),
                "previous_version_id": tx.state.get("active_version_id"),
                "reproducibility_note": ("Configuration is reproducible; exact token output is not guaranteed. "
                                         "Ollama tags are mutable and models may be removed."),
            }
            self.repo.write_manifest(manifest)
            vs = {"status": "draft", "status_reason": None, "updated_at": now().isoformat(), "evaluation": None}
            tx.state["versions"][version_id] = vs
            tx.audit(self._event("create", version_id, actor, request_id, None, "draft", body.description,
                                 {"config_fingerprint": fp, "change_summary": manifest["change_summary"]}))
        return self._record(manifest, vs)

    @staticmethod
    def _change_summary(ref: dict[str, Any], new: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for task, version in sorted(new["prompt_versions"].items()):
            old = ref["prompt_versions"].get(task)
            if old != version:
                out.append(f"prompt:{task} {old}->{version}")
        if ref["llm"]["model"] != new["llm"]["model"]:
            out.append(f"llm_model {ref['llm']['model']}->{new['llm']['model']}")
        if (ref["llm"]["temperature"], ref["llm"]["num_predict"]) != (new["llm"]["temperature"], new["llm"]["num_predict"]):
            out.append("generation_params")
        if ref["rag"] != new["rag"]:
            out.append("retrieval_params")
        if ref["embedding_model"] != new["embedding_model"]:
            out.append("embedding_model (re-embedding required)")
        if ref["chunking"] != new["chunking"]:
            out.append("chunking (re-ingestion required)")
        return out or ["no change vs the configuration in effect at creation"]

    def get(self, version_id: str) -> AIVersionRecord:
        manifest = self._manifest(version_id)
        return self._record(manifest, self._vstate(self.repo.read_state(), version_id))

    def list(self) -> AIVersionList:
        state = self.repo.read_state()
        records = []
        for vid in state["versions"]:
            m = self.repo.read_manifest(vid)
            if m is not None:
                records.append(self._record(m, state["versions"][vid]))
        records.sort(key=lambda r: r.created_at)
        return AIVersionList(active_version_id=state.get("active_version_id"), revision=state["revision"],
                             versions=records)

    def audit(self, limit: int = 100, version_id: Optional[str] = None) -> list[AuditEvent]:
        return [AuditEvent(**e) for e in self.repo.read_audit(limit, version_id)]

    def active(self) -> ActiveVersionResponse:
        state = self.repo.read_state()
        active_id = state.get("active_version_id")
        eff = rc.build_effective(rc.get_active_override())
        record = None
        drift: list[str] = []
        if active_id:
            m = self.repo.read_manifest(active_id)
            if m is not None:
                record = self._record(m, state["versions"][active_id])
                drift = self._static_problems(m)
        if eff.version_id != (active_id or rc.BOOTSTRAP_VERSION_ID):
            drift.append("this process has not yet applied the persisted active version (refreshes on next request)")
        return ActiveVersionResponse(
            active_version_id=active_id, source=eff.source, effective_config=eff.config_dict(),
            effective_fingerprint=eff.fingerprint, record=record, drift=drift, store_degraded=self.degraded,
            note="No version is active: the environment (.env) configuration applies." if not active_id else "")

    # ------------------------------------------------------------------ evaluate / approve / reject
    def evaluate(self, version_id: str, baseline_version_id: Optional[str], actor: str,
                 request_id: Optional[str]) -> AIVersionRecord:
        manifest = self._manifest(version_id)
        with self.repo.transaction() as tx:
            vs = self._vstate(tx.state, version_id)
            self._require(vs, {"draft", "inconclusive", "validated"}, "evaluate")
            base_id = baseline_version_id or tx.state.get("active_version_id")
            if base_id == version_id:
                base_id = None
            baseline_reports = None
            notes: list[str] = []
            if base_id:
                self._manifest(base_id)
                bstatus = self._vstate(tx.state, base_id)["status"]
                if bstatus in TRUSTED_BASELINE:
                    baseline_reports = self._read_evidence(base_id)[0] or None
                    if baseline_reports is None:
                        notes.append(f"baseline {base_id} has no stored evidence")
                elif baseline_version_id:
                    raise AppError(ErrorCode.AI_VERSION_INVALID,
                                   f"Baseline '{base_id}' has status '{bstatus}' and is not trusted as a baseline.")
            reports, functional, infra = self._read_evidence(version_id)
            result: GateResult = evaluate_evidence(manifest["config"], reports, functional, baseline_reports,
                                                   self.policy, infra)
            result.reasons += notes
            summary = EvaluationSummary(
                verdict=result.verdict, evaluated_at=now(), reasons=result.reasons,
                suites={k: SuiteSummary(**v) for k, v in result.suites.items()},
                baseline_version_id=base_id if result.baseline_used else None,
                policy_fingerprint=self.policy.fingerprint(), evidence_sha256=self._evidence_hashes(version_id),
                functional_tests=functional)
            new_status = ("validated" if result.verdict == "PASS"
                          else "inconclusive" if result.verdict in ("INCONCLUSIVE", "BLOCKED") else "rejected")
            old_status = vs["status"]
            vs.update(status=new_status, status_reason="; ".join(result.reasons)[:500],
                      evaluation=json.loads(summary.model_dump_json()), updated_at=now().isoformat())
            tx.audit(self._event("evaluate", version_id, actor, request_id, old_status, new_status,
                                 vs["status_reason"], {"verdict": result.verdict,
                                                       "policy_fingerprint": summary.policy_fingerprint,
                                                       "baseline": summary.baseline_version_id}))
        logger.info("AI version %s evaluated: verdict=%s status=%s", version_id, result.verdict, new_status)
        return self._record(manifest, vs)

    def approve(self, version_id: str, reason: str, actor: str, request_id: Optional[str]) -> AIVersionRecord:
        manifest = self._manifest(version_id)
        with self.repo.transaction() as tx:
            vs = self._vstate(tx.state, version_id)
            self._require(vs, {"validated"}, "approve")
            ev = vs.get("evaluation") or {}
            if ev.get("verdict") != "PASS":
                raise AppError(ErrorCode.AI_VERSION_GATE_BLOCKED, "No stored PASS verdict; run evaluate first.")
            if self._evidence_hashes(version_id) != ev.get("evidence_sha256"):
                raise AppError(ErrorCode.AI_VERSION_GATE_BLOCKED,
                               "Evidence files changed after evaluation; re-run evaluate.")
            old = vs["status"]
            vs.update(status="approved", status_reason=reason, approved_at=now().isoformat(),
                      approved_by=actor, updated_at=now().isoformat())
            tx.audit(self._event("approve", version_id, actor, request_id, old, "approved", reason))
        return self._record(manifest, vs)

    def reject(self, version_id: str, reason: str, actor: str, request_id: Optional[str]) -> AIVersionRecord:
        manifest = self._manifest(version_id)
        with self.repo.transaction() as tx:
            vs = self._vstate(tx.state, version_id)
            self._require(vs, {"draft", "inconclusive", "validated", "approved"}, "reject")
            old = vs["status"]
            vs.update(status="rejected", status_reason=reason, updated_at=now().isoformat())
            tx.audit(self._event("reject", version_id, actor, request_id, old, "rejected", reason))
        return self._record(manifest, vs)

    # ------------------------------------------------------------------ promote / rollback
    async def promote(self, version_id: str, reason: str, actor: str, request_id: Optional[str]) -> AIVersionRecord:
        return await self._activate(version_id, reason, actor, request_id, ACTIVATABLE, "promote")

    async def rollback(self, target_version_id: str, reason: str, actor: str,
                       request_id: Optional[str]) -> AIVersionRecord | ActiveVersionResponse:
        if target_version_id == rc.BOOTSTRAP_VERSION_ID:
            return self._rollback_to_env(reason, actor, request_id)
        return await self._activate(target_version_id, reason, actor, request_id, ROLLBACK_TARGETS, "rollback")

    async def _activate(self, version_id: str, reason: str, actor: str, request_id: Optional[str],
                        allowed: set[str], action: str) -> AIVersionRecord:
        manifest = self._manifest(version_id)
        snapshot = self.repo.read_state()
        revision = snapshot["revision"]
        vs0 = self._vstate(snapshot, version_id)
        self._require(vs0, allowed, action)
        if snapshot.get("active_version_id") == version_id:
            raise AppError(ErrorCode.AI_VERSION_INVALID_TRANSITION, f"Version '{version_id}' is already active.")
        ev = vs0.get("evaluation") or {}
        if ev.get("verdict") != "PASS" or self._evidence_hashes(version_id) != ev.get("evidence_sha256"):
            self._fail(action, version_id, actor, request_id, ["no valid PASS evidence on record for this version"])
        problems = await self._preflight(manifest)
        if problems:
            self._fail(action, version_id, actor, request_id, problems)
        with self.repo.transaction(expected_revision=revision) as tx:
            vs = self._vstate(tx.state, version_id)
            self._require(vs, allowed, action)
            previous = tx.state.get("active_version_id")
            ts = now().isoformat()
            if previous:
                pv = self._vstate(tx.state, previous)
                pv.update(status="superseded" if action == "promote" else "rolled_back",
                          status_reason=f"replaced by {version_id}: {reason}"[:500], updated_at=ts)
            old = vs["status"]
            vs.update(status="active", status_reason=reason, activated_at=ts, updated_at=ts, previous_active_id=previous)
            tx.state["active_version_id"] = version_id
            smoke = self._static_problems(manifest)  # before commit: a failure aborts with nothing persisted
            if smoke:
                raise AppError(ErrorCode.AI_VERSION_PREFLIGHT_FAILED, "Smoke test failed: " + "; ".join(smoke))
            tx.audit(self._event(action, version_id, actor, request_id, old, "active", reason,
                                 {"previous_active": previous, "config_fingerprint": manifest["config_fingerprint"],
                                  "smoke_test": "static checks passed (no live LLM call)"}))
        rc.set_active(override_from_manifest(manifest))
        self._seen_mtime = self.repo.state_mtime_ns()
        logger.info("AI version %s is now active (%s; previous=%s)", version_id, action, previous)
        return self._record(manifest, vs)

    def _rollback_to_env(self, reason: str, actor: str, request_id: Optional[str]) -> ActiveVersionResponse:
        snapshot = self.repo.read_state()
        if not snapshot.get("active_version_id"):
            raise AppError(ErrorCode.AI_VERSION_INVALID_TRANSITION, "No version is active; nothing to roll back.")
        with self.repo.transaction(expected_revision=snapshot["revision"]) as tx:
            previous = tx.state["active_version_id"]
            self._vstate(tx.state, previous).update(
                status="rolled_back", status_reason=f"rolled back to environment configuration: {reason}"[:500],
                updated_at=now().isoformat())
            tx.state["active_version_id"] = None
            tx.audit(self._event("rollback", previous, actor, request_id, "active", "rolled_back", reason,
                                 {"target": rc.BOOTSTRAP_VERSION_ID}))
        rc.set_active(None)
        self._seen_mtime = self.repo.state_mtime_ns()
        return self.active()

    def _fail(self, action: str, version_id: str, actor: str, request_id: Optional[str], problems: list[str]) -> None:
        """Record the refusal (nothing else changes) and raise. Never reports success."""
        try:
            self.repo.append_audit(self._event(f"{action}_blocked", version_id, actor, request_id,
                                               details={"problems": problems}))
        except AppError:
            logger.exception("could not audit blocked %s", action)
        raise AppError(ErrorCode.AI_VERSION_PREFLIGHT_FAILED,
                       f"{action} refused; active configuration unchanged. " + "; ".join(problems))


_service: Optional[AIVersionService] = None
_service_dir: Optional[str] = None


def get_ai_version_service() -> AIVersionService:
    global _service, _service_dir
    s = get_settings()
    if _service is None or _service_dir != s.ai_versions_dir:
        _service = AIVersionService(AIVersionRepository(s.ai_versions_dir), load_policy(s.ai_gate_policy_file))
        _service_dir = s.ai_versions_dir
    return _service


def reset_ai_version_service() -> None:
    global _service, _service_dir
    _service = None
    _service_dir = None
    rc.set_active(None)
    rc.set_refresh_hook(None)


def init_ai_versions() -> None:
    """Startup hook: apply the persisted active version and keep workers in sync."""
    svc = get_ai_version_service()
    svc.load_active()
    rc.set_refresh_hook(svc.refresh_if_changed)
