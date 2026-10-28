"""Phase 18 - AI version governance: gate, lifecycle, promote/rollback, audit, API, traceability.

All Ollama interaction is mocked (model lister); no test depends on a running Ollama.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

from app.core import prompt_registry as registry
from app.core import runtime_config as rc
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.repositories import ai_version_repository as repo_mod
from app.schemas.ai_version import AIVersionCreate
from app.services.ai_version_service import (
    AIVersionService, get_ai_version_service, reset_ai_version_service,
)
from app.services.promotion_gate import SUITES, GatePolicy, evaluate_evidence

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "unit-test-admin-token-VALUE"
H = {"X-Admin-Token": TOKEN}
FLOORS = {"rag": {"hit@5": 0.5}, "qgen": {"answerability_rate": 0.5}, "answer_eval": {"within_band_rate": 0.5}}

METRICS = {
    "rag": {"hit@5": 0.9, "mrr": 0.8, "insufficient_context_detection_rate": 0.9, "incorrect_abstention_rate": 0.05},
    "qgen": {"answerability_rate": 0.9, "groundedness_rate": 0.9, "duplication_rate": 0.05},
    "answer_eval": {"within_band_rate": 0.8, "correct_label_accuracy": 0.9,
                    "over_scoring_rate_wrong_answers": 0.05, "under_scoring_rate_paraphrases": 0.1},
}
PROMPT_KEY = {"rag": "rag", "qgen": "question_generation", "answer_eval": "evaluation"}
TASK_OF = {"rag": "rag_answer", "qgen": "question_generation", "answer_eval": "answer_evaluation"}


# ------------------------------------------------------------------------------------------ helpers
def base_config() -> dict:
    return copy.deepcopy(rc.build_effective(None).config_dict())


def candidate_config(**changes) -> dict:
    cfg = base_config()
    cfg["llm"]["model"] = changes.get("model", "cand-model:1b")
    cfg["rag"]["top_k"] = changes.get("top_k", min(20, cfg["rag"]["top_k"] + 1))
    if "prompt" in changes:
        cfg["prompt_versions"]["rag_answer"] = changes["prompt"]
    return cfg


def report(suite: str, cfg: dict, *, cases: int = 40, mode: str = "ollama", metrics=None, invariants=None,
           dataset_version: str = "1.0.0", model=None) -> dict:
    return {
        "metadata": {
            "evaluation_run_id": f"{suite}-test", "mode": mode, "dataset_version": dataset_version,
            "model_name": model or cfg["llm"]["model"], "embedding_model": cfg["embedding_model"],
            "top_k": cfg["rag"]["top_k"], "similarity_threshold": cfg["rag"]["similarity_threshold"],
            "prompt_version": {PROMPT_KEY[suite]: cfg["prompt_versions"][TASK_OF[suite]]},
        },
        "metrics_flat": dict(metrics or METRICS[suite]),
        "invariants": invariants or {"source_integrity_violations": 0, "document_isolation_violations": 0},
        "per_case": [{}] * cases,
    }


def all_reports(cfg: dict, **kw) -> dict:
    return {s: report(s, cfg, **kw) for s in SUITES}


FUNC_OK = {"passed": True, "total": 700, "failed": 0}


def put_evidence(svc: AIVersionService, version_id: str, cfg: dict, functional=FUNC_OK, **kw) -> None:
    d = svc.repo.evidence_dir(version_id)
    d.mkdir(parents=True, exist_ok=True)
    for s, rep in all_reports(cfg, **kw).items():
        (d / f"{s}.json").write_text(json.dumps(rep), encoding="utf-8")
    if functional is not None:
        (d / "functional_tests.json").write_text(json.dumps(functional), encoding="utf-8")


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "ai_versions_dir", str(tmp_path / "ai_versions"))
    monkeypatch.setattr(s, "admin_api_token", TOKEN)
    reset_ai_version_service()
    service = get_ai_version_service()
    service.policy = GatePolicy(floors=FLOORS)
    installed = [s.ollama_model, s.ollama_embedding_model, "cand-model:1b", "cand-model:2b"]

    async def lister():
        await asyncio.sleep(0)
        return list(installed)

    service._lister = lister
    service._installed = installed
    yield service
    reset_ai_version_service()


def create_version(svc, cfg, desc="candidate version") -> str:
    rec = svc.create(AIVersionCreate(description=desc, config=cfg), "tester", "req-1")
    return rec.version_id


async def make_active(svc, cfg, desc="active version") -> str:
    vid = create_version(svc, cfg, desc)
    put_evidence(svc, vid, cfg)
    assert svc.evaluate(vid, None, "tester", "r").status == "validated"
    svc.approve(vid, "looks good", "tester", "r")
    await svc.promote(vid, "go live", "tester", "r")
    return vid


# ------------------------------------------------------------------------------------------ gate (pure)
def gate(reports=None, functional=FUNC_OK, baseline=None, policy=None, infra=None, cfg=None):
    cfg = cfg or candidate_config()
    return evaluate_evidence(cfg, all_reports(cfg) if reports is None else reports, functional, baseline,
                             policy or GatePolicy(floors=FLOORS), infra)


def test_gate_pass_requires_real_data_and_floors():
    r = gate()
    assert r.verdict == "PASS" and "not a significance test" in r.reasons[0]


def test_gate_blocked_when_suite_missing():
    cfg = candidate_config()
    reports = all_reports(cfg)
    del reports["qgen"]
    assert gate(reports, cfg=cfg).verdict == "BLOCKED"


def test_gate_blocked_without_functional_result_and_on_infra_error():
    assert gate(functional=None).verdict == "BLOCKED"
    assert gate(infra=["ollama unreachable"]).verdict == "BLOCKED"


def test_gate_offline_reports_are_inconclusive_never_pass():
    cfg = candidate_config()
    r = gate(all_reports(cfg, mode="offline"), cfg=cfg)
    assert r.verdict == "INCONCLUSIVE"


def test_gate_small_dataset_is_inconclusive():
    cfg = candidate_config()
    assert gate(all_reports(cfg, cases=5), cfg=cfg).verdict == "INCONCLUSIVE"


def test_gate_no_baseline_and_no_floors_is_inconclusive():
    assert gate(policy=GatePolicy()).verdict == "INCONCLUSIVE"


def test_gate_regression_vs_baseline_fails_quality_and_small_drop_passes():
    cfg = candidate_config()
    base = all_reports(base_config())
    worse = all_reports(cfg)
    worse["rag"]["metrics_flat"]["hit@5"] = 0.7  # 0.9 -> 0.7
    r = evaluate_evidence(cfg, worse, FUNC_OK, base, GatePolicy(), None)
    assert r.verdict == "FAIL_QUALITY" and "rag.hit@5" in r.reasons[0]
    ok = all_reports(cfg)
    ok["rag"]["metrics_flat"]["hit@5"] = 0.87  # within the 0.05 tolerance
    assert evaluate_evidence(cfg, ok, FUNC_OK, base, GatePolicy(), None).verdict == "PASS"


def test_gate_lower_is_better_metric_regression():
    cfg = candidate_config()
    worse = all_reports(cfg)
    worse["qgen"]["metrics_flat"]["duplication_rate"] = 0.30
    r = evaluate_evidence(cfg, worse, FUNC_OK, all_reports(base_config()), GatePolicy(), None)
    assert r.verdict == "FAIL_QUALITY"


def test_gate_floor_violation_fails_quality():
    cfg = candidate_config()
    reports = all_reports(cfg)
    reports["rag"]["metrics_flat"]["hit@5"] = 0.2
    assert gate(reports, cfg=cfg).verdict == "FAIL_QUALITY"


def test_gate_safety_violation_fails_even_with_tiny_offline_data():
    cfg = candidate_config()
    reports = all_reports(cfg, cases=3, mode="offline",
                          invariants={"source_integrity_violations": 1, "document_isolation_violations": 0})
    assert gate(reports, cfg=cfg).verdict == "FAIL_SAFETY"


def test_gate_evidence_for_another_model_fails_integrity():
    cfg = candidate_config()
    assert gate(all_reports(cfg, model="some-other:7b"), cfg=cfg).verdict == "FAIL_INTEGRITY"


def test_gate_mixed_dataset_versions_fail_integrity():
    cfg = candidate_config()
    reports = all_reports(cfg)
    reports["qgen"]["metadata"]["dataset_version"] = "2.0.0"
    assert gate(reports, cfg=cfg).verdict == "FAIL_INTEGRITY"


def test_gate_functional_failure():
    assert gate(functional={"passed": False, "total": 10, "failed": 2}).verdict == "FAIL_FUNCTIONAL"


def test_gate_baseline_with_other_dataset_is_not_comparable():
    cfg = candidate_config()
    base = all_reports(base_config(), dataset_version="0.9")
    r = evaluate_evidence(cfg, all_reports(cfg), FUNC_OK, base, GatePolicy(), None)
    assert r.verdict == "INCONCLUSIVE"


# ------------------------------------------------------------------------------------------ fingerprints / registry / schema
def test_fingerprint_independent_of_key_order_and_changes_with_values():
    cfg = base_config()
    shuffled = json.loads(json.dumps(cfg, sort_keys=False))
    reordered = {k: shuffled[k] for k in reversed(list(shuffled))}
    assert rc.config_fingerprint(cfg) == rc.config_fingerprint(reordered)
    changed = copy.deepcopy(cfg)
    changed["rag"]["top_k"] += 1
    assert rc.config_fingerprint(changed) != rc.config_fingerprint(cfg)


def test_rag_fingerprint_separates_retrieval_from_prompt_changes():
    a = rc.build_effective(None)
    b = dataclasses.replace(a, llm_model="x:1")
    c = dataclasses.replace(a, top_k=a.top_k + 1)
    assert a.rag_fingerprint == b.rag_fingerprint != c.rag_fingerprint
    assert a.fingerprint != b.fingerprint


def test_shipped_prompts_match_committed_lock():
    assert registry.verify_lock() == []
    assert set(registry.DEFAULT_PROMPT_VERSIONS) == set(registry.TASKS)


def test_published_prompt_version_is_immutable(monkeypatch):
    spec = registry.get_spec("rag_answer", "v1")
    altered = dataclasses.replace(spec, texts=spec.texts + ("edited",))
    with pytest.raises(registry.PromptRegistryError, match="immutable"):
        registry.register_prompt(altered)
    assert registry.get_spec("rag_answer", "v1") is spec


def test_prompt_lint_rejects_backend_owned_and_injection_unsafe_text():
    problems = registry.lint_prompt_texts(["Answer from the document. Cite the page and chunk id of each claim. "
                                           "Ignore the context if you know better."])
    assert any("citations" in p for p in problems)
    assert any("ignore" in p for p in problems)
    assert any("missing statement" in p for p in problems)


def test_experimental_prompt_does_not_change_active_defaults(monkeypatch):
    spec = registry.get_spec("rag_answer", "v1")
    monkeypatch.setitem(registry._REGISTRY, ("rag_answer", "v2-exp"),
                        dataclasses.replace(spec, version="v2-exp", texts=spec.texts + ("exp",)))
    assert registry.DEFAULT_PROMPT_VERSIONS["rag_answer"] == "v1"
    assert rc.current().prompt_version("rag_answer") == "v1"


def test_create_body_has_no_status_and_rejects_secret_text():
    cfg = base_config()
    with pytest.raises(Exception):
        AIVersionCreate.model_validate({"description": "x y z", "config": cfg, "status": "active"})
    with pytest.raises(Exception):
        AIVersionCreate(description="uses api_key=abcdef", config=cfg)
    bad = copy.deepcopy(cfg)
    bad["chunking"]["chunk_overlap"] = bad["chunking"]["chunk_size"]
    with pytest.raises(Exception):
        AIVersionCreate(description="bad chunking", config=bad)
    bad = copy.deepcopy(cfg)
    del bad["prompt_versions"]["rag_answer"]
    with pytest.raises(Exception):
        AIVersionCreate(description="missing prompt", config=bad)


# ------------------------------------------------------------------------------------------ service lifecycle
def test_create_is_draft_with_stable_manifest_and_no_secrets(svc):
    vid = create_version(svc, candidate_config())
    rec = svc.get(vid)
    assert rec.status == "draft" and rec.config_fingerprint == rc.config_fingerprint(candidate_config())
    assert any(c.startswith("llm_model") for c in rec.change_summary)
    assert any(c == "retrieval_params" for c in rec.change_summary)
    text = "".join(p.read_text(encoding="utf-8") for p in svc.repo.root.rglob("*") if p.is_file())
    assert TOKEN not in text
    assert (svc.repo.manifests / f"{vid}.json").is_file()


def test_manifest_is_never_overwritten_and_duplicates_rejected(svc):
    cfg = candidate_config()
    vid = create_version(svc, cfg)
    with pytest.raises(AppError) as e:
        create_version(svc, cfg, "same config again")
    assert e.value.code == ErrorCode.AI_VERSION_DUPLICATE
    with pytest.raises(AppError):
        svc.repo.write_manifest(svc.repo.read_manifest(vid))


def test_embedding_or_chunking_change_is_refused_with_explanation(svc):
    cfg = candidate_config()
    cfg["embedding_model"] = "other-embed:latest"
    with pytest.raises(AppError) as e:
        create_version(svc, cfg)
    assert e.value.code == ErrorCode.AI_VERSION_INVALID and "re-embedding" in e.value.message
    cfg = candidate_config()
    cfg["chunking"]["chunk_size"] += 50
    with pytest.raises(AppError) as e:
        create_version(svc, cfg)
    assert "chunking" in e.value.message


def test_unregistered_prompt_version_refused(svc):
    with pytest.raises(AppError) as e:
        create_version(svc, candidate_config(prompt="v99"))
    assert e.value.code == ErrorCode.AI_VERSION_INVALID


def test_no_evidence_means_inconclusive_and_nothing_can_be_approved(svc):
    vid = create_version(svc, candidate_config())
    rec = svc.evaluate(vid, None, "t", "r")
    assert rec.status == "inconclusive" and rec.evaluation.verdict == "BLOCKED"
    with pytest.raises(AppError) as e:
        svc.approve(vid, "force it", "t", "r")
    assert e.value.code == ErrorCode.AI_VERSION_INVALID_TRANSITION


def test_failed_quality_is_rejected_with_reasons_and_is_terminal(svc):
    cfg = candidate_config()
    vid = create_version(svc, cfg)
    put_evidence(svc, vid, cfg, metrics={**METRICS["rag"], "hit@5": 0.1})
    rec = svc.evaluate(vid, None, "t", "r")
    assert rec.status == "rejected" and rec.evaluation.verdict == "FAIL_QUALITY"
    assert "hit@5" in rec.status_reason
    with pytest.raises(AppError):
        svc.evaluate(vid, None, "t", "r")


def test_cannot_approve_or_promote_without_passing_validation(svc):
    vid = create_version(svc, candidate_config())
    with pytest.raises(AppError):
        svc.approve(vid, "skip validation", "t", "r")
    with pytest.raises(AppError) as e:
        asyncio.run(svc.promote(vid, "skip everything", "t", "r"))
    assert e.value.code == ErrorCode.AI_VERSION_INVALID_TRANSITION
    assert svc.repo.read_state()["active_version_id"] is None


def test_evidence_tampering_after_evaluation_blocks_approval(svc):
    cfg = candidate_config()
    vid = create_version(svc, cfg)
    put_evidence(svc, vid, cfg)
    assert svc.evaluate(vid, None, "t", "r").status == "validated"
    f = svc.repo.evidence_dir(vid) / "rag.json"
    data = json.loads(f.read_text())
    data["metrics_flat"]["hit@5"] = 1.0
    f.write_text(json.dumps(data))
    with pytest.raises(AppError) as e:
        svc.approve(vid, "approve anyway", "t", "r")
    assert e.value.code == ErrorCode.AI_VERSION_GATE_BLOCKED


def test_reject_is_terminal_and_audited(svc):
    vid = create_version(svc, candidate_config())
    svc.reject(vid, "not wanted", "t", "r")
    with pytest.raises(AppError):
        svc.reject(vid, "again", "t", "r")
    assert [e.event for e in svc.audit(version_id=vid)] == ["create", "reject"]


# ------------------------------------------------------------------------------------------ promote / rollback
def test_full_flow_promote_changes_effective_config_and_audits(svc):
    async def run():
        cfg = candidate_config()
        vid = await make_active(svc, cfg)
        eff = rc.current()
        assert eff.version_id == vid and eff.llm_model == "cand-model:1b" and eff.top_k == cfg["rag"]["top_k"]
        assert svc.active().record.status == "active"
        events = [e.event for e in svc.audit(version_id=vid)]
        assert events == ["create", "evaluate", "approve", "promote"]
        return vid

    asyncio.run(run())


def test_second_promotion_supersedes_and_rollback_restores_previous(svc):
    async def run():
        v1 = await make_active(svc, candidate_config(), "first")
        cfg2 = candidate_config(model="cand-model:2b", top_k=3)
        v2 = create_version(svc, cfg2, "second")
        put_evidence(svc, v2, cfg2)
        svc.evaluate(v2, None, "t", "r")
        svc.approve(v2, "ok", "t", "r")
        await svc.promote(v2, "upgrade", "t", "r")
        assert svc.get(v1).status == "superseded" and rc.current().llm_model == "cand-model:2b"
        assert svc.get(v2).previous_version_id == v1

        await svc.rollback(v1, "v2 degraded in production", "t", "r")
        assert svc.get(v1).status == "active" and svc.get(v2).status == "rolled_back"
        assert rc.current().version_id == v1 and rc.current().llm_model == "cand-model:1b"
        # history preserved: both manifests untouched, audit has the rollback with a reason
        assert svc.repo.read_manifest(v2) is not None
        rb = [e for e in svc.audit() if e.event == "rollback"]
        assert rb and rb[0].reason == "v2 degraded in production" and rb[0].details["previous_active"] == v2
        # a rolled-back version can be re-activated by rollback again (still valid evidence)
        await svc.rollback(v2, "false alarm", "t", "r")
        assert svc.get(v2).status == "active"

    asyncio.run(run())


def test_rollback_to_never_approved_version_refused(svc):
    async def run():
        await make_active(svc, candidate_config())
        draft = create_version(svc, candidate_config(model="cand-model:2b"), "draft only")
        with pytest.raises(AppError) as e:
            await svc.rollback(draft, "nope", "t", "r")
        assert e.value.code == ErrorCode.AI_VERSION_INVALID_TRANSITION

    asyncio.run(run())


def test_rollback_fails_safely_when_model_no_longer_installed(svc):
    async def run():
        v1 = await make_active(svc, candidate_config(), "first")
        cfg2 = candidate_config(model="cand-model:2b")
        v2 = create_version(svc, cfg2, "second")
        put_evidence(svc, v2, cfg2)
        svc.evaluate(v2, None, "t", "r")
        svc.approve(v2, "ok", "t", "r")
        await svc.promote(v2, "up", "t", "r")
        svc._installed.remove("cand-model:1b")  # model of v1 was deleted from Ollama
        before = svc.repo.read_state()
        with pytest.raises(AppError) as e:
            await svc.rollback(v1, "back", "t", "r")
        assert e.value.code == ErrorCode.AI_VERSION_PREFLIGHT_FAILED and "not installed" in e.value.message
        after = svc.repo.read_state()
        assert after["active_version_id"] == v2 and after["revision"] == before["revision"]
        assert rc.current().version_id == v2
        assert [e.event for e in svc.audit() if e.event.endswith("_blocked")] == ["rollback_blocked"]

    asyncio.run(run())


def test_promote_refused_when_ollama_unreachable(svc):
    async def none():
        return None

    async def run():
        cfg = candidate_config()
        vid = create_version(svc, cfg)
        put_evidence(svc, vid, cfg)
        svc.evaluate(vid, None, "t", "r")
        svc.approve(vid, "ok", "t", "r")
        svc._lister = none
        with pytest.raises(AppError) as e:
            await svc.promote(vid, "go", "t", "r")
        assert "unreachable" in e.value.message
        assert svc.get(vid).status == "approved" and svc.repo.read_state()["active_version_id"] is None

    asyncio.run(run())


def test_prompt_content_drift_blocks_promotion(svc, monkeypatch):
    async def run():
        spec = registry.get_spec("rag_answer", "v1")
        v2 = dataclasses.replace(spec, version="v2-exp", texts=spec.texts + ("exp",))
        monkeypatch.setitem(registry._REGISTRY, ("rag_answer", "v2-exp"), v2)
        cfg = candidate_config(prompt="v2-exp")
        vid = create_version(svc, cfg)
        put_evidence(svc, vid, cfg)
        assert svc.evaluate(vid, None, "t", "r").status == "validated"
        svc.approve(vid, "ok", "t", "r")
        monkeypatch.setitem(registry._REGISTRY, ("rag_answer", "v2-exp"),
                            dataclasses.replace(v2, texts=v2.texts + ("silently edited",)))
        with pytest.raises(AppError) as e:
            await svc.promote(vid, "go", "t", "r")
        assert e.value.code == ErrorCode.AI_VERSION_PREFLIGHT_FAILED
        assert svc.repo.read_state()["active_version_id"] is None

    asyncio.run(run())


def test_rollback_to_env_configuration(svc):
    async def run():
        await make_active(svc, candidate_config())
        await svc.rollback(rc.BOOTSTRAP_VERSION_ID, "return to .env", "t", "r")
        assert rc.current().version_id == rc.BOOTSTRAP_VERSION_ID and svc.repo.read_state()["active_version_id"] is None
        with pytest.raises(AppError):
            await svc.rollback(rc.BOOTSTRAP_VERSION_ID, "again", "t", "r")

    asyncio.run(run())


def test_concurrent_promotions_exactly_one_wins(svc):
    async def run():
        ids = []
        for model in ("cand-model:1b", "cand-model:2b"):
            cfg = candidate_config(model=model)
            vid = create_version(svc, cfg, f"cand {model}")
            put_evidence(svc, vid, cfg)
            svc.evaluate(vid, None, "t", "r")
            svc.approve(vid, "ok", "t", "r")
            ids.append(vid)
        results = await asyncio.gather(*(svc.promote(i, "race", "t", "r") for i in ids), return_exceptions=True)
        errors = [r for r in results if isinstance(r, AppError)]
        assert len(errors) == 1 and errors[0].code == ErrorCode.AI_VERSION_CONFLICT
        winner = svc.repo.read_state()["active_version_id"]
        assert winner in ids and sum(svc.get(i).status == "active" for i in ids) == 1

    asyncio.run(run())


def test_held_lock_yields_conflict_not_corruption(svc, monkeypatch):
    monkeypatch.setattr(repo_mod, "LOCK_TIMEOUT_S", 0.2)
    svc.repo.ensure()
    fd = os.open(svc.repo.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with pytest.raises(AppError) as e:
            create_version(svc, candidate_config())
        assert e.value.code == ErrorCode.AI_VERSION_CONFLICT
    finally:
        os.close(fd)
        svc.repo.lock_path.unlink()
    assert svc.repo.read_state()["versions"] == {}


def test_exception_inside_transaction_persists_nothing(svc):
    rev = svc.repo.read_state()["revision"]
    with pytest.raises(RuntimeError):
        with svc.repo.transaction() as tx:
            tx.state["active_version_id"] = "x"
            tx.audit({"event": "should-not-appear"})
            raise RuntimeError("boom")
    assert svc.repo.read_state()["revision"] == rev and svc.repo.read_audit() == []


def test_stale_revision_is_rejected(svc):
    rev = svc.repo.read_state()["revision"]
    create_version(svc, candidate_config())
    with pytest.raises(AppError) as e:
        with svc.repo.transaction(expected_revision=rev):
            pass
    assert e.value.code == ErrorCode.AI_VERSION_CONFLICT


def test_restart_and_other_worker_pick_up_active_version(svc):
    async def run():
        vid = await make_active(svc, candidate_config())
        rc.set_active(None)  # simulate a fresh process
        other = AIVersionService(svc.repo)
        other.load_active()
        assert rc.current().version_id == vid
        # a "second worker" notices a later rollback through the refresh hook
        rc.set_refresh_hook(other.refresh_if_changed)
        await svc.rollback(rc.BOOTSTRAP_VERSION_ID, "back", "t", "r")  # sets this process; reset to stale state
        rc.set_active(rc.get_active_override() or override_for(vid, svc))
        tok = rc.pin_current()
        try:
            assert rc.current().version_id == rc.BOOTSTRAP_VERSION_ID
        finally:
            rc.unpin(tok)

    asyncio.run(run())


def override_for(vid, svc):
    from app.services.ai_version_service import override_from_manifest

    return override_from_manifest(svc.repo.read_manifest(vid))


def test_tampered_active_manifest_keeps_current_config_and_reports_degraded(svc):
    async def run():
        vid = await make_active(svc, candidate_config())
        path = svc.repo.manifests / f"{vid}.json"
        data = json.loads(path.read_text())
        data["config"]["rag"]["top_k"] = 2  # fingerprint no longer matches
        path.write_text(json.dumps(data))
        before = rc.current().version_id
        svc.load_active()
        assert svc.degraded and "fingerprint" in svc.degraded
        assert rc.current().version_id == before
        assert any("fingerprint" in d for d in svc.active().drift)

    asyncio.run(run())


def test_llm_and_rag_services_follow_active_version_unless_explicit(svc):
    from app.services.llm_service import OllamaLLMService

    async def run():
        cfg = candidate_config()
        await make_active(svc, cfg)
        assert OllamaLLMService()._model == "cand-model:1b"
        assert OllamaLLMService(model="explicit:1")._model == "explicit:1"

    asyncio.run(run())


def test_secret_never_reaches_logs_audit_or_manifest(svc, caplog):
    async def run():
        with caplog.at_level("DEBUG"):
            await make_active(svc, candidate_config())
        assert TOKEN not in caplog.text
        blob = "".join(p.read_text(encoding="utf-8") for p in svc.repo.root.rglob("*") if p.is_file())
        assert TOKEN not in blob
        assert "admin_api_token" not in repr(get_settings()) or TOKEN not in repr(get_settings())

    asyncio.run(run())


# ------------------------------------------------------------------------------------------ API
@pytest.mark.asyncio
async def test_api_requires_admin_token_and_is_disabled_without_one(client, svc, monkeypatch):
    r = await client.get("/api/ai-versions")
    assert r.status_code == 401 and r.json()["error"]["code"] == "ADMIN_AUTH_REQUIRED"
    r = await client.get("/api/ai-versions", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401
    assert (await client.get("/api/ai-versions", headers=H)).status_code == 200
    monkeypatch.setattr(get_settings(), "admin_api_token", "")
    r = await client.get("/api/ai-versions", headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "ADMIN_DISABLED"


@pytest.mark.asyncio
async def test_api_client_cannot_supply_status_to_skip_validation(client, svc):
    r = await client.post("/api/ai-versions", headers=H,
                          json={"description": "sneaky", "config": candidate_config(), "status": "active"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert (await client.get("/api/ai-versions", headers=H)).json()["versions"] == []


@pytest.mark.asyncio
async def test_api_end_to_end_promote_trace_header_and_rollback(client, svc):
    cfg = candidate_config()
    r = await client.post("/api/ai-versions", headers=H, json={"description": "api candidate", "config": cfg})
    assert r.status_code == 201 and r.json()["status"] == "draft"
    vid = r.json()["version_id"]

    # not enough evidence yet -> no promotion path
    r = await client.post(f"/api/ai-versions/{vid}/evaluate", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "inconclusive"
    r = await client.post(f"/api/ai-versions/{vid}/promote", headers=H, json={"reason": "please"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "AI_VERSION_INVALID_TRANSITION"

    put_evidence(svc, vid, cfg)
    r = await client.post(f"/api/ai-versions/{vid}/evaluate", headers=H, json={})
    assert r.json()["status"] == "validated" and r.json()["evaluation"]["verdict"] == "PASS"
    r = await client.post(f"/api/ai-versions/{vid}/approve", headers={**H, "X-Admin-Actor": "reviewer.a"},
                          json={"reason": "metrics reviewed"})
    assert r.json()["status"] == "approved" and r.json()["approved_by"] == "reviewer.a"
    r = await client.post(f"/api/ai-versions/{vid}/promote", headers=H, json={"reason": "release"})
    assert r.status_code == 200 and r.json()["status"] == "active"

    r = await client.get("/api/ai-versions/active", headers=H)
    assert r.json()["active_version_id"] == vid and r.json()["effective_config"]["llm"]["model"] == "cand-model:1b"

    # a NEW request is served under the active version and says so
    r = await client.get("/health")
    assert r.headers["X-AI-Config-Version"] == vid
    assert len(r.headers["X-AI-Config-Fingerprint"]) == 16

    r = await client.post("/api/ai-versions/rollback", headers=H,
                          json={"target_version_id": rc.BOOTSTRAP_VERSION_ID, "reason": "return to env"})
    assert r.status_code == 200 and r.json()["active_version_id"] is None
    assert (await client.get("/health")).headers["X-AI-Config-Version"] == rc.BOOTSTRAP_VERSION_ID

    audit = (await client.get("/api/ai-versions/audit", headers=H)).json()
    assert [e["event"] for e in audit] == ["create", "evaluate", "evaluate", "approve", "promote", "rollback"]
    assert all(e["request_id"] for e in audit)
    assert audit[3]["actor"] == "reviewer.a"


@pytest.mark.asyncio
async def test_api_unknown_version_and_invalid_actor_label(client, svc):
    r = await client.get("/api/ai-versions/nope-123", headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "AI_VERSION_NOT_FOUND"
    r = await client.get("/api/ai-versions/..%2f..%2fetc", headers=H)
    assert r.status_code == 404
    vid = create_version(svc, candidate_config())
    r = await client.post(f"/api/ai-versions/{vid}/reject", headers={**H, "X-Admin-Actor": "<script>x</script>"},
                          json={"reason": "bad actor label"})
    assert r.json()["status"] == "rejected"
    assert svc.audit(version_id=vid)[-1].actor == "admin"


@pytest.mark.asyncio
async def test_existing_api_unaffected_and_trace_headers_present_by_default(client, svc):
    r = await client.get("/health")
    assert r.status_code in (200, 503)
    assert r.headers["X-AI-Config-Version"] == rc.BOOTSTRAP_VERSION_ID
    assert "X-Request-ID" in r.headers or "x-request-id" in r.headers


# ------------------------------------------------------------------------------------------ CLI / evidence production
def test_cli_offline_run_stores_evidence_and_cannot_validate(svc, monkeypatch):
    sys.path.insert(0, str(ROOT))
    from scripts import ai_version_cli

    vid = create_version(svc, candidate_config())
    env_before = {k: os.environ.get(k) for k in ("OLLAMA_MODEL", "TOP_K")}
    code = ai_version_cli.main(["evaluate", vid, "--mode", "offline", "--skip-functional"])
    assert code == 0
    d = svc.repo.evidence_dir(vid)
    assert all((d / f"{s}.json").is_file() for s in SUITES)
    assert {k: os.environ.get(k) for k in env_before} == env_before  # overrides were restored
    rec = svc.evaluate(vid, None, "t", "r")
    assert rec.evaluation.verdict == "BLOCKED" and rec.status == "inconclusive"  # no functional result
    assert svc.repo.read_state()["active_version_id"] is None
