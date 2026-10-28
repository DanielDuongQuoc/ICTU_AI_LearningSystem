"""Phase 21 - AI version rollback must change what the RUNTIME really uses (not just a stored label),
and an Ollama restart must degrade safely and recover (reuses Phase 18 fixtures and the Phase 14 smoke flow)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.core import runtime_config as rc
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.services.llm_service import OllamaLLMService
from scripts.smoke_test import run_smoke
from tests.fakes import Stack
from tests.test_phase18_governance import candidate_config, create_version, make_active, put_evidence, svc  # noqa: F401


def _capturing_llm(seen: list[str]) -> OllamaLLMService:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    llm = OllamaLLMService(base_url="http://ollama.test")           # model follows the ACTIVE version per call
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return llm


def test_rollback_changes_the_model_actually_sent_to_ollama(svc):  # noqa: F811
    async def run():
        seen: list[str] = []
        llm = _capturing_llm(seen)
        v_a = await make_active(svc, candidate_config(), "version A")             # AI version A active
        await llm.generate("s", "u")
        cfg_b = candidate_config(model="cand-model:2b", top_k=3)
        v_b = create_version(svc, cfg_b, "version B")
        put_evidence(svc, v_b, cfg_b)
        svc.evaluate(v_b, None, "t", "r")
        svc.approve(v_b, "ok", "t", "r")
        await svc.promote(v_b, "upgrade", "t", "r")                                # B promoted
        await llm.generate("s", "u")
        await svc.rollback(v_a, "B has latency/quality issues", "t", "r")          # back to A
        await llm.generate("s", "u")
        assert seen == ["cand-model:1b", "cand-model:2b", "cand-model:1b"], "runtime did not follow the rollback"
        assert rc.current().version_id == v_a
        await llm.close()

    asyncio.run(run())


def test_rollback_keeps_manifests_audit_and_does_not_touch_vectors(svc):  # noqa: F811
    async def run():
        stack = Stack()
        before = stack.repo.count()
        v_a = await make_active(svc, candidate_config(), "version A")
        cfg_b = candidate_config(model="cand-model:2b", top_k=3)
        v_b = create_version(svc, cfg_b, "version B")
        put_evidence(svc, v_b, cfg_b)
        svc.evaluate(v_b, None, "t", "r")
        svc.approve(v_b, "ok", "t", "r")
        await svc.promote(v_b, "up", "t", "r")
        await svc.rollback(v_a, "rollback drill", "t", "r")
        assert svc.repo.read_manifest(v_a) is not None and svc.repo.read_manifest(v_b) is not None
        assert [e.event for e in svc.audit() if e.event in ("promote", "rollback")].count("rollback") == 1
        assert stack.repo.count() == before                      # AI-version rollback never touches ChromaDB

    asyncio.run(run())


# --------------------------------------------------------------- Ollama restart through the smoke flow
@pytest.mark.asyncio
async def test_ollama_restart_degrades_then_smoke_flow_passes(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    state = {"ollama": False}

    async def ollama_ok() -> bool:
        return state["ollama"]

    monkeypatch.setattr("app.api.health.check_ollama_health", ollama_ok)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: True)

    down = await run_smoke(client, require_ready=True)                          # Ollama "restarting"
    assert [s for s, ok, _ in down if not ok] == ["ready"], down                  # fails at readiness, nothing crashes
    assert (await client.get("/health/live")).status_code == 200
    assert stack.llm.calls == []                                                 # readiness never generated anything

    state["ollama"] = True                                                       # Ollama back
    up = await run_smoke(client, require_ready=True)
    assert [s for s, ok, _ in up if not ok] == [], up
    assert {s for s, _, _ in up} >= {"health", "ready", "upload", "retrieve", "questions", "evaluate", "stream", "delete"}
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_request_failing_during_ollama_outage_then_next_succeeds(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    real = stack.llm.generate_json
    state = {"down": True}

    async def flaky(*a, **k):
        if state["down"]:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.")
        return await real(*a, **k)

    monkeypatch.setattr(stack.llm, "generate_json", flaky)
    up = await client.post("/api/documents/upload", files={"file": ("a.pdf", __import__("tests.fakes", fromlist=["x"]).sample_pdf_bytes(
        ["Zebracorn systems explain machine learning."]), "application/pdf")})
    doc = up.json()["document_id"]
    body = {"document_id": doc, "number_of_questions": 1}
    r = await client.post("/api/questions/generate", json=body)
    assert r.status_code == 503 and r.json()["error"]["request_id"]
    state["down"] = False
    assert (await client.post("/api/questions/generate", json=body)).status_code == 200
    get_settings.cache_clear()
