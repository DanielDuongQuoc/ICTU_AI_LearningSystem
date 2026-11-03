"""Phase 21 - an LLM provider restart must degrade safely and recover (reuses the Phase 14 smoke flow)."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from scripts.smoke_test import run_smoke
from tests.fakes import Stack


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
