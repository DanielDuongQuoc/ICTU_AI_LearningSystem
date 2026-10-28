"""Phase 20 - readiness includes storage; liveness never depends on dependencies; recovery without restart."""

from __future__ import annotations

import pytest

from tests.fakes import Stack


@pytest.mark.asyncio
async def test_low_disk_makes_ready_503_then_recovers_and_live_stays_200(client, monkeypatch):
    async def ok() -> bool:
        return True

    monkeypatch.setattr("app.api.health.check_ollama_health", ok)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: True)
    state = {"disk_ok": False}
    monkeypatch.setattr("app.api.health.check_storage_health", lambda: state["disk_ok"])

    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["services"]["storage"] == "error"
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/health")).status_code == 200            # /health always 200, body says degraded
    state["disk_ok"] = True
    r = await client.get("/ready")
    assert r.status_code == 200 and r.json()["services"]["storage"] == "ok"


@pytest.mark.asyncio
async def test_storage_check_exception_is_reported_not_raised(client, monkeypatch):
    def boom() -> bool:
        raise OSError("statvfs failed")

    monkeypatch.setattr("app.api.health.check_storage_health", boom)
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["services"]["storage"] == "error"


def test_real_storage_check_respects_threshold(monkeypatch):
    from app.api.health import check_storage_health
    from app.core.config import get_settings

    monkeypatch.setenv("MIN_FREE_DISK_MB", "1")
    get_settings.cache_clear()
    assert check_storage_health() is True
    monkeypatch.setenv("MIN_FREE_DISK_MB", str(10**9))                 # ~1 PB: always too much
    get_settings.cache_clear()
    assert check_storage_health() is False
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_health_never_calls_generation(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    await client.get("/health")
    await client.get("/ready")
    assert stack.llm.calls == []
