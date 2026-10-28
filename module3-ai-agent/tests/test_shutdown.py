"""Phase 16 - graceful shutdown / startup after dependency failure."""

from __future__ import annotations

import httpx
import pytest

from app.main import app, lifespan
from app.services import embedding_service, llm_service


class _Closable:
    def __init__(self, fail: bool = False) -> None:
        self.closed = False
        self.fail = fail

    async def close(self) -> None:
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


def _no_deps(monkeypatch, *, chroma_ok: bool = True, ollama_ok: bool = True):
    def init():
        if not chroma_ok:
            raise RuntimeError("chroma unavailable")

    async def ollama() -> bool:
        return ollama_ok

    monkeypatch.setattr("app.main.init_chroma", init)
    monkeypatch.setattr("app.services.ollama_checker.check_ollama_health", ollama)


@pytest.mark.asyncio
async def test_shutdown_closes_shared_http_clients(monkeypatch):
    _no_deps(monkeypatch)
    llm, emb = _Closable(), _Closable()
    monkeypatch.setattr(llm_service, "_llm_service", llm)
    monkeypatch.setattr(embedding_service, "_embedding_service", emb)
    async with lifespan(app):
        assert not llm.closed and not emb.closed
    assert llm.closed and emb.closed


@pytest.mark.asyncio
async def test_shutdown_never_raises_when_a_client_fails_to_close(monkeypatch):
    _no_deps(monkeypatch)
    llm, emb = _Closable(fail=True), _Closable()
    monkeypatch.setattr(llm_service, "_llm_service", llm)
    monkeypatch.setattr(embedding_service, "_embedding_service", emb)
    async with lifespan(app):
        pass
    assert llm.closed and emb.closed                     # the second client was still closed


@pytest.mark.asyncio
async def test_shutdown_with_clients_never_created(monkeypatch):
    _no_deps(monkeypatch)
    monkeypatch.setattr(llm_service, "_llm_service", None)
    monkeypatch.setattr(embedding_service, "_embedding_service", None)
    async with lifespan(app):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("chroma_ok,ollama_ok", [(False, True), (True, False), (False, False)])
async def test_startup_degrades_instead_of_crashing_when_dependencies_are_down(monkeypatch, chroma_ok, ollama_ok):
    _no_deps(monkeypatch, chroma_ok=chroma_ok, ollama_ok=ollama_ok)
    monkeypatch.setattr(llm_service, "_llm_service", None)
    monkeypatch.setattr(embedding_service, "_embedding_service", None)
    async with lifespan(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/health/live")).status_code == 200


@pytest.mark.asyncio
async def test_shutdown_runs_even_if_the_app_body_raises(monkeypatch):
    _no_deps(monkeypatch)
    llm = _Closable()
    monkeypatch.setattr(llm_service, "_llm_service", llm)
    monkeypatch.setattr(embedding_service, "_embedding_service", None)
    with pytest.raises(RuntimeError):
        async with lifespan(app):
            raise RuntimeError("boom")
    assert llm.closed
