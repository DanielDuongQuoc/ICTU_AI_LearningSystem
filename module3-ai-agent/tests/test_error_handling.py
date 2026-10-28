"""Phase 9 — standard error contract, request id, dependency error normalisation."""

from __future__ import annotations

import logging
import re
import uuid
from unittest.mock import MagicMock

import httpx
import pytest

from app.core.exceptions import AppError, ErrorCode
from app.core.request_context import resolve_request_id
from app.main import app
from app.repositories.chroma_repository import ChromaRepository
from app.services.llm_service import OllamaLLMService

RETRIEVE_BODY = {"document_id": str(uuid.uuid4()), "question": "What is AI?"}


class _FailingRAG:
    """RAG service stand-in whose retrieve() raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def retrieve(self, **kwargs):
        raise self.exc

    async def answer_question(self, **kwargs):
        raise self.exc


def _patch_rag(monkeypatch, exc: Exception) -> None:
    monkeypatch.setattr("app.api.rag._get_rag_service", lambda: _FailingRAG(exc))


def _assert_contract(resp: httpx.Response, code: str, status: int) -> dict:
    assert resp.status_code == status
    body = resp.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert set(err) == {"code", "message", "request_id"}
    assert err["code"] == code
    assert err["message"]
    assert err["request_id"] == resp.headers["X-Request-ID"]
    return err


# ── Validation / generic handlers ──────────────────────────────────


@pytest.mark.asyncio
async def test_validation_error_format(client):
    resp = await client.post("/api/rag/retrieve", json={})
    err = _assert_contract(resp, "VALIDATION_ERROR", 422)
    assert err["message"] == "Invalid request data."
    assert "detail" not in resp.json()


@pytest.mark.asyncio
async def test_unknown_route_uses_standard_format(client):
    resp = await client.get("/api/does-not-exist")
    _assert_contract(resp, "NOT_FOUND", 404)


@pytest.mark.asyncio
async def test_method_not_allowed_uses_standard_format(client):
    resp = await client.delete("/health")
    _assert_contract(resp, "METHOD_NOT_ALLOWED", 405)


@pytest.mark.asyncio
async def test_application_error_format(client, monkeypatch):
    _patch_rag(monkeypatch, AppError(ErrorCode.DOCUMENT_NOT_FOUND, "Document not found."))
    resp = await client.post("/api/rag/retrieve", json=RETRIEVE_BODY)
    err = _assert_contract(resp, "DOCUMENT_NOT_FOUND", 404)
    assert err["message"] == "Document not found."


@pytest.mark.asyncio
async def test_unexpected_error_does_not_expose_traceback(monkeypatch):
    from httpx import ASGITransport, AsyncClient

    _patch_rag(monkeypatch, RuntimeError("secret internals /srv/app/leak.py"))
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        resp = await ac.post("/api/rag/retrieve", json=RETRIEVE_BODY)
    err = _assert_contract(resp, "INTERNAL_SERVER_ERROR", 500)
    assert err["message"] == "An unexpected error occurred."
    assert "Traceback" not in resp.text
    assert "secret internals" not in resp.text
    assert "leak.py" not in resp.text


# ── Dependency errors → status mapping ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,status",
    [
        (ErrorCode.OLLAMA_UNAVAILABLE, 503),
        (ErrorCode.OLLAMA_TIMEOUT, 504),
        (ErrorCode.OLLAMA_MODEL_NOT_FOUND, 503),
        (ErrorCode.CHROMA_ERROR, 503),
        (ErrorCode.EMBEDDING_ERROR, 502),
        (ErrorCode.LLM_INVALID_JSON, 502),
        (ErrorCode.LLM_EMPTY_RESPONSE, 502),
        (ErrorCode.INSUFFICIENT_CONTEXT, 422),
        (ErrorCode.SOURCE_INTEGRITY_ERROR, 500),
    ],
)
async def test_error_code_status_mapping(client, monkeypatch, code, status):
    _patch_rag(monkeypatch, AppError(code, "Dependency failed."))
    resp = await client.post("/api/rag/retrieve", json=RETRIEVE_BODY)
    _assert_contract(resp, code.value if hasattr(code, "value") else str(code), status)


# ── Request id ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_id_generated_when_missing(client):
    resp = await client.get("/health/live")
    rid = resp.headers["X-Request-ID"]
    assert str(uuid.UUID(rid)) == rid


@pytest.mark.asyncio
async def test_valid_request_id_is_reused(client):
    resp = await client.get("/health/live", headers={"X-Request-ID": "test-request-id"})
    assert resp.headers["X-Request-ID"] == "test-request-id"


@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced(client):
    bad = "bad id with spaces <script>"
    resp = await client.get("/health/live", headers={"X-Request-ID": bad})
    rid = resp.headers["X-Request-ID"]
    assert rid != bad
    uuid.UUID(rid)


@pytest.mark.asyncio
async def test_error_request_id_matches_header_and_reuses_client_id(client):
    resp = await client.post(
        "/api/rag/retrieve", json={}, headers={"X-Request-ID": "client-abc-123"}
    )
    err = _assert_contract(resp, "VALIDATION_ERROR", 422)
    assert err["request_id"] == "client-abc-123"


def test_resolve_request_id_rules():
    assert resolve_request_id("abc-123") == "abc-123"
    assert resolve_request_id(None) != resolve_request_id(None)
    assert resolve_request_id("x" * 500) != "x" * 500
    assert resolve_request_id("") != ""


# ── Ollama normalisation (real OllamaLLMService over a mock transport) ─


def _ollama(handler) -> OllamaLLMService:
    svc = OllamaLLMService(base_url="http://ollama.test")
    svc._client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    svc._retry_backoff_s = 0
    return svc


@pytest.mark.asyncio
async def test_ollama_unavailable_normalised():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(AppError) as ei:
        await _ollama(handler).generate("sys", "user")
    assert ei.value.code == ErrorCode.OLLAMA_UNAVAILABLE


@pytest.mark.asyncio
async def test_ollama_timeout_normalised_and_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    with pytest.raises(AppError) as ei:
        await _ollama(handler).generate("sys", "user")
    assert ei.value.code == ErrorCode.OLLAMA_TIMEOUT
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ollama_transient_503_retried_once_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"message": {"content": "hello"}})

    assert await _ollama(handler).generate("sys", "user") == "hello"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_ollama_retry_is_capped_at_one():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="busy")

    with pytest.raises(AppError):
        await _ollama(handler).generate("sys", "user")
    assert len(calls) == 2  # original + exactly one retry


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500, 400])
async def test_ollama_non_transient_status_not_retried(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, text="nope")

    with pytest.raises(AppError):
        await _ollama(handler).generate("sys", "user")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ollama_empty_response_normalised():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "   "}})

    with pytest.raises(AppError) as ei:
        await _ollama(handler).generate("sys", "user")
    assert ei.value.code == ErrorCode.LLM_EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_ollama_stream_transient_503_retried_once():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        body = '{"message":{"content":"hi"},"done":false}\n{"message":{"content":""},"done":true}\n'
        return httpx.Response(200, text=body)

    pieces = [p async for p in _ollama(handler).generate_stream("s", "u")]
    assert pieces == ["hi"]
    assert len(calls) == 2


# ── Chroma normalisation ────────────────────────────────────────────


def _repo_with_broken_collection() -> ChromaRepository:
    repo = ChromaRepository.__new__(ChromaRepository)
    collection = MagicMock()
    boom = RuntimeError("chroma internals /var/db/leak")
    for name in ("upsert", "query", "get", "delete", "count"):
        getattr(collection, name).side_effect = boom
    repo._collection = collection
    repo._collection_name = "test"
    return repo


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.add_documents(["1"], ["t"], [[0.1]], [{"document_id": "d"}]),
        lambda r: r.query([0.1], "d", top_k=1),
        lambda r: r.get_document("d"),
        lambda r: r.delete_document("d"),
        lambda r: r.count(),
        lambda r: r.count_by_document("d"),
        lambda r: r.document_exists("d"),
    ],
)
def test_chroma_exceptions_become_chroma_error(call):
    with pytest.raises(AppError) as ei:
        call(_repo_with_broken_collection())
    assert ei.value.code == ErrorCode.CHROMA_ERROR
    assert "leak" not in ei.value.message


# ── Logging hygiene ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_logs_contain_request_id_and_no_body(client, caplog):
    secret = "STUDENT-SECRET-ANSWER-12345"
    with caplog.at_level(logging.DEBUG):
        resp = await client.post(
            "/api/rag/retrieve",
            json={"question": secret},
            headers={"X-Request-ID": "log-test-id"},
        )
    assert resp.status_code == 422
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "log-test-id" in text
    assert "[request] started" in text and "[request] completed" in text
    assert secret not in text


@pytest.mark.asyncio
async def test_slow_request_warning(client, caplog, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "slow_request_threshold_ms", 0.0001, raising=False)
    with caplog.at_level(logging.WARNING, logger="app.request"):
        await client.get("/health/live")
    assert any(
        r.levelno == logging.WARNING and "slow" in r.getMessage().lower()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_unexpected_exception_logged_server_side(monkeypatch, caplog):
    from httpx import ASGITransport, AsyncClient

    _patch_rag(monkeypatch, RuntimeError("kaboom-internal"))
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    with caplog.at_level(logging.ERROR):
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.post("/api/rag/retrieve", json=RETRIEVE_BODY)
    assert "kaboom-internal" not in resp.text
    assert any(r.exc_info for r in caplog.records)
    assert re.search(r"request_id=\S+", "\n".join(r.getMessage() for r in caplog.records))
