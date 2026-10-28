"""Phase 16 - dependency failure and recovery, document lifecycle. No real Ollama needed."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.exceptions import AppError, ErrorCode
from app.core.config import get_settings
from app.services.llm_service import OllamaLLMService
from tests.fakes import Stack, sample_pdf_bytes

TEXT_A = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."
TEXT_B = "Quokkaplex networks describe database indexing. Quokkaplex indexes speed up queries."


@pytest.fixture(autouse=True)
def _default_limits(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _upload(client, text: str, name: str = "a.pdf") -> str:
    r = await client.post("/api/documents/upload", files={"file": (name, sample_pdf_bytes([text]), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()["document_id"]


def _events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def _eval_body(doc: str) -> dict[str, Any]:
    return {"document_id": doc, "question": "What do zebracorn models learn from?",
            "student_answer": "They learn from data.", "expected_answer": "They learn patterns from data.",
            "max_score": 10}


# ------------------------------------------------------------------ Ollama: service level

def _svc(handler) -> OllamaLLMService:
    svc = OllamaLLMService(base_url="http://ollama.test", model="m")
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return svc


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [
    (httpx.ConnectError("refused"), ErrorCode.OLLAMA_UNAVAILABLE),
    (httpx.ReadTimeout("slow"), ErrorCode.OLLAMA_TIMEOUT),
])
async def test_ollama_down_then_recovers(failure, code):
    state = {"down": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["down"]:
            raise failure
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    svc = _svc(handler)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == code
    state["down"] = False
    assert await svc.generate("s", "u") == "ok"          # same instance, no restart needed
    await svc.close()


@pytest.mark.asyncio
async def test_stream_interruption_and_invalid_json_then_recovery():
    mode = {"m": "timeout"}

    async def body():
        yield (json.dumps({"message": {"content": "par"}, "done": False}) + "\n").encode()
        if mode["m"] == "timeout":
            raise httpx.ReadTimeout("stream stalled")
        if mode["m"] == "garbage":
            yield b"{not json}\n"
            return
        yield (json.dumps({"message": {"content": "tial"}, "done": True}) + "\n").encode()

    svc = _svc(lambda request: httpx.Response(200, content=body()))

    async def drain() -> str:
        return "".join([p async for p in svc.generate_stream("s", "u")])

    with pytest.raises(AppError) as exc:
        await drain()
    assert exc.value.code == ErrorCode.OLLAMA_TIMEOUT
    mode["m"] = "garbage"
    with pytest.raises(AppError) as exc:
        await drain()
    assert exc.value.code == ErrorCode.OLLAMA_INVALID_RESPONSE
    mode["m"] = "ok"
    assert await drain() == "partial"
    await svc.close()


# ------------------------------------------------------------------ readiness follows dependencies

@pytest.mark.asyncio
async def test_readiness_tracks_ollama_and_chroma_recovery(client, monkeypatch):
    state = {"ollama": False, "chroma": True}

    async def ollama() -> bool:
        return state["ollama"]

    monkeypatch.setattr("app.api.health.check_ollama_health", ollama)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: state["chroma"])

    assert (await client.get("/ready")).status_code == 503
    assert (await client.get("/health/live")).status_code == 200      # process stays alive
    assert (await client.get("/health")).json()["status"] == "degraded"
    state["ollama"] = True
    assert (await client.get("/ready")).status_code == 200
    state["chroma"] = False
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["services"]["chroma"] == "error"
    state["chroma"] = True
    assert (await client.get("/ready")).status_code == 200


@pytest.mark.asyncio
async def test_health_never_calls_llm_generation(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    await client.get("/health")
    await client.get("/ready")
    assert stack.llm.calls == []


# ------------------------------------------------------------------ ChromaDB failure + recovery via API

@pytest.mark.asyncio
async def test_chroma_failure_returns_structured_error_then_recovers(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client, TEXT_A)
    real_query = stack.repo.query
    state = {"fail": True}

    def flaky(*a, **k):
        if state["fail"]:
            raise RuntimeError("chroma exploded: /secret/path")
        return real_query(*a, **k)

    monkeypatch.setattr(stack.repo, "query", flaky)
    body = {"document_id": doc, "question": "zebracorn data"}
    r = await client.post("/api/rag/retrieve", json=body)
    assert r.status_code == 503 and r.json()["error"]["code"] == "CHROMA_ERROR"
    assert "secret/path" not in r.text and "Traceback" not in r.text and r.json()["error"]["request_id"]
    state["fail"] = False
    assert (await client.post("/api/rag/retrieve", json=body)).status_code == 200


# ------------------------------------------------------------------ streaming failure through the API

@pytest.mark.asyncio
async def test_stream_error_event_then_next_stream_succeeds(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client, TEXT_A)
    original = stack.llm.generate_stream
    state = {"fail": True}

    async def flaky(*a, **k):
        if state["fail"]:
            yield '{"score": 8, "corr'
            raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.")
        async for piece in original(*a, **k):
            yield piece

    monkeypatch.setattr(stack.llm, "generate_stream", flaky)
    r = await client.post("/api/evaluations/evaluate/stream", json=_eval_body(doc))
    ev = _events(r.text)
    types = [e["type"] for e in ev]
    assert r.status_code == 200 and types.count("error") == 1 and types.count("result") == 0
    assert types[-1] == "done" and types.count("done") == 1
    assert [e for e in ev if e["type"] == "error"][0]["code"] == "OLLAMA_TIMEOUT"
    assert "Traceback" not in r.text

    state["fail"] = False
    ev = _events((await client.post("/api/evaluations/evaluate/stream", json=_eval_body(doc))).text)
    types = [e["type"] for e in ev]
    assert types.count("result") == 1 and types[-1] == "done" and "error" not in types
    assert types.index("result") < types.index("done")


# ------------------------------------------------------------------ document lifecycle / integrity

@pytest.mark.asyncio
async def test_delete_one_document_leaves_other_intact(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    a, b = await _upload(client, TEXT_A, "a.pdf"), await _upload(client, TEXT_B, "b.pdf")
    before_b = stack.repo.count_by_document(b)
    assert (await client.delete(f"/api/documents/{a}")).status_code == 200
    assert stack.repo.count_by_document(a) == 0 and stack.repo.count_by_document(b) == before_b > 0
    r = await client.post("/api/rag/retrieve", json={"document_id": b, "question": "quokkaplex queries"})
    assert r.status_code == 200 and all(c["document_id"] == b for c in r.json()["chunks"])


@pytest.mark.asyncio
async def test_failed_ingestion_leaves_no_false_success(client, monkeypatch):
    stack = Stack().install(monkeypatch)

    async def broken(texts):
        raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "embedding down")

    monkeypatch.setattr(stack.embedding, "embed_documents", broken)
    r = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([TEXT_A]), "application/pdf")})
    assert r.status_code >= 400 and "document_id" not in r.json()
    assert stack.repo.count() == 0                       # nothing half-indexed


@pytest.mark.asyncio
async def test_invalid_upload_stores_nothing(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    r = await client.post("/api/documents/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400 and stack.repo.count() == 0


@pytest.mark.asyncio
async def test_duplicate_upload_follows_current_behaviour(client, monkeypatch):
    """Documented behaviour: uploads are not idempotent; each gets its own isolated document_id."""
    stack = Stack().install(monkeypatch)
    d1, d2 = await _upload(client, TEXT_A), await _upload(client, TEXT_A)
    assert d1 != d2
    assert stack.repo.count_by_document(d1) == stack.repo.count_by_document(d2) > 0
    await client.delete(f"/api/documents/{d1}")
    assert stack.repo.count_by_document(d2) > 0
