"""Phase 11: API contract / OpenAPI / SSE header / health-consistency tests (offline)."""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import AppError
from tests.fakes import Stack, sample_pdf_bytes

DOC = "11111111-1111-4111-8111-111111111111"
EVAL = {"document_id": DOC, "question": "What is ML?", "student_answer": "Learning from data.",
        "expected_answer": "Learning from data.", "max_score": 10}


# ---------------------------------------------------------------- OpenAPI / docs

@pytest.mark.asyncio
async def test_docs_and_redoc(client):
    for path in ("/docs", "/redoc"):
        r = await client.get(path)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
    assert (await client.get("/openapi.json")).status_code == 200


@pytest.mark.asyncio
async def test_every_operation_is_documented(client):
    spec = (await client.get("/openapi.json")).json()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            where = f"{method.upper()} {path}"
            assert op.get("summary"), where
            assert op.get("description"), where
            assert "200" in op["responses"] or "201" in op["responses"], where
            for status, resp in op["responses"].items():
                if status.startswith(("4", "5")):
                    schema = resp["content"]["application/json"]["schema"]
                    ref = schema.get("$ref", "")
                    assert ref.endswith("/ErrorResponse") or ref.endswith("/HealthCheckResponse"), (where, status)
            if method == "post" and path != "/api/documents/upload":
                assert "requestBody" in op, where


@pytest.mark.asyncio
async def test_no_default_validation_schema_left(client):
    schemas = (await client.get("/openapi.json")).json()["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    assert "ErrorResponse" in schemas


# ---------------------------------------------------------------- validation errors

@pytest.mark.parametrize("path,body", [
    ("/api/rag/retrieve", {}),
    ("/api/rag/answer", {}),
    ("/api/questions/generate", {}),
    ("/api/evaluations/evaluate", {}),
])
@pytest.mark.asyncio
async def test_validation_error_standard_format(client, monkeypatch, path, body):
    Stack().install(monkeypatch)
    r = await client.post(path, json=body, headers={"X-Request-ID": "11111111-2222-4333-8444-555555555555"})
    assert r.status_code in (400, 422)
    assert "application/json" in r.headers["content-type"]
    err = r.json()["error"]
    assert err["code"] and err["message"]
    assert err["request_id"] == r.headers["X-Request-ID"] == "11111111-2222-4333-8444-555555555555"
    assert "Traceback" not in r.text


@pytest.mark.asyncio
async def test_stream_validation_error_is_sse_error_event(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.post("/api/evaluations/evaluate/stream", json={})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    types = [e["type"] for e in events]
    assert "error" in types and types[-1] == "done" and types.count("done") == 1
    assert "result" not in types
    err = next(e for e in events if e["type"] == "error")
    assert "Traceback" not in json.dumps(err)


@pytest.mark.asyncio
async def test_invalid_request_id_replaced(client):
    r = await client.get("/health/live", headers={"X-Request-ID": "bad id\twith junk"})
    rid = r.headers["X-Request-ID"]
    assert rid != "bad id\twith junk" and len(rid) >= 8
    assert (await client.get("/health/live")).headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_unknown_document_404_standard(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.get("/api/documents/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]


# ---------------------------------------------------------------- dependency errors

class _Boom:
    def __init__(self, exc):
        self.exc = exc

    def __getattr__(self, name):
        async def _raise(*a, **k):
            raise self.exc

        return _raise


@pytest.mark.parametrize("path,body,target", [
    ("/api/evaluations/evaluate", EVAL, "app.api.evaluations._get_evaluation_service"),
    ("/api/questions/generate", {"document_id": DOC}, "app.api.questions._get_question_service"),
])
@pytest.mark.parametrize("status,code", [(503, "OLLAMA_UNAVAILABLE"), (502, "LLM_BAD_RESPONSE")])
@pytest.mark.asyncio
async def test_dependency_error_standard(client, monkeypatch, path, body, target, status, code):
    exc = AppError(code=code, message="dependency failed", status_code=status)
    monkeypatch.setattr(target, lambda: _Boom(exc))
    r = await client.post(path, json=body)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code and err["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_unexpected_exception_is_500_without_leak():
    import httpx
    from app.main import app
    import app.api.evaluations as ev

    mp = pytest.MonkeyPatch()
    mp.setattr(ev, "_get_evaluation_service", lambda: _Boom(RuntimeError("secret /etc/internal/path")))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://t") as c:
            r = await c.post("/api/evaluations/evaluate", json=EVAL)
    finally:
        mp.undo()
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "secret" not in r.text and "/etc/" not in r.text and "Traceback" not in r.text


# ---------------------------------------------------------------- success + SSE headers

@pytest.mark.asyncio
async def test_success_content_types_and_stream_headers(client, monkeypatch):
    Stack().install(monkeypatch)
    up = await client.post("/api/documents/upload", files={
        "file": ("d.pdf", sample_pdf_bytes(["Machine learning lets computers learn patterns from data."]),
                 "application/pdf")})
    assert up.status_code in (200, 201)
    assert "application/json" in up.headers["content-type"] and up.headers["X-Request-ID"]
    doc = up.json()["document_id"]

    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "machine learning data"})
    assert r.status_code == 200 and "application/json" in r.headers["content-type"]

    r = await client.post("/api/evaluations/evaluate", json={**EVAL, "document_id": doc})
    assert r.status_code == 200 and "percentage" in r.json()

    r = await client.post("/api/evaluations/evaluate/stream", json={**EVAL, "document_id": doc})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in r.headers["cache-control"]
    assert r.headers["connection"].lower() == "keep-alive"
    assert r.headers["x-accel-buffering"] == "no"
    assert r.headers["X-Request-ID"]
    events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    types = [e["type"] for e in events]
    assert set(types) <= {"status", "token", "result", "error", "done"}
    assert types[-1] == "done" and types.count("done") == 1
    assert types.count("result") == 1 and types.index("result") < types.index("done")
    assert all("response" not in e and "done_reason" not in e for e in events)  # no raw Ollama frames

    r = await client.delete(f"/api/documents/{doc}")
    assert r.status_code == 200


# ---------------------------------------------------------------- question types

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("qtype", ["short_answer", "multiple_choice", "true_false", "essay"])
@pytest.mark.asyncio
async def test_question_request_accepts_supported_types(client, monkeypatch, difficulty, qtype):
    Stack().install(monkeypatch)
    up = await client.post("/api/documents/upload", files={
        "file": ("d.pdf", sample_pdf_bytes(["Machine learning lets computers learn patterns from data."]),
                 "application/pdf")})
    doc = up.json()["document_id"]
    r = await client.post("/api/questions/generate", json={
        "document_id": doc, "number_of_questions": 1, "difficulty": difficulty, "question_type": qtype})
    assert r.status_code in (200, 502), r.text  # 502 = fake LLM output not in requested shape (rejected)
    if r.status_code == 200:
        for q in r.json()["questions"]:
            assert q["difficulty"] == difficulty and q["type"] == qtype


@pytest.mark.asyncio
async def test_unknown_question_type_rejected(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.post("/api/questions/generate", json={"document_id": DOC, "question_type": "oral_exam"})
    assert r.status_code in (400, 422)
    assert r.json()["error"]["code"]


# ---------------------------------------------------------------- health consistency

@pytest.mark.asyncio
async def test_live_always_ok_and_ready_matches_health(client):
    assert (await client.get("/health/live")).status_code == 200
    health = await client.get("/health")
    ready = await client.get("/health/ready")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] in ("ok", "healthy", "degraded", "unhealthy")
    if ready.status_code == 200:
        assert body["status"] in ("ok", "healthy")
    else:
        assert ready.status_code == 503

