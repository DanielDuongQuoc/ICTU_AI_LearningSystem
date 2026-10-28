"""Phase 16 - request id propagation and log privacy."""

from __future__ import annotations

import logging

import pytest

from app.core.config import get_settings
from tests.fakes import Stack, sample_pdf_bytes

SECRET_ANSWER = "STUDENT-ANSWER-SENTINEL-8841"
SECRET_QUESTION = "QUESTION-SENTINEL-5520 about zebracorn?"
SECRET_PDF = "Zebracorn PDF-CONTENT-SENTINEL-3309 models learn patterns from data."


@pytest.fixture(autouse=True)
def _no_llm_cap(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_client_request_id_is_echoed_and_logged(client, caplog):
    caplog.set_level(logging.INFO)
    r = await client.get("/health/live", headers={"X-Request-ID": "ops-req-12345"})
    assert r.headers["x-request-id"] == "ops-req-12345"
    assert any("request_id=ops-req-12345" in rec.getMessage() and "completed" in rec.getMessage()
               for rec in caplog.records)


@pytest.mark.asyncio
async def test_error_body_header_and_log_share_one_request_id(client, monkeypatch, caplog):
    Stack().install(monkeypatch)
    caplog.set_level(logging.INFO)
    r = await client.get("/api/documents/zzz-unknown")
    rid = r.headers["x-request-id"]
    assert r.json()["error"]["request_id"] == rid
    assert any(f"request_id={rid}" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_logs_never_contain_student_answer_question_or_pdf_text(client, monkeypatch, caplog):
    Stack().install(monkeypatch)
    caplog.set_level(logging.DEBUG)
    up = await client.post("/api/documents/upload",
                           files={"file": ("a.pdf", sample_pdf_bytes([SECRET_PDF]), "application/pdf")})
    doc = up.json()["document_id"]
    body = {"document_id": doc, "question": SECRET_QUESTION, "student_answer": SECRET_ANSWER,
            "expected_answer": "They learn patterns from data.", "max_score": 10}
    assert (await client.post("/api/evaluations/evaluate", json=body)).status_code == 200
    await client.post("/api/evaluations/evaluate/stream", json=body)
    await client.post("/api/rag/retrieve", json={"document_id": doc, "question": SECRET_QUESTION})
    await client.post("/api/evaluations/evaluate", json={**body, "student_answer": ""})   # validation error path

    logged = "\n".join(rec.getMessage() for rec in caplog.records)
    for sentinel in ("STUDENT-ANSWER-SENTINEL-8841", "QUESTION-SENTINEL-5520", "PDF-CONTENT-SENTINEL-3309"):
        assert sentinel not in logged, sentinel


@pytest.mark.asyncio
async def test_unexpected_500_hides_internals_but_logs_once_server_side(monkeypatch, caplog):
    import httpx

    from app.main import app

    stack = Stack().install(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("internal-detail-/etc/passwd")

    monkeypatch.setattr(stack.rag, "retrieve", boom)
    caplog.set_level(logging.INFO)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/api/rag/retrieve", json={"document_id": "11111111-1111-4111-8111-111111111111", "question": "q"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "internal-detail" not in r.text and "Traceback" not in r.text
    tracebacks = [rec for rec in caplog.records if rec.exc_info]
    assert len(tracebacks) == 1                          # not logged repeatedly at several layers


def test_log_level_setting(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert Settings(_env_file=None).log_level == "warning"
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
