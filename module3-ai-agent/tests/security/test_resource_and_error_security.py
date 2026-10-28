"""Resource limits, rate limiting, error leakage, HTTP headers, config guards (Phase 19)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.main import app
from tests.security.conftest import BAD_MARKERS, TEXT_A, assert_safe_error, upload_text


# ---------------------------------------------------------------- bounded inputs
@pytest.mark.parametrize("top_k", [1_000_000, 10**12, 0, -1, "many", 1.5])
async def test_top_k_is_validated_everywhere(client, stack, top_k):
    doc = await upload_text(client, TEXT_A)
    for url, body in (
        ("/api/rag/retrieve", {"document_id": doc, "question": "zebracorn", "top_k": top_k}),
        (f"/api/documents/{doc}/search", {"query": "zebracorn", "top_k": top_k}),
    ):
        resp = await client.post(url, json=body)
        assert resp.status_code in (400, 422), (url, top_k, resp.text)
        assert_safe_error(resp)


async def test_question_and_query_length_capped(client, stack, settings):
    doc = await upload_text(client, TEXT_A)
    huge = "a" * (settings.max_question_chars + 1)
    for url, body in (
        ("/api/rag/retrieve", {"document_id": doc, "question": huge}),
        ("/api/rag/answer", {"document_id": doc, "question": huge}),
        (f"/api/documents/{doc}/search", {"query": huge}),
    ):
        resp = await client.post(url, json=body)
        assert resp.status_code in (400, 413, 422), (url, resp.status_code)
        assert_safe_error(resp)


async def test_student_answer_length_capped_before_llm(client, stack, settings):
    doc = await upload_text(client, TEXT_A)
    resp = await client.post("/api/evaluations/evaluate", json={
        "document_id": doc, "question": "q?", "expected_answer": "e",
        "student_answer": "x" * (settings.max_answer_chars + 1), "max_score": 10})
    assert resp.status_code in (400, 413, 422)
    assert not stack.llm.calls


async def test_number_of_questions_capped(client, stack):
    doc = await upload_text(client, TEXT_A)
    resp = await client.post("/api/questions/generate", json={"document_id": doc, "number_of_questions": 10**6})
    assert resp.status_code in (400, 422)
    assert not stack.llm.calls


@pytest.mark.parametrize("max_score", [0, -1, 1e12, "inf", "nan"])
async def test_max_score_bounded(client, stack, max_score):
    doc = await upload_text(client, TEXT_A)
    resp = await client.post("/api/evaluations/evaluate", json={
        "document_id": doc, "question": "q?", "expected_answer": "e", "student_answer": "a", "max_score": max_score})
    assert resp.status_code in (400, 422)


async def test_oversized_json_body_rejected_with_413(client, stack, set_limit):
    set_limit(max_request_body_bytes=2048)
    resp = await client.post("/api/rag/retrieve", content=json.dumps(
        {"document_id": "x", "question": "q" * 5000}), headers={"content-type": "application/json"})
    assert_safe_error(resp, status=413, code="REQUEST_TOO_LARGE")


async def test_oversized_declared_content_length_rejected(client, stack, set_limit):
    set_limit(max_request_body_bytes=1000)
    resp = await client.post("/api/rag/retrieve", content=b"{}" + b" " * 3000,
                             headers={"content-type": "application/json"})
    assert resp.status_code == 413


def test_llm_output_size_cap():
    from app.core.exceptions import AppError
    from app.core.output_guard import enforce_output_size
    from app.core.config import get_settings

    enforce_output_size("x" * 10)
    with pytest.raises(AppError) as e:
        enforce_output_size("x" * (get_settings().max_llm_output_chars + 1))
    assert str(e.value.code) == "LLM_OUTPUT_REJECTED"


# ---------------------------------------------------------------- rate limiting
async def test_rate_limit_applies_with_retry_after(client, stack, set_limit):
    set_limit(rate_limit_per_minute=3)
    codes = [(await client.post("/api/rag/retrieve", json={"document_id": "x", "question": "q"})).status_code
             for _ in range(3)]
    assert 429 not in codes
    resp = await client.post("/api/rag/retrieve", json={"document_id": "x", "question": "q"})
    assert_safe_error(resp, status=429, code="RATE_LIMITED")
    assert int(resp.headers["retry-after"]) >= 1


async def test_rate_limit_covers_all_expensive_routes(client, stack, set_limit):
    set_limit(rate_limit_per_minute=1)
    doc = "00000000-0000-0000-0000-000000000000"
    calls = [
        ("/api/rag/answer", {"document_id": doc, "question": "q"}),
        ("/api/questions/generate", {"document_id": doc, "number_of_questions": 1}),
        ("/api/evaluations/evaluate", {"document_id": doc, "question": "q", "student_answer": "a", "expected_answer": "e"}),
        ("/api/evaluations/evaluate/stream", {"document_id": doc, "question": "q", "student_answer": "a", "expected_answer": "e"}),
        (f"/api/documents/{doc}/search", {"query": "q"}),
    ]
    from app.core.rate_limit import limiter
    for url, body in calls:
        limiter.reset()
        await client.post(url, json=body)                      # consumes the single token
        second = await client.post(url, json=body)
        assert second.status_code == 429, url


async def test_rate_limit_zero_disables(client, stack, set_limit):
    set_limit(rate_limit_per_minute=0)
    for _ in range(20):
        assert (await client.post("/api/rag/retrieve", json={"document_id": "x", "question": "q"})).status_code != 429


async def test_forged_forwarded_for_does_not_evade_limit(client, stack, set_limit):
    set_limit(rate_limit_per_minute=2)
    last = None
    for i in range(3):
        last = await client.post("/api/rag/retrieve", json={"document_id": "x", "question": "q"},
                                 headers={"X-Forwarded-For": f"10.0.0.{i}"})
    assert last.status_code == 429


# ---------------------------------------------------------------- error leakage
async def test_unexpected_exception_is_generic_and_leaks_nothing(stack, monkeypatch):
    secret = "SECRET-DETAIL /home/user/project/x.py C:\\Users\\bob\\x.py Traceback"

    class Boom:
        async def retrieve(self, *a, **k):
            raise RuntimeError(secret)

    monkeypatch.setattr("app.api.rag._get_rag_service", lambda: Boom())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.post("/api/rag/retrieve", json={"document_id": "x", "question": "q"})
    assert resp.status_code == 500
    err = assert_safe_error(resp)
    assert "SECRET-DETAIL" not in resp.text and err["request_id"]


async def test_validation_error_never_echoes_submitted_values(client, stack):
    resp = await client.post("/api/evaluations/evaluate", json={
        "document_id": "not-a-uuid", "question": "q", "student_answer": "TOPSECRET-ANSWER-TEXT",
        "expected_answer": "e", "max_score": "NaN-TOPSECRET"})
    assert resp.status_code in (400, 422)
    assert "TOPSECRET" not in resp.text
    assert_safe_error(resp)


async def test_malformed_json_body_is_safe(client, stack):
    resp = await client.post("/api/rag/retrieve", content=b'{"document_id": ', headers={"content-type": "application/json"})
    assert resp.status_code in (400, 422)
    assert_safe_error(resp)


async def test_unknown_route_and_wrong_method_are_safe(client):
    for resp in (await client.get("/api/nope"), await client.get("/api/rag/retrieve")):
        assert resp.status_code in (404, 405)
        for m in BAD_MARKERS:
            assert m not in resp.text


# ---------------------------------------------------------------- HTTP headers / CORS
async def test_security_headers_present_on_api_and_errors(client):
    for resp in (await client.get("/health/live"), await client.get("/api/nope")):
        h = resp.headers
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in h["content-security-policy"]


async def test_docs_not_broken_by_csp(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200
    assert "default-src 'none'" not in resp.headers.get("content-security-policy", "")


async def test_cors_is_off_by_default(client):
    resp = await client.get("/health/live", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
    assert "access-control-allow-credentials" not in resp.headers


# ---------------------------------------------------------------- fail-fast configuration
@pytest.mark.parametrize("overrides", [
    {"max_pdf_pages": 0}, {"max_extracted_chars": 0}, {"max_chunks_per_document": -1},
    {"pdf_processing_timeout_s": 0}, {"max_request_body_bytes": 0}, {"max_llm_output_chars": 0},
    {"stream_max_duration_s": 0}, {"rate_limit_per_minute": -1}, {"admin_api_token": "short"},
    {"app_env": "production", "cors_allowed_origins": "*"}, {"app_env": "production", "log_level": "DEBUG"},
])
def test_insecure_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(**overrides)
