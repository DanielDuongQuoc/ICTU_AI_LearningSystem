"""Reliability: provider down/timeout, concurrency limits and backpressure, readiness and recovery,
graceful shutdown, observability (metrics, logs, request id), performance bounds."""

from __future__ import annotations

import asyncio
import httpx
import json
import logging
import pytest
import pytest_asyncio
import random
import re
import uuid
from httpx import ASGITransport, AsyncClient
from app.main import app, lifespan
from app.core.config import get_settings, Settings
from app.core.exceptions import AppError, ErrorCode, ERROR_HTTP_STATUS
from app.core.metrics import metrics, EVENTS, STAGES, MetricsRegistry
from tests.fakes import Stack, sample_pdf_bytes, gemini_body, gemini_llm, gemini_sse
from app.core.logging import JsonFormatter, RedactionFilter, redact
from scripts.smoke_test import run_smoke
from app.services.llm_service import GeminiLLMService
from typing import Any
from unittest.mock import MagicMock
from app.core.request_context import resolve_request_id
from app.repositories.chroma_repository import ChromaRepository
from app.services import embedding_service, llm_service, chunk_service
from app.utils.performance import PerformanceTimer
from app.core.concurrency import embedding_gate, llm_gate
from app.schemas.document import ExtractedPage
from app.services.chunk_service import _join_units, _metadata_for_pages, _overlap_units, _units_from_page


# ====================================================================================================
# shared reliability fixtures
# ====================================================================================================


@pytest_asyncio.fixture()
async def soft_client():
    """Like production: an unexpected exception becomes the catch-all 500 response instead of being re-raised
    into the test (httpx re-raises app exceptions by default after the response was already produced)."""
    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False),
                           base_url="http://testserver") as ac:
        yield ac


# ====================================================================================================
# concurrency and SSE under load
# ====================================================================================================


TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


EVAL = {"question": "What do zebracorn models learn from?", "student_answer": "They learn from data.",
        "expected_answer": "They learn patterns from data.", "max_score": 10}


@pytest.fixture()
def _env_concurrency_sse(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


async def _upload(client, text=TEXT):
    r = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([text]), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()["document_id"]


def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------- concurrency
@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_parallel_requests_on_same_document_are_consistent(client, monkeypatch):
    Stack().install(monkeypatch)
    doc = await _upload(client)
    rs = await asyncio.gather(
        *[client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"}) for _ in range(8)],
        *[client.post("/api/evaluations/evaluate", json={"document_id": doc, **EVAL}) for _ in range(6)],
    )
    assert [r.status_code for r in rs] == [200] * 14
    retrieved = [tuple(c["chunk_id"] if "chunk_id" in c else c["text"] for c in r.json()["chunks"]) for r in rs[:8]]
    assert len(set(retrieved)) == 1, "same query on same document must give the same sources under load"
    assert len({r.headers["x-request-id"] for r in rs}) == 14, "every request keeps its own request id"


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_concurrent_deletes_of_same_document_do_not_corrupt_or_crash(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)
    other = await _upload(client, "Quokkaplex networks describe database indexing. Quokkaplex indexes speed queries.")
    keep = stack.repo.count_by_document(other)
    rs = await asyncio.gather(*[client.delete(f"/api/documents/{doc}") for _ in range(4)])
    assert all(r.status_code in (200, 404) for r in rs) and any(r.status_code == 200 for r in rs)
    assert all("Traceback" not in r.text for r in rs)
    assert stack.repo.count_by_document(doc) == 0 and stack.repo.count_by_document(other) == keep > 0


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_concurrent_uploads_get_isolated_ids_and_no_duplicate_vectors(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    rs = await asyncio.gather(*[client.post("/api/documents/upload",
                                            files={"file": ("a.pdf", sample_pdf_bytes([TEXT]), "application/pdf")})
                                for _ in range(4)])
    # Upload backpressure (upload_slot) may reject the excess with 503 SERVICE_OVERLOADED: that is the
    # designed behaviour. Anything accepted must be isolated and complete; anything rejected stores nothing.
    ok = [r for r in rs if r.status_code in (200, 201)]
    rejected = [r for r in rs if r.status_code not in (200, 201)]
    assert ok and all(r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_OVERLOADED" for r in rejected)
    ids = [r.json()["document_id"] for r in ok]
    assert len(set(ids)) == len(ids)
    counts = {stack.repo.count_by_document(i) for i in ids}
    assert len(counts) == 1 and 0 not in counts
    assert stack.repo.count() == sum(counts) * len(ids), "rejected uploads left no vectors behind"


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_failure_of_one_concurrent_request_does_not_poison_others(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)
    real = stack.llm.generate_json
    state = {"n": 0}

    async def every_third_fails(*a, **k):
        state["n"] += 1
        if state["n"] % 3 == 0:
            raise AppError(ErrorCode.LLM_TIMEOUT, "Language model request timed out.")
        return await real(*a, **k)

    monkeypatch.setattr(stack.llm, "generate_json", every_third_fails)
    rs = await asyncio.gather(*[client.post("/api/evaluations/evaluate", json={"document_id": doc, **EVAL}) for _ in range(9)])
    codes = sorted(r.status_code for r in rs)
    assert codes.count(200) == 6 and all(c >= 500 for c in codes if c != 200)
    after = await client.post("/api/evaluations/evaluate", json={"document_id": doc, **EVAL})
    assert after.status_code in (200, 504, 503)       # still serving (deterministic: 10th call is not a multiple of 3)
    assert after.status_code == 200


# ---------------------------------------------------------------- SSE reliability
@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_sse_unexpected_exception_yields_single_error_then_done_no_traceback(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)

    async def boom(*a, **k):
        yield '{"score": 8'
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(stack.llm, "generate_stream", boom)
    r = await client.post("/api/evaluations/evaluate/stream", json={"document_id": doc, **EVAL})
    types = [e["type"] for e in _events(r.text)]
    assert types.count("error") == 1 and types[-1] == "done" and types.count("done") == 1 and "result" not in types
    assert "secret internal failure" not in r.text and "Traceback" not in r.text
    assert metrics.snapshot()["streams"].get("error", 0) >= 1


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_sse_budget_exhaustion_is_a_controlled_error_event(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)

    async def expired(*a, **k):
        raise AppError(ErrorCode.LLM_TIMEOUT, "The request time budget was exhausted.")
        yield ""  # pragma: no cover

    monkeypatch.setattr(stack.llm, "generate_stream", expired)
    r = await client.post("/api/evaluations/evaluate/stream", json={"document_id": doc, **EVAL})
    ev = _events(r.text)
    assert [e["type"] for e in ev][-1] == "done"
    assert [e for e in ev if e["type"] == "error"][0]["code"] == "OLLAMA_TIMEOUT"


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_sse_failure_then_reconnect_does_not_duplicate_generation(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)
    before = stack.llm.stream_calls
    r1 = await client.post("/api/evaluations/evaluate/stream", json={"document_id": doc, **EVAL})
    r2 = await client.post("/api/evaluations/evaluate/stream", json={"document_id": doc, **EVAL})     # "reconnect" = new request
    assert stack.llm.stream_calls - before == 2, "exactly one generation per stream request"
    for r in (r1, r2):
        types = [e["type"] for e in _events(r.text)]
        assert types.index("result") < types.index("done") and types[-1] == "done"


@pytest.mark.usefixtures("_env_concurrency_sse")
@pytest.mark.asyncio
async def test_sse_cancellation_releases_llm_slot(monkeypatch):
    """Client goes away mid-generation: the cancelled task must free its concurrency slot (no leak)."""
    from app.core.concurrency import llm_gate

    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("LLM_QUEUE_TIMEOUT_S", "0.2")
    get_settings.cache_clear()
    started = asyncio.Event()

    async def hold():
        async with llm_gate.slot():
            started.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(hold())
    await asyncio.wait_for(started.wait(), timeout=5)
    assert metrics.snapshot()["llm_active"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with llm_gate.slot():     # slot is free again; would raise SERVICE_OVERLOADED on a leak
        pass
    assert metrics.snapshot()["llm_active"] == 0


# ====================================================================================================
# observability
# ====================================================================================================


@pytest.fixture()
def _env_observability(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


def _record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord("app.test", logging.INFO, __file__, 1, msg, (), None)
    rec.request_id = "rid-1"
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


# ---------------------------------------------------------------- logging
@pytest.mark.usefixtures("_env_observability")
@pytest.mark.parametrize("raw", [
    "Authorization: Bearer abc.def.ghi", "token=supersecret", "api_key=AKIA123456", "password=hunter2",
])
def test_secrets_are_redacted(raw):
    out = redact(raw)
    for leaked in ("abc.def.ghi", "supersecret", "AKIA123456", "hunter2"):
        assert leaked not in out
    assert "REDACTED" in out.upper() or "***" in out


@pytest.mark.usefixtures("_env_observability")
def test_redaction_keeps_harmless_metric_fields():
    assert redact("first_token_ms=12 max_tokens=512") == "first_token_ms=12 max_tokens=512"


@pytest.mark.usefixtures("_env_observability")
def test_redaction_filter_cleans_record_before_any_handler():
    rec = _record("calling with password=hunter2")
    assert RedactionFilter().filter(rec) is True
    assert "hunter2" not in rec.getMessage()


@pytest.mark.usefixtures("_env_observability")
def test_json_formatter_has_standard_fields_and_lifts_key_values():
    rec = _record("request completed method=POST endpoint=/api/rag/retrieve status=200 duration_ms=12.5 error_code=-")
    data = json.loads(JsonFormatter().format(rec))
    for field in ("timestamp", "level", "service", "module", "request_id", "message"):
        assert field in data
    assert data["request_id"] == "rid-1" and data["level"] == "INFO"
    assert str(data.get("status")) == "200"


@pytest.mark.usefixtures("_env_observability")
def test_json_formatter_redacts_secrets_in_message():
    rec = _record("failed token=abc123secret")
    RedactionFilter().filter(rec)
    assert "abc123secret" not in JsonFormatter().format(rec)


@pytest.mark.usefixtures("_env_observability")
def test_log_format_setting_is_validated(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "xml")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


@pytest.mark.usefixtures("_env_observability")
def test_request_timeout_must_cover_a_model_call(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "120")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "30")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


# ---------------------------------------------------------------- metrics: bounded cardinality
@pytest.mark.usefixtures("_env_observability")
def test_unknown_stage_and_event_names_are_ignored():
    metrics.observe_stage("doc-1234-abcd", 0.1)
    metrics.event("llm_request", "request-uuid")
    metrics.event("made_up", "ok")
    snap = metrics.snapshot()
    assert snap["stages"] == {} and snap["events"] == {}


@pytest.mark.usefixtures("_env_observability")
def test_stage_and_event_allow_lists_have_no_dynamic_values():
    assert len(STAGES) <= 20 and sum(len(v) for v in EVENTS.values()) <= 40
    for name in STAGES | set(EVENTS):
        assert name.replace("_", "").isalpha()


@pytest.mark.usefixtures("_env_observability")
@pytest.mark.asyncio
async def test_metrics_output_never_contains_request_or_document_ids(client, monkeypatch):
    Stack().install(monkeypatch)
    up = await client.post("/api/documents/upload", files={"file": ("secret-name.pdf", sample_pdf_bytes([TEXT]), "application/pdf")})
    doc = up.json()["document_id"]
    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn secretquestion"},
                          headers={"X-Request-ID": "my-trace-12345"})
    assert r.headers["x-request-id"] == "my-trace-12345"
    text = (await client.get("/metrics")).text
    for leaked in (doc, "my-trace-12345", "secret-name", "secretquestion"):
        assert leaked not in text


@pytest.mark.usefixtures("_env_observability")
@pytest.mark.asyncio
async def test_pipeline_stages_are_measured_end_to_end(client, monkeypatch):
    Stack().install(monkeypatch)
    up = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([TEXT]), "application/pdf")})
    doc = up.json()["document_id"]
    await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})
    await client.post("/api/evaluations/evaluate", json={
        "document_id": doc, "question": "What do zebracorn models learn?", "student_answer": "data",
        "expected_answer": "patterns from data", "max_score": 10})
    snap = metrics.snapshot()
    for stage in ("pdf_extract", "chunking", "index_write", "chroma_query", "context_build", "validation"):
        assert snap["stages"].get(stage, (0, 0, []))[1] >= 1, f"stage {stage} not measured"
    assert snap["events"][("document_processing", "ok")] == 1 and snap["events"][("index", "ok")] == 1
    assert snap["events"][("rag_retrieval", "ok")] >= 1 and snap["events"][("evaluation", "ok")] == 1
    text = (await client.get("/metrics")).text
    assert "module3_events_total" in text and 'stage="pdf_extract"' in text


# ---------------------------------------------------------------- slow detection / correlation
@pytest.mark.usefixtures("_env_observability")
def test_slow_stage_warning_has_kind_and_no_payload(caplog, monkeypatch):
    monkeypatch.setenv("SLOW_STAGE_THRESHOLD_MS", "10")
    get_settings.cache_clear()
    with caplog.at_level(logging.WARNING, logger="app.slow"):
        metrics.observe_stage("chroma_query", 0.5)
        metrics.observe_stage("pdf_extract", 0.5)
        metrics.observe_stage("embedding", 0.001)           # under threshold -> silent
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("slow_retrieval") for m in msgs) and any(m.startswith("slow_pdf") for m in msgs)
    assert not any("embedding" in m for m in msgs)


@pytest.mark.usefixtures("_env_observability")
@pytest.mark.asyncio
async def test_slow_request_log_names_slowest_stage(client, monkeypatch, caplog):
    monkeypatch.setenv("SLOW_REQUEST_THRESHOLD_MS", "1")
    get_settings.cache_clear()
    Stack().install(monkeypatch)
    up = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([TEXT]), "application/pdf")})
    with caplog.at_level(logging.WARNING):
        await client.post("/api/rag/retrieve", json={"document_id": up.json()["document_id"], "question": "zebracorn data"})
    slow = [r.getMessage() for r in caplog.records if "slow" in r.getMessage().lower() and "slowest_stage=" in r.getMessage()]
    assert slow, "slow request warning with slowest_stage hint expected"
    assert "zebracorn" not in " ".join(slow)               # never the payload


# ---------------------------------------------------------------- request id + AI version
@pytest.mark.usefixtures("_env_observability")
@pytest.mark.asyncio
async def test_request_id_in_header_error_body_and_log(client, caplog, monkeypatch):
    Stack().install(monkeypatch)
    with caplog.at_level(logging.INFO):
        r = await client.post("/api/rag/retrieve", json={"document_id": "nope", "question": ""},
                              headers={"X-Request-ID": "trace-abc-1"})
    assert r.headers["x-request-id"] == "trace-abc-1"
    if r.status_code >= 400 and isinstance(r.json(), dict) and "error" in r.json():
        assert r.json()["error"]["request_id"] == "trace-abc-1"
    assert any("trace-abc-1" in rec.getMessage() or getattr(rec, "request_id", "") == "trace-abc-1"
               for rec in caplog.records)


@pytest.mark.usefixtures("_env_observability")
@pytest.mark.asyncio
async def test_completed_log_carries_ai_version(client, caplog):
    with caplog.at_level(logging.INFO):
        await client.get("/health")
    done = [r.getMessage() for r in caplog.records if "config_version=" in r.getMessage()]
    assert done, "request completed log should include config_version (AI version trace)"


# ====================================================================================================
# pipeline failure injection
# ====================================================================================================


@pytest.fixture()
def _env_pipeline_failure(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


async def _upload_pipeline_failure(client, text: str = TEXT, name: str = "a.pdf"):
    return await client.post("/api/documents/upload", files={"file": (name, sample_pdf_bytes([text]), "application/pdf")})


def _ev(name: str, outcome: str) -> int:
    return metrics.snapshot()["events"].get((name, outcome), 0)


# ---------------------------------------------------------------- ChromaDB read failures
@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("chroma down: /secret/path"), ConnectionError("refused"), ValueError("collection missing")])
async def test_chroma_query_failure_controlled_and_recovers(client, monkeypatch, exc):
    stack = Stack().install(monkeypatch)
    doc = (await _upload_pipeline_failure(client)).json()["document_id"]
    real = stack.repo.query
    state = {"fail": True}

    def flaky(*a, **k):
        if state["fail"]:
            raise exc
        return real(*a, **k)

    monkeypatch.setattr(stack.repo, "query", flaky)
    body = {"document_id": doc, "question": "zebracorn data"}
    r = await client.post("/api/rag/retrieve", json=body)
    assert r.status_code >= 500 and r.json()["error"]["request_id"]
    assert "secret/path" not in r.text and "Traceback" not in r.text
    assert "chunks" not in r.json()                                   # no invented sources
    assert _ev("rag_retrieval", "error") == 1
    state["fail"] = False
    ok = await client.post("/api/rag/retrieve", json=body)
    assert ok.status_code == 200 and ok.json()["chunks"]
    assert _ev("rag_retrieval", "ok") == 1


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_retrieval_ok_then_llm_failure_is_controlled_and_recovers(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = (await _upload_pipeline_failure(client)).json()["document_id"]
    real = stack.llm.generate_json
    state = {"fail": True}

    async def flaky(*a, **k):
        if state["fail"]:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API is not reachable.")
        return await real(*a, **k)

    monkeypatch.setattr(stack.llm, "generate_json", flaky)
    body = {"document_id": doc, "question": "What do zebracorn models learn?", "student_answer": "data",
            "expected_answer": "patterns from data", "max_score": 10}
    r = await client.post("/api/evaluations/evaluate", json=body)
    assert r.status_code == 503 and r.json()["error"]["code"] == "OLLAMA_UNAVAILABLE"
    assert "score" not in r.json()                                    # no fabricated grade
    assert _ev("evaluation", "error") == 1
    state["fail"] = False
    r = await client.post("/api/evaluations/evaluate", json=body)
    assert r.status_code == 200 and _ev("evaluation", "ok") == 1


# ---------------------------------------------------------------- ingestion: partial failure
@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_embedding_failure_leaves_nothing_and_recovers(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    real = stack.embedding.embed_documents
    state = {"fail": True}

    async def flaky(texts):
        if state["fail"]:
            raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "embedding down")
        return await real(texts)

    monkeypatch.setattr(stack.embedding, "embed_documents", flaky)
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "document_id" not in r.json() and stack.repo.count() == 0
    state["fail"] = False
    r = await _upload_pipeline_failure(client)
    assert r.status_code in (200, 201) and r.json()["status"] == "indexed" and stack.repo.count() > 0


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("disk I/O error"), OSError(28, "No space left on device")])
async def test_chroma_write_failure_after_partial_write_is_cleaned_up(client, monkeypatch, exc):
    """embedding OK -> Chroma write raises *after* storing the vectors: nothing may stay behind."""
    stack = Stack().install(monkeypatch)
    real = stack.repo.add_documents
    state = {"fail": True}

    def write_then_fail(*a, **k):
        out = real(*a, **k)
        if state["fail"]:
            raise exc
        return out

    monkeypatch.setattr(stack.repo, "add_documents", write_then_fail)
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "document_id" not in r.json()
    assert "No space" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0, "partial vectors must be removed, never reported as processed"
    assert _ev("index", "failed") == 1 and _ev("index", "ok") == 0
    state["fail"] = False
    assert (await _upload_pipeline_failure(client)).status_code in (200, 201)
    assert _ev("index", "ok") == 1


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_failed_cleanup_does_not_mask_original_error(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    real = stack.repo.add_documents

    def write_then_fail(*a, **k):
        real(*a, **k)
        raise RuntimeError("write failed")

    def delete_fails(*a, **k):
        raise RuntimeError("delete failed too")

    monkeypatch.setattr(stack.repo, "add_documents", write_then_fail)
    monkeypatch.setattr(stack.repo, "delete_document", delete_fails)
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "delete failed" not in r.text and r.json()["error"]["request_id"]


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_index_count_mismatch_is_not_reported_as_indexed(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    monkeypatch.setattr(stack.vector, "get_document_info", lambda doc: {"document_id": doc, "chunk_count": 0})
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "document_id" not in r.json()
    assert stack.repo.count() == 0 and _ev("index", "verification_failed") == 1


# ---------------------------------------------------------------- PDF parser / invalid input
@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_pdf_parser_exception_rejects_document_then_recovers(soft_client, monkeypatch):
    client = soft_client
    stack = Stack().install(monkeypatch)
    import app.api.documents as docs

    real = docs.process_pdf
    state = {"fail": True}

    def flaky(*a, **k):
        if state["fail"]:
            raise RuntimeError("mupdf crashed")
        return real(*a, **k)

    monkeypatch.setattr(docs, "process_pdf", flaky)
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "mupdf" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0
    state["fail"] = False
    assert (await _upload_pipeline_failure(client)).status_code in (200, 201)


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"not a pdf at all", b"%PDF-1.7\ngarbage"])
async def test_invalid_empty_corrupt_pdf_store_nothing(client, monkeypatch, payload):
    stack = Stack().install(monkeypatch)
    r = await client.post("/api/documents/upload", files={"file": ("x.pdf", payload, "application/pdf")})
    assert 400 <= r.status_code < 500 and stack.repo.count() == 0 and "Traceback" not in r.text


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_filesystem_error_on_temp_dir_is_controlled(soft_client, monkeypatch):
    client = soft_client
    stack = Stack().install(monkeypatch)

    def boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("app.api.documents.tempfile.mkdtemp", boom)
    r = await _upload_pipeline_failure(client)
    assert r.status_code >= 500 and "No space" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0


# ---------------------------------------------------------------- delete / idempotency / consistency
@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_delete_failure_keeps_data_then_retry_succeeds(soft_client, monkeypatch):
    client = soft_client
    stack = Stack().install(monkeypatch)
    doc = (await _upload_pipeline_failure(client)).json()["document_id"]
    before = stack.repo.count_by_document(doc)
    real = stack.repo.delete_document
    state = {"fail": True}

    def flaky(d):
        if state["fail"]:
            raise RuntimeError("delete exploded")
        return real(d)

    monkeypatch.setattr(stack.repo, "delete_document", flaky)
    r = await client.delete(f"/api/documents/{doc}")
    assert r.status_code >= 500 and "exploded" not in r.text
    assert stack.repo.count_by_document(doc) == before
    state["fail"] = False
    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    assert stack.repo.count_by_document(doc) == 0


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_delete_twice_second_is_clean_not_found(client, monkeypatch):
    Stack().install(monkeypatch)
    doc = (await _upload_pipeline_failure(client)).json()["document_id"]
    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    r = await client.delete(f"/api/documents/{doc}")
    assert r.status_code in (200, 404) and "Traceback" not in r.text


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_reindexing_same_document_id_does_not_duplicate_vectors(monkeypatch):
    stack = Stack()
    from app.services.document_service import process_pdf
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.pdf"
        p.write_bytes(sample_pdf_bytes([TEXT]))
        res = process_pdf(p, "doc-fixed", source_filename="a.pdf", content_type="application/pdf", max_bytes=10_000_000)
    await stack.vector.index_chunks(chunks=res.chunks, document_id="doc-fixed", source="a.pdf")
    first = stack.repo.count_by_document("doc-fixed")
    await stack.vector.index_chunks(chunks=res.chunks, document_id="doc-fixed", source="a.pdf")   # retry of indexing
    assert stack.repo.count_by_document("doc-fixed") == first > 0


@pytest.mark.usefixtures("_env_pipeline_failure")
@pytest.mark.asyncio
async def test_metadata_and_vectors_stay_consistent_through_failures(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = (await _upload_pipeline_failure(client)).json()["document_id"]
    info = (await client.get(f"/api/documents/{doc}")).json()
    assert info["chunks"] == stack.repo.count_by_document(doc) > 0
    await client.delete(f"/api/documents/{doc}")
    assert (await client.get(f"/api/documents/{doc}")).status_code == 404       # app view == vector store view
    assert stack.repo.count_by_document(doc) == 0


# ====================================================================================================
# readiness recovery
# ====================================================================================================


@pytest.mark.asyncio
async def test_low_disk_makes_ready_503_then_recovers_and_live_stays_200(client, monkeypatch):
    async def ok() -> bool:
        return True

    monkeypatch.setattr("app.api.health.check_provider_health", ok)
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


# ====================================================================================================
# provider restart via smoke flow
# ====================================================================================================


# --------------------------------------------------------------- provider restart through the smoke flow
@pytest.mark.asyncio
async def test_provider_restart_degrades_then_smoke_flow_passes(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    state = {"up": False}

    async def provider_ok() -> bool:
        return state["up"]

    monkeypatch.setattr("app.api.health.check_provider_health", provider_ok)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: True)

    down = await run_smoke(client, require_ready=True)                          # provider "restarting"
    assert [s for s, ok, _ in down if not ok] == ["ready"], down                  # fails at readiness, nothing crashes
    assert (await client.get("/health/live")).status_code == 200
    assert stack.llm.calls == []                                                 # readiness never generated anything

    state["up"] = True                                                           # provider back
    up = await run_smoke(client, require_ready=True)
    assert [s for s, ok, _ in up if not ok] == [], up
    assert {s for s, _, _ in up} >= {"health", "ready", "upload", "retrieve", "questions", "evaluate", "stream", "delete"}
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_request_failing_during_provider_outage_then_next_succeeds(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    real = stack.llm.generate_json
    state = {"down": True}

    async def flaky(*a, **k):
        if state["down"]:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API is not reachable.")
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


# ====================================================================================================
# LLM concurrency limits
# ====================================================================================================


CHAT_OK = gemini_body("hello")


@pytest.fixture()
def limits(monkeypatch):
    def apply(max_concurrency: int, queue_timeout: float = 0.2) -> None:
        monkeypatch.setenv("LLM_MAX_CONCURRENCY", str(max_concurrency))
        monkeypatch.setenv("LLM_QUEUE_TIMEOUT_S", str(queue_timeout))
        get_settings.cache_clear()

    metrics.reset()
    yield apply
    get_settings.cache_clear()
    metrics.reset()


def _service(handler) -> GeminiLLMService:
    return gemini_llm(handler, model="m")


def test_overload_error_is_503_and_documented():
    assert ERROR_HTTP_STATUS[ErrorCode.SERVICE_OVERLOADED] == 503
    assert AppError(ErrorCode.SERVICE_OVERLOADED, "x").status_code == 503


@pytest.mark.parametrize("env,value", [("LLM_MAX_CONCURRENCY", "-1"), ("LLM_MAX_CONCURRENCY", "x"),
                                       ("LLM_QUEUE_TIMEOUT_S", "0"), ("LLM_QUEUE_TIMEOUT_S", "-3")])
def test_invalid_limits_are_rejected(monkeypatch, env, value):
    from app.core.config import Settings

    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.asyncio
async def test_excess_call_is_rejected_then_slot_is_reusable(limits):
    limits(1, 0.2)
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    first = asyncio.create_task(svc.generate("s", "u"))
    await asyncio.sleep(0.05)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.SERVICE_OVERLOADED and exc.value.status_code == 503
    assert "module3_llm_rejected_total 1" in metrics.render()
    assert "module3_llm_waiting 0" in metrics.render()

    release.set()
    assert await first == "hello"
    assert await svc.generate("s", "u") == "hello"          # slot released after completion
    assert "module3_llm_active 0" in metrics.render()
    await svc.close()


@pytest.mark.asyncio
async def test_cancellation_releases_slot_and_does_not_affect_others(limits):
    limits(1, 1.0)
    block = asyncio.Event()
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            await block.wait()
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    victim = asyncio.create_task(svc.generate("s", "u"))
    await asyncio.sleep(0.05)
    waiter = asyncio.create_task(svc.generate("s", "u"))     # queued behind the victim
    await asyncio.sleep(0.05)
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim
    assert await asyncio.wait_for(waiter, 2) == "hello"       # queued request proceeds
    assert "module3_llm_active 0" in metrics.render()
    await svc.close()


@pytest.mark.asyncio
async def test_failure_inside_slot_releases_it(limits):
    limits(1, 0.2)
    state = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.LLM_UNAVAILABLE
    state["fail"] = False
    assert await svc.generate("s", "u") == "hello"
    await svc.close()


@pytest.mark.asyncio
async def test_limit_zero_means_unlimited(limits):
    limits(0, 0.2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    results = await asyncio.gather(*(svc.generate("s", "u") for _ in range(8)))
    assert results == ["hello"] * 8
    await svc.close()


@pytest.mark.asyncio
async def test_concurrent_burst_within_limit_all_succeed(limits):
    limits(2, 5.0)
    active = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    results = await asyncio.gather(*(svc.generate("s", "u") for _ in range(10)))
    assert results == ["hello"] * 10 and active["peak"] <= 2
    await svc.close()


@pytest.mark.asyncio
async def test_stream_holds_slot_until_closed(limits):
    limits(1, 0.2)
    release = asyncio.Event()

    async def body():
        yield gemini_sse("hi")
        await release.wait()
        yield gemini_sse("!")

    def handler(request: httpx.Request) -> httpx.Response:
        if "streamGenerateContent" in request.url.path:
            return httpx.Response(200, content=body())
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    stream = svc.generate_stream("s", "u")
    assert await stream.__anext__() == "hi"                 # slot now held by the open stream
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.SERVICE_OVERLOADED

    await stream.aclose()                                    # client disconnect / cancel
    assert "module3_llm_active 0" in metrics.render()
    assert await svc.generate("s", "u") == "hello"           # slot is free again
    release.set()
    await svc.close()


# ====================================================================================================
# dependency failure and recovery
# ====================================================================================================


TEXT_A = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


TEXT_B = "Quokkaplex networks describe database indexing. Quokkaplex indexes speed up queries."


@pytest.fixture()
def _default_limits_dependency_recovery(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _upload_dependency_recovery(client, text: str, name: str = "a.pdf") -> str:
    r = await client.post("/api/documents/upload", files={"file": (name, sample_pdf_bytes([text]), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()["document_id"]


def _events_dependency_recovery(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def _eval_body(doc: str) -> dict[str, Any]:
    return {"document_id": doc, "question": "What do zebracorn models learn from?",
            "student_answer": "They learn from data.", "expected_answer": "They learn patterns from data.",
            "max_score": 10}


# ------------------------------------------------------------------ readiness follows dependencies
@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_readiness_tracks_provider_and_chroma_recovery(client, monkeypatch):
    state = {"provider": False, "chroma": True}

    async def provider() -> bool:
        return state["provider"]

    monkeypatch.setattr("app.api.health.check_provider_health", provider)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: state["chroma"])

    assert (await client.get("/ready")).status_code == 503
    assert (await client.get("/health/live")).status_code == 200      # process stays alive
    assert (await client.get("/health")).json()["status"] == "degraded"
    state["provider"] = True
    assert (await client.get("/ready")).status_code == 200
    state["chroma"] = False
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["services"]["chroma"] == "error"
    state["chroma"] = True
    assert (await client.get("/ready")).status_code == 200


@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_health_never_calls_llm_generation(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    await client.get("/health")
    await client.get("/ready")
    assert stack.llm.calls == []


# ------------------------------------------------------------------ ChromaDB failure + recovery via API
@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_chroma_failure_returns_structured_error_then_recovers(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload_dependency_recovery(client, TEXT_A)
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
@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_stream_error_event_then_next_stream_succeeds(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload_dependency_recovery(client, TEXT_A)
    original = stack.llm.generate_stream
    state = {"fail": True}

    async def flaky(*a, **k):
        if state["fail"]:
            yield '{"score": 8, "corr'
            raise AppError(ErrorCode.LLM_TIMEOUT, "Language model request timed out.")
        async for piece in original(*a, **k):
            yield piece

    monkeypatch.setattr(stack.llm, "generate_stream", flaky)
    r = await client.post("/api/evaluations/evaluate/stream", json=_eval_body(doc))
    ev = _events_dependency_recovery(r.text)
    types = [e["type"] for e in ev]
    assert r.status_code == 200 and types.count("error") == 1 and types.count("result") == 0
    assert types[-1] == "done" and types.count("done") == 1
    assert [e for e in ev if e["type"] == "error"][0]["code"] == "OLLAMA_TIMEOUT"
    assert "Traceback" not in r.text

    state["fail"] = False
    ev = _events_dependency_recovery((await client.post("/api/evaluations/evaluate/stream", json=_eval_body(doc))).text)
    types = [e["type"] for e in ev]
    assert types.count("result") == 1 and types[-1] == "done" and "error" not in types
    assert types.index("result") < types.index("done")


# ------------------------------------------------------------------ document lifecycle / integrity
@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_delete_one_document_leaves_other_intact(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    a, b = await _upload_dependency_recovery(client, TEXT_A, "a.pdf"), await _upload_dependency_recovery(client, TEXT_B, "b.pdf")
    before_b = stack.repo.count_by_document(b)
    assert (await client.delete(f"/api/documents/{a}")).status_code == 200
    assert stack.repo.count_by_document(a) == 0 and stack.repo.count_by_document(b) == before_b > 0
    r = await client.post("/api/rag/retrieve", json={"document_id": b, "question": "quokkaplex queries"})
    assert r.status_code == 200 and all(c["document_id"] == b for c in r.json()["chunks"])


@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_failed_ingestion_leaves_no_false_success(client, monkeypatch):
    stack = Stack().install(monkeypatch)

    async def broken(texts):
        raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "embedding down")

    monkeypatch.setattr(stack.embedding, "embed_documents", broken)
    r = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([TEXT_A]), "application/pdf")})
    assert r.status_code >= 400 and "document_id" not in r.json()
    assert stack.repo.count() == 0                       # nothing half-indexed


@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_invalid_upload_stores_nothing(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    r = await client.post("/api/documents/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400 and stack.repo.count() == 0


@pytest.mark.usefixtures("_default_limits_dependency_recovery")
@pytest.mark.asyncio
async def test_duplicate_upload_follows_current_behaviour(client, monkeypatch):
    """Documented behaviour: uploads are not idempotent; each gets its own isolated document_id."""
    stack = Stack().install(monkeypatch)
    d1, d2 = await _upload_dependency_recovery(client, TEXT_A), await _upload_dependency_recovery(client, TEXT_A)
    assert d1 != d2
    assert stack.repo.count_by_document(d1) == stack.repo.count_by_document(d2) > 0
    await client.delete(f"/api/documents/{d1}")
    assert stack.repo.count_by_document(d2) > 0


# ====================================================================================================
# error handling and request ids
# ====================================================================================================


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
        (ErrorCode.LLM_UNAVAILABLE, 503),
        (ErrorCode.LLM_TIMEOUT, 504),
        (ErrorCode.LLM_MODEL_NOT_FOUND, 503),
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


# ====================================================================================================
# startup / graceful shutdown
# ====================================================================================================


class _Closable:
    def __init__(self, fail: bool = False) -> None:
        self.closed = False
        self.fail = fail

    async def close(self) -> None:
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


def _no_deps(monkeypatch, *, chroma_ok: bool = True, provider_ok: bool = True):
    def init():
        if not chroma_ok:
            raise RuntimeError("chroma unavailable")

    async def provider() -> bool:
        return provider_ok

    monkeypatch.setattr("app.main.init_chroma", init)
    monkeypatch.setattr("app.services.provider_checker.check_provider_health", provider)


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
@pytest.mark.parametrize("chroma_ok,provider_ok", [(False, True), (True, False), (False, False)])
async def test_startup_degrades_instead_of_crashing_when_dependencies_are_down(monkeypatch, chroma_ok, provider_ok):
    _no_deps(monkeypatch, chroma_ok=chroma_ok, provider_ok=provider_ok)
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


# ====================================================================================================
# metrics
# ====================================================================================================


@pytest.fixture()
def _fresh_metrics_operational_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
def test_registry_counts_histogram_and_inflight():
    reg = MetricsRegistry()
    reg.request_started()
    assert "module3_http_requests_inflight 1" in reg.render()
    reg.request_finished("GET", "/health", 200, 0.07)
    text = reg.render()
    assert 'module3_http_requests_total{method="GET",route="/health",status="200"} 1' in text
    assert 'duration_seconds_bucket{method="GET",route="/health",le="0.05"} 0' in text
    assert 'duration_seconds_bucket{method="GET",route="/health",le="0.1"} 1' in text
    assert 'route="/health",le="+Inf"} 1' in text
    assert "module3_http_requests_inflight 0" in text


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
def test_unknown_stage_and_outcome_cannot_create_series():
    reg = MetricsRegistry()
    reg.observe_stage("user-controlled-stage", 1.0)
    reg.stream_finished("some free text")
    text = reg.render()
    assert "user-controlled-stage" not in text
    assert 'module3_streams_total{outcome="error"} 1' in text
    for stage in STAGES:
        reg.observe_stage(stage, 0.2)
    assert all(f'stage="{s}"' in reg.render() for s in STAGES)


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
def test_reset():
    reg = MetricsRegistry()
    reg.request_finished("GET", "/x", 500, 1.0)
    reg.error("X")
    reg.reset()
    text = reg.render()
    assert "module3_http_requests_total{" not in text and "module3_errors_total{" not in text


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
@pytest.mark.asyncio
async def test_metrics_endpoint_route_templates_error_codes_no_leak(client, monkeypatch):
    Stack().install(monkeypatch)
    await client.get("/health/live")
    await client.get("/api/documents/not-a-real-id-123")      # error response
    await client.get("/definitely-not-a-route")
    r = await client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert 'route="/health/live"' in body
    assert 'route="/api/documents/{document_id}"' in body
    assert 'route="unmatched"' in body
    assert 'module3_errors_total{code="NOT_FOUND"}' in body or "module3_errors_total{code=" in body
    assert "not-a-real-id-123" not in body and "definitely-not-a-route" not in body


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
@pytest.mark.asyncio
async def test_metrics_not_part_of_openapi_contract(client):
    spec = (await client.get("/openapi.json")).json()
    assert "/metrics" not in spec["paths"]


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
@pytest.mark.asyncio
async def test_pipeline_stage_metrics_and_no_content_leak(client, monkeypatch):
    Stack().install(monkeypatch)
    secret_text = "Zebracorn confidential-passage-ABC models learn patterns from data."
    up = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes([secret_text]), "application/pdf")})
    doc = up.json()["document_id"]
    await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "student-secret-question zebracorn data"})
    body = (await client.get("/metrics")).text
    assert 'module3_stage_duration_seconds_count{stage="chroma_query"}' in body
    assert doc not in body
    assert "confidential-passage-ABC" not in body and "student-secret-question" not in body


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
@pytest.mark.asyncio
async def test_error_counter_counts_each_error_response(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.post("/api/rag/retrieve", json={"document_id": "11111111-1111-4111-8111-111111111111", "question": "  "})
    assert r.status_code in (400, 422)
    code = r.json()["error"]["code"]
    assert f'module3_errors_total{{code="{code}"}} 1' in (await client.get("/metrics")).text


@pytest.mark.usefixtures("_fresh_metrics_operational_metrics")
@pytest.mark.asyncio
async def test_inflight_returns_to_zero_after_requests(client):
    for _ in range(3):
        await client.get("/health/live")
    assert "module3_http_requests_inflight 1" in (await client.get("/metrics")).text  # the /metrics call itself


# ====================================================================================================
# request logging
# ====================================================================================================


SECRET_ANSWER = "STUDENT-ANSWER-SENTINEL-8841"


SECRET_QUESTION = "QUESTION-SENTINEL-5520 about zebracorn?"


SECRET_PDF = "Zebracorn PDF-CONTENT-SENTINEL-3309 models learn patterns from data."


@pytest.fixture()
def _no_llm_cap_request_logging(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("_no_llm_cap_request_logging")
@pytest.mark.asyncio
async def test_client_request_id_is_echoed_and_logged(client, caplog):
    caplog.set_level(logging.INFO)
    r = await client.get("/health/live", headers={"X-Request-ID": "ops-req-12345"})
    assert r.headers["x-request-id"] == "ops-req-12345"
    assert any("request_id=ops-req-12345" in rec.getMessage() and "completed" in rec.getMessage()
               for rec in caplog.records)


@pytest.mark.usefixtures("_no_llm_cap_request_logging")
@pytest.mark.asyncio
async def test_error_body_header_and_log_share_one_request_id(client, monkeypatch, caplog):
    Stack().install(monkeypatch)
    caplog.set_level(logging.INFO)
    r = await client.get("/api/documents/zzz-unknown")
    rid = r.headers["x-request-id"]
    assert r.json()["error"]["request_id"] == rid
    assert any(f"request_id={rid}" in rec.getMessage() for rec in caplog.records)


@pytest.mark.usefixtures("_no_llm_cap_request_logging")
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


@pytest.mark.usefixtures("_no_llm_cap_request_logging")
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


@pytest.mark.usefixtures("_no_llm_cap_request_logging")
def test_log_level_setting(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert Settings(_env_file=None).log_level == "warning"
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


# ====================================================================================================
# timing helpers
# ====================================================================================================


def test_timer_measures_non_negative_elapsed():
    timer = PerformanceTimer()
    timer.start()
    timer.stop()
    assert timer.elapsed_ms() >= 0


def test_timer_context_manager():
    with PerformanceTimer() as timer:
        sum(range(1000))
    assert timer.elapsed_ms() >= 0


def test_timer_elapsed_while_running_is_non_negative():
    timer = PerformanceTimer()
    timer.start()
    assert timer.elapsed_ms() >= 0


def test_measure_records_named_laps():
    timer = PerformanceTimer()
    with timer.measure("rag"):
        pass
    with timer.measure("llm"):
        pass
    assert set(timer.laps) == {"rag", "llm"}
    assert all(v >= 0 for v in timer.laps.values())


def test_measure_records_lap_even_on_exception():
    timer = PerformanceTimer()
    with pytest.raises(ValueError):
        with timer.measure("boom"):
            raise ValueError("x")
    assert "boom" in timer.laps


def test_stop_before_start_raises():
    with pytest.raises(Exception):
        PerformanceTimer().stop()


# ── Streaming metrics  ────────────────────────────────────
@pytest.mark.asyncio
async def test_stream_logs_duration_and_token_metrics(caplog):
    import logging

    from tests.test_evaluation import DOC_A, default_hits, make_llm, make_stack
    from tests.test_evaluation import EXPECTED, QUESTION, STUDENT, good_stream, set_stream

    llm = make_llm()
    set_stream(llm, good_stream())
    svc, _ = make_stack(default_hits(), llm)
    with caplog.at_level(logging.INFO):
        _ = [f async for f in svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)]
    line = next(r.message for r in caplog.records if r.message.startswith("Evaluation stream |"))
    for key in ("first_token_latency_ms=", "last_token_ms=", "stream_duration_ms=",
                "token_count=", "total_latency_ms="):
        assert key in line
    assert "stream_duration_ms=null" not in line
    assert STUDENT not in line


# ====================================================================================================
# performance and capacity bounds
# ====================================================================================================


@pytest.fixture()
def _settings_cache_perf():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ------------------------------------------------------------------ chunking: faster, identical output
def _reference_chunks(pages, size, overlap):
    """The pre-Phase-22 selection loop (rebuilds the candidate string for every unit)."""
    units = []
    for page in pages:
        units.extend(_units_from_page(page, size))
    out, index, previous = [], 0, []
    while index < len(units):
        prefix = _overlap_units(previous, overlap) if out else []
        prefix_len = len(_join_units(prefix)) if prefix else 0
        budget = size if not prefix else max(size - prefix_len, 1)
        selected, cursor = [], index
        while cursor < len(units):
            candidate = _join_units(selected + [units[cursor]])
            if selected and len(candidate) > budget:
                break
            selected.append(units[cursor])
            cursor += 1
            if len(candidate) >= budget:
                break
        if not selected:
            selected, cursor = [units[index]], index + 1
        combined = prefix + selected
        meta = _metadata_for_pages([u.page for u in combined], "s")
        out.append((_join_units(combined), meta.page, meta.pages))
        previous, index = selected, cursor
    return out


def _random_pages(rng: random.Random, n_pages: int) -> list[ExtractedPage]:
    words = "alpha beta gamma delta epsilon model data train loss layer".split()
    pages = []
    for p in range(1, n_pages + 1):
        paras = []
        for _ in range(rng.randint(1, 5)):
            sentences = []
            for _ in range(rng.randint(1, 12)):
                s = " ".join(rng.choice(words) for _ in range(rng.randint(3, 40)))
                sentences.append(s + rng.choice([".", "!", "?", ""]))
            paras.append(" ".join(sentences))
        if rng.random() < 0.3:
            paras.insert(0, "1.2 Section Title")
        pages.append(ExtractedPage(page=p, text="\n\n".join(paras)))
    return pages


@pytest.mark.usefixtures("_settings_cache_perf")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("size,overlap", [(800, 150), (200, 40), (60, 10), (1500, 0)])
def test_optimised_chunking_matches_previous_algorithm(seed, size, overlap):
    pages = _random_pages(random.Random(seed), 6)
    new = chunk_service.create_chunks(pages, "doc", "s", chunk_size=size, chunk_overlap=overlap)
    got = [(c.text, c.metadata.page, c.metadata.pages) for c in new]
    assert got == _reference_chunks(pages, size, overlap)
    assert len({c.chunk_id for c in new}) == len(new)


@pytest.mark.usefixtures("_settings_cache_perf")
def test_chunking_scales_roughly_linearly_with_input():
    """Guards against the old quadratic loop: 8x the text must not cost anywhere near 64x (generous bound)."""
    import time

    def run(n):
        pages = [ExtractedPage(page=p, text=" ".join(["word"] * 330)) for p in range(1, n + 1)]
        t = time.perf_counter()
        chunk_service.create_chunks(pages, "d", "s")
        return time.perf_counter() - t

    run(4)  # warm up
    small, large = min(run(10) for _ in range(3)), min(run(80) for _ in range(3))
    assert large < small * 8 * 4, f"chunking looks super-linear: 10 pages {small:.4f}s, 80 pages {large:.4f}s"


# ------------------------------------------------------------------ embedding gate
@pytest.mark.usefixtures("_settings_cache_perf")
@pytest.mark.asyncio
async def test_unlimited_embedding_gate_does_not_block(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "0")
    get_settings.cache_clear()
    async with embedding_gate.slot():
        async with embedding_gate.slot():
            pass


# ------------------------------------------------------------------ configuration limits
@pytest.mark.usefixtures("_settings_cache_perf")
@pytest.mark.parametrize("field,value", [
    ("embedding_batch_size", 0), ("embedding_batch_size", -1), ("max_concurrent_embeddings", -1),
    ("embedding_queue_timeout_s", 0),
])
def test_new_limits_are_validated(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.usefixtures("_settings_cache_perf")
def test_new_limit_defaults_are_sane():
    s = Settings(_env_file=None)
    assert (s.embedding_batch_size, s.max_concurrent_embeddings) == (64, 4)


# ------------------------------------------------------------------ load / burst through the real API (fakes behind it)
TEXT_perf = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


async def _upload_perf(client):
    r = await client.post("/api/documents/upload",
                          files={"file": ("a.pdf", sample_pdf_bytes([TEXT_perf]), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()["document_id"]


@pytest.mark.usefixtures("_settings_cache_perf")
@pytest.mark.asyncio
@pytest.mark.parametrize("n", [10, 20, 60])
async def test_burst_of_retrievals_succeeds_with_isolated_request_ids(client, monkeypatch, n):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    Stack().install(monkeypatch)
    doc = await _upload_perf(client)
    rs = await asyncio.gather(*[client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})
                                for _ in range(n)])
    assert [r.status_code for r in rs] == [200] * n
    assert len({r.headers["x-request-id"] for r in rs}) == n


@pytest.mark.usefixtures("_settings_cache_perf")
@pytest.mark.asyncio
async def test_llm_burst_over_capacity_is_rejected_cleanly_then_recovers(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("LLM_QUEUE_TIMEOUT_S", "0.05")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    doc = await _upload_perf(client)
    payload = {"document_id": doc, "question": "What do zebracorn models learn from?",
               "student_answer": "They learn from data.", "expected_answer": "They learn patterns from data.",
               "max_score": 10}
    original = stack.llm.generate_json

    async def slow(*a, **kw):  # the real GeminiLLMService takes the slot inside its call; the fake does not
        async with llm_gate.slot():
            await asyncio.sleep(0.25)
            return await original(*a, **kw)

    monkeypatch.setattr(stack.llm, "generate_json", slow)
    rs = await asyncio.gather(*[client.post("/api/evaluations/evaluate", json=payload) for _ in range(6)])
    codes = sorted(r.status_code for r in rs)
    assert set(codes) <= {200, 503} and 200 in codes and 503 in codes, codes
    assert all("Traceback" not in r.text for r in rs)
    monkeypatch.setattr(stack.llm, "generate_json", original)
    again = await client.post("/api/evaluations/evaluate", json=payload)
    assert again.status_code == 200, "no permanent broken state after overload"
