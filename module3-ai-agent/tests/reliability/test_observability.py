"""Phase 20 - observability: structured/redacted logs, stage metrics, slow-request hints, bounded cardinality,
request-id propagation, AI-version traceability. Offline."""

from __future__ import annotations

import json
import logging

import pytest

from app.core.config import get_settings
from app.core.logging import JsonFormatter, RedactionFilter, redact
from app.core.metrics import EVENTS, STAGES, metrics
from tests.fakes import Stack, sample_pdf_bytes

TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


@pytest.fixture(autouse=True)
def _env(monkeypatch):
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
@pytest.mark.parametrize("raw", [
    "Authorization: Bearer abc.def.ghi", "token=supersecret", "api_key=AKIA123456", "password=hunter2",
])
def test_secrets_are_redacted(raw):
    out = redact(raw)
    for leaked in ("abc.def.ghi", "supersecret", "AKIA123456", "hunter2"):
        assert leaked not in out
    assert "REDACTED" in out.upper() or "***" in out


def test_redaction_keeps_harmless_metric_fields():
    assert redact("first_token_ms=12 max_tokens=512") == "first_token_ms=12 max_tokens=512"


def test_redaction_filter_cleans_record_before_any_handler():
    rec = _record("calling with password=hunter2")
    assert RedactionFilter().filter(rec) is True
    assert "hunter2" not in rec.getMessage()


def test_json_formatter_has_standard_fields_and_lifts_key_values():
    rec = _record("request completed method=POST endpoint=/api/rag/retrieve status=200 duration_ms=12.5 error_code=-")
    data = json.loads(JsonFormatter().format(rec))
    for field in ("timestamp", "level", "service", "module", "request_id", "message"):
        assert field in data
    assert data["request_id"] == "rid-1" and data["level"] == "INFO"
    assert str(data.get("status")) == "200"


def test_json_formatter_redacts_secrets_in_message():
    rec = _record("failed token=abc123secret")
    RedactionFilter().filter(rec)
    assert "abc123secret" not in JsonFormatter().format(rec)


def test_log_format_setting_is_validated(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "xml")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_request_timeout_must_cover_a_model_call(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "120")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "30")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


# ---------------------------------------------------------------- metrics: bounded cardinality
def test_unknown_stage_and_event_names_are_ignored():
    metrics.observe_stage("doc-1234-abcd", 0.1)
    metrics.event("llm_request", "request-uuid")
    metrics.event("made_up", "ok")
    snap = metrics.snapshot()
    assert snap["stages"] == {} and snap["events"] == {}


def test_stage_and_event_allow_lists_have_no_dynamic_values():
    assert len(STAGES) <= 20 and sum(len(v) for v in EVENTS.values()) <= 40
    for name in STAGES | set(EVENTS):
        assert name.replace("_", "").isalpha()


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


@pytest.mark.asyncio
async def test_completed_log_carries_ai_version(client, caplog):
    with caplog.at_level(logging.INFO):
        await client.get("/health")
    done = [r.getMessage() for r in caplog.records if "config_version=" in r.getMessage()]
    assert done, "request completed log should include config_version (AI version trace)"
