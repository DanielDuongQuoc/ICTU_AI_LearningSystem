"""Phase 16 - operational metrics: registry, /metrics endpoint, cardinality and privacy."""

from __future__ import annotations

import pytest

from app.core.metrics import STAGES, MetricsRegistry, metrics
from tests.fakes import Stack, sample_pdf_bytes


@pytest.fixture(autouse=True)
def _fresh_metrics():
    metrics.reset()
    yield
    metrics.reset()


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


def test_reset():
    reg = MetricsRegistry()
    reg.request_finished("GET", "/x", 500, 1.0)
    reg.error("X")
    reg.reset()
    text = reg.render()
    assert "module3_http_requests_total{" not in text and "module3_errors_total{" not in text


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


@pytest.mark.asyncio
async def test_metrics_not_part_of_openapi_contract(client):
    spec = (await client.get("/openapi.json")).json()
    assert "/metrics" not in spec["paths"]


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


@pytest.mark.asyncio
async def test_error_counter_counts_each_error_response(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.post("/api/rag/retrieve", json={"document_id": "11111111-1111-4111-8111-111111111111", "question": "  "})
    assert r.status_code in (400, 422)
    code = r.json()["error"]["code"]
    assert f'module3_errors_total{{code="{code}"}} 1' in (await client.get("/metrics")).text


@pytest.mark.asyncio
async def test_inflight_returns_to_zero_after_requests(client):
    for _ in range(3):
        await client.get("/health/live")
    assert "module3_http_requests_inflight 1" in (await client.get("/metrics")).text  # the /metrics call itself
