"""Phase 20 - concurrency on shared documents/LLM and SSE reliability (offline)."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from tests.fakes import Stack, sample_pdf_bytes

TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."
EVAL = {"question": "What do zebracorn models learn from?", "student_answer": "They learn from data.",
        "expected_answer": "They learn patterns from data.", "max_score": 10}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
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


@pytest.mark.asyncio
async def test_failure_of_one_concurrent_request_does_not_poison_others(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)
    real = stack.llm.generate_json
    state = {"n": 0}

    async def every_third_fails(*a, **k):
        state["n"] += 1
        if state["n"] % 3 == 0:
            raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.")
        return await real(*a, **k)

    monkeypatch.setattr(stack.llm, "generate_json", every_third_fails)
    rs = await asyncio.gather(*[client.post("/api/evaluations/evaluate", json={"document_id": doc, **EVAL}) for _ in range(9)])
    codes = sorted(r.status_code for r in rs)
    assert codes.count(200) == 6 and all(c >= 500 for c in codes if c != 200)
    after = await client.post("/api/evaluations/evaluate", json={"document_id": doc, **EVAL})
    assert after.status_code in (200, 504, 503)       # still serving (deterministic: 10th call is not a multiple of 3)
    assert after.status_code == 200


# ---------------------------------------------------------------- SSE reliability
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


@pytest.mark.asyncio
async def test_sse_budget_exhaustion_is_a_controlled_error_event(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)

    async def expired(*a, **k):
        raise AppError(ErrorCode.OLLAMA_TIMEOUT, "The request time budget was exhausted.")
        yield ""  # pragma: no cover

    monkeypatch.setattr(stack.llm, "generate_stream", expired)
    r = await client.post("/api/evaluations/evaluate/stream", json={"document_id": doc, **EVAL})
    ev = _events(r.text)
    assert [e["type"] for e in ev][-1] == "done"
    assert [e for e in ev if e["type"] == "error"][0]["code"] == "OLLAMA_TIMEOUT"


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
