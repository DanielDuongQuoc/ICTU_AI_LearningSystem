"""Phase 20 - ChromaDB / embedding / PDF / filesystem failure injection, partial failure, idempotency, recovery.

Real in-memory ChromaDB + fake embedding + recording LLM (tests.fakes). Nothing here needs Ollama.
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from tests.fakes import Stack, sample_pdf_bytes

TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


async def _upload(client, text: str = TEXT, name: str = "a.pdf"):
    return await client.post("/api/documents/upload", files={"file": (name, sample_pdf_bytes([text]), "application/pdf")})


def _ev(name: str, outcome: str) -> int:
    return metrics.snapshot()["events"].get((name, outcome), 0)


# ---------------------------------------------------------------- ChromaDB read failures
@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("chroma down: /secret/path"), ConnectionError("refused"), ValueError("collection missing")])
async def test_chroma_query_failure_controlled_and_recovers(client, monkeypatch, exc):
    stack = Stack().install(monkeypatch)
    doc = (await _upload(client)).json()["document_id"]
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


@pytest.mark.asyncio
async def test_retrieval_ok_then_llm_failure_is_controlled_and_recovers(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = (await _upload(client)).json()["document_id"]
    real = stack.llm.generate_json
    state = {"fail": True}

    async def flaky(*a, **k):
        if state["fail"]:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.")
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
    r = await _upload(client)
    assert r.status_code >= 500 and "document_id" not in r.json() and stack.repo.count() == 0
    state["fail"] = False
    r = await _upload(client)
    assert r.status_code in (200, 201) and r.json()["status"] == "indexed" and stack.repo.count() > 0


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
    r = await _upload(client)
    assert r.status_code >= 500 and "document_id" not in r.json()
    assert "No space" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0, "partial vectors must be removed, never reported as processed"
    assert _ev("index", "failed") == 1 and _ev("index", "ok") == 0
    state["fail"] = False
    assert (await _upload(client)).status_code in (200, 201)
    assert _ev("index", "ok") == 1


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
    r = await _upload(client)
    assert r.status_code >= 500 and "delete failed" not in r.text and r.json()["error"]["request_id"]


@pytest.mark.asyncio
async def test_index_count_mismatch_is_not_reported_as_indexed(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    monkeypatch.setattr(stack.vector, "get_document_info", lambda doc: {"document_id": doc, "chunk_count": 0})
    r = await _upload(client)
    assert r.status_code >= 500 and "document_id" not in r.json()
    assert stack.repo.count() == 0 and _ev("index", "verification_failed") == 1


# ---------------------------------------------------------------- PDF parser / invalid input
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
    r = await _upload(client)
    assert r.status_code >= 500 and "mupdf" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0
    state["fail"] = False
    assert (await _upload(client)).status_code in (200, 201)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"not a pdf at all", b"%PDF-1.7\ngarbage"])
async def test_invalid_empty_corrupt_pdf_store_nothing(client, monkeypatch, payload):
    stack = Stack().install(monkeypatch)
    r = await client.post("/api/documents/upload", files={"file": ("x.pdf", payload, "application/pdf")})
    assert 400 <= r.status_code < 500 and stack.repo.count() == 0 and "Traceback" not in r.text


@pytest.mark.asyncio
async def test_filesystem_error_on_temp_dir_is_controlled(soft_client, monkeypatch):
    client = soft_client
    stack = Stack().install(monkeypatch)

    def boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("app.api.documents.tempfile.mkdtemp", boom)
    r = await _upload(client)
    assert r.status_code >= 500 and "No space" not in r.text and "Traceback" not in r.text
    assert stack.repo.count() == 0


# ---------------------------------------------------------------- delete / idempotency / consistency
@pytest.mark.asyncio
async def test_delete_failure_keeps_data_then_retry_succeeds(soft_client, monkeypatch):
    client = soft_client
    stack = Stack().install(monkeypatch)
    doc = (await _upload(client)).json()["document_id"]
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


@pytest.mark.asyncio
async def test_delete_twice_second_is_clean_not_found(client, monkeypatch):
    Stack().install(monkeypatch)
    doc = (await _upload(client)).json()["document_id"]
    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    r = await client.delete(f"/api/documents/{doc}")
    assert r.status_code in (200, 404) and "Traceback" not in r.text


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


@pytest.mark.asyncio
async def test_metadata_and_vectors_stay_consistent_through_failures(client, monkeypatch):
    stack = Stack().install(monkeypatch)
    doc = (await _upload(client)).json()["document_id"]
    info = (await client.get(f"/api/documents/{doc}")).json()
    assert info["chunks"] == stack.repo.count_by_document(doc) > 0
    await client.delete(f"/api/documents/{doc}")
    assert (await client.get(f"/api/documents/{doc}")).status_code == 404       # app view == vector store view
    assert stack.repo.count_by_document(doc) == 0
