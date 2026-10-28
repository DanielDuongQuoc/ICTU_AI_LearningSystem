"""Phase 21 - embedding-model compatibility of the vector store."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.repositories import chroma_repository
from app.repositories.chroma_repository import check_embedding_compat
from tests.fakes import LexicalEmbedding, Stack, sample_pdf_bytes

TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


class OtherEmbedding(LexicalEmbedding):
    model_name = "another-embedding:v2"


def _chunks():
    import tempfile
    from pathlib import Path

    from app.services.document_service import process_pdf

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.pdf"
        p.write_bytes(sample_pdf_bytes([TEXT]))
        return process_pdf(p, "doc-x", source_filename="a.pdf", content_type="application/pdf", max_bytes=10_000_000).chunks


@pytest.fixture()
def wired(monkeypatch):
    """Readiness check pointed at the Stack's in-memory collection, configured model = lexical-test-embedding."""
    stack = Stack()
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "lexical-test-embedding")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", stack.repo._collection_name)
    get_settings.cache_clear()
    monkeypatch.setattr(chroma_repository, "get_chroma_client", lambda: stack.repo._client)
    yield stack
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_new_chunks_are_tagged_with_their_embedding_model(wired):
    await wired.vector.index_chunks(chunks=_chunks(), document_id="doc-x", source="a.pdf")
    assert wired.repo.stored_embedding_models() == {"lexical-test-embedding"}
    assert check_embedding_compat() is True


@pytest.mark.asyncio
async def test_indexing_with_a_different_embedding_model_is_refused(wired):
    await wired.vector.index_chunks(chunks=_chunks(), document_id="doc-x", source="a.pdf")
    before = wired.repo.count()
    from app.services.vector_service import VectorService

    other = VectorService(embedding_service=OtherEmbedding(), chroma_repo=wired.repo)
    with pytest.raises(AppError) as e:
        await other.index_chunks(chunks=_chunks(), document_id="doc-y", source="b.pdf")
    assert e.value.code == ErrorCode.INDEXING_ERROR and "re-index" in e.value.message
    assert wired.repo.count() == before, "no incompatible vectors may be added"


@pytest.mark.asyncio
async def test_readiness_detects_model_change_after_release(wired, client, monkeypatch):
    await wired.vector.index_chunks(chunks=_chunks(), document_id="doc-x", source="a.pdf")

    async def up() -> bool:
        return True

    monkeypatch.setattr("app.api.health.check_ollama_health", up)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: True)
    assert (await client.get("/ready")).status_code == 200

    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")      # "new release" with another model
    get_settings.cache_clear()
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["services"]["embedding_index"] == "error"
    assert (await client.get("/health/live")).status_code == 200                # process stays alive


def test_legacy_chunks_without_tag_cannot_be_judged_and_do_not_block(wired):
    wired.repo.add_documents(ids=["legacy_1"], documents=["old text"], embeddings=[[0.1] * 512],
                             metadatas=[{"document_id": "legacy", "chunk_id": 1, "source": "old.pdf"}])
    assert wired.repo.stored_embedding_models() == set()
    assert check_embedding_compat() is True            # NOT VERIFIED territory: documented limitation


def test_empty_store_and_uninitialised_client_are_compatible(monkeypatch):
    def not_initialised():
        raise RuntimeError("not initialised")

    monkeypatch.setattr(chroma_repository, "get_chroma_client", not_initialised)
    assert check_embedding_compat() is True
