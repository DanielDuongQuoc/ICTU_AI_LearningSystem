"""Phase 3 — Vector service tests.

Integration-level tests that wire EmbeddingService (mocked) →
VectorService → ChromaRepository (real, in-memory).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.repositories.chroma_repository import ChromaRepository
from app.schemas.document import Chunk, ChunkMetadata
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


# ── Fixtures ────────────────────────────────────────────────────────

DIM = 5


def _normalise(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


def _fake_embedding(seed: float = 0.0) -> list[float]:
    return _normalise([seed + i * 0.1 for i in range(DIM)])


@pytest.fixture()
def chroma_client():
    return chromadb.Client(ChromaSettings(anonymized_telemetry=False))


@pytest.fixture()
def repo(chroma_client) -> ChromaRepository:
    return ChromaRepository(
        client=chroma_client, collection_name=f"vs_test_{uuid.uuid4().hex[:12]}"
    )


@pytest.fixture()
def mock_embedding_service():
    """Return an EmbeddingService with mocked embed methods."""
    service = EmbeddingService(
        base_url="http://fake:11434",
        model="test-embed",
        timeout=10,
    )
    return service


@pytest.fixture()
def vector_service(mock_embedding_service, repo) -> VectorService:
    return VectorService(embedding_service=mock_embedding_service, chroma_repo=repo)


def _make_chunks(document_id: str, texts: list[str]) -> list[Chunk]:
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"c{i}",
                document_id=document_id,
                text=text,
                metadata=ChunkMetadata(source="test.pdf", page=i + 1),
            )
        )
    return chunks


# ── Tests ───────────────────────────────────────────────────────────


class TestIndexChunks:
    @pytest.mark.asyncio
    async def test_index_and_count(self, vector_service: VectorService, mock_embedding_service) -> None:
        chunks = _make_chunks("doc1", ["alpha", "beta", "gamma"])
        embeddings = [_fake_embedding(float(i)) for i in range(3)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            stats = await vector_service.index_chunks(chunks, "doc1", "test.pdf")

        assert stats["document_id"] == "doc1"
        assert stats["chunk_count"] == 3
        assert stats["embedding_time_ms"] >= 0
        assert stats["chroma_insert_time_ms"] >= 0
        assert stats["total_time_ms"] >= 0
        assert stats["model_name"] == "test-embed"

        assert vector_service.count() == 3
        assert vector_service.count_by_document("doc1") == 3

    @pytest.mark.asyncio
    async def test_index_empty_chunks(self, vector_service: VectorService) -> None:
        stats = await vector_service.index_chunks([], "doc-empty", "test.pdf")
        assert stats["chunk_count"] == 0
        assert vector_service.count() == 0

    @pytest.mark.asyncio
    async def test_reindex_idempotent(self, vector_service: VectorService, mock_embedding_service) -> None:
        """Re-indexing the same document must not create duplicates."""
        chunks = _make_chunks("doc_re", ["text1", "text2"])
        embeddings = [_fake_embedding(float(i)) for i in range(2)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_re", "test.pdf")
            assert vector_service.count() == 2

            # Index again
            await vector_service.index_chunks(chunks, "doc_re", "test.pdf")
            assert vector_service.count() == 2  # no duplicates


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, vector_service: VectorService, mock_embedding_service) -> None:
        chunks = _make_chunks("doc_search", ["ML is cool", "DL is better"])
        embeddings = [_fake_embedding(float(i)) for i in range(2)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_search", "test.pdf")

        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _fake_embedding(0.0)
            result = await vector_service.search("What is ML?", "doc_search")

        assert "results" in result
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert r["document_id"] == "doc_search"

    @pytest.mark.asyncio
    async def test_search_isolation(self, vector_service: VectorService, mock_embedding_service) -> None:
        """Search in doc A must not return doc B chunks."""
        chunks_a = _make_chunks("docA", ["A chunk 1", "A chunk 2"])
        chunks_b = _make_chunks("docB", ["B chunk 1", "B chunk 2"])

        embeddings_a = [_fake_embedding(float(i)) for i in range(2)]
        embeddings_b = [_fake_embedding(10.0 + i) for i in range(2)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings_a
            await vector_service.index_chunks(chunks_a, "docA", "a.pdf")
            mock_embed.return_value = embeddings_b
            await vector_service.index_chunks(chunks_b, "docB", "b.pdf")

        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _fake_embedding(0.0)
            result = await vector_service.search("query", "docA")

        for r in result["results"]:
            assert r["document_id"] == "docA", f"Cross-contamination: got {r}"


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_document(self, vector_service: VectorService, mock_embedding_service) -> None:
        chunks = _make_chunks("doc_del", ["x", "y", "z"])
        embeddings = [_fake_embedding(float(i)) for i in range(3)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_del", "test.pdf")

        assert vector_service.document_exists("doc_del") is True
        deleted = vector_service.delete_document("doc_del")
        assert deleted == 3
        assert vector_service.document_exists("doc_del") is False


class TestInfo:
    @pytest.mark.asyncio
    async def test_get_document_info(self, vector_service: VectorService, mock_embedding_service) -> None:
        chunks = _make_chunks("doc_info", ["info chunk"])
        embeddings = [_fake_embedding(0.0)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_info", "info.pdf")

        info = vector_service.get_document_info("doc_info")
        assert info["document_id"] == "doc_info"
        assert info["chunk_count"] == 1


class TestIntegrationPipeline:
    """Integration test: Phase 2 chunks → Embedding → ChromaDB → Query → Correct results."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, vector_service: VectorService, mock_embedding_service) -> None:
        # Simulate Phase 2 output
        chunks_a = _make_chunks("integ_A", [
            "Machine learning uses data to find patterns.",
            "Supervised learning requires labeled data.",
        ])
        chunks_b = _make_chunks("integ_B", [
            "Deep learning uses neural networks.",
            "CNNs are used for image recognition.",
        ])

        emb_a = [_normalise([1.0, 0.0, 0.0, 0.0, 0.0]), _normalise([0.9, 0.1, 0.0, 0.0, 0.0])]
        emb_b = [_normalise([0.0, 1.0, 0.0, 0.0, 0.0]), _normalise([0.0, 0.9, 0.1, 0.0, 0.0])]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = emb_a
            await vector_service.index_chunks(chunks_a, "integ_A", "ml.pdf")
            mock_embed.return_value = emb_b
            await vector_service.index_chunks(chunks_b, "integ_B", "dl.pdf")

        # Query document A
        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _normalise([1.0, 0.0, 0.0, 0.0, 0.0])
            result = await vector_service.search("ML patterns", "integ_A", top_k=5)

        results = result["results"]
        assert len(results) > 0
        for r in results:
            assert r["document_id"] == "integ_A"

        # Query document B
        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _normalise([0.0, 1.0, 0.0, 0.0, 0.0])
            result = await vector_service.search("neural networks", "integ_B", top_k=5)

        results = result["results"]
        assert len(results) > 0
        for r in results:
            assert r["document_id"] == "integ_B"

        # Verify total
        assert vector_service.count() == 4
        assert vector_service.count_by_document("integ_A") == 2
        assert vector_service.count_by_document("integ_B") == 2

        # Delete A, B survives
        vector_service.delete_document("integ_A")
        assert vector_service.count() == 2
        assert vector_service.document_exists("integ_A") is False
        assert vector_service.document_exists("integ_B") is True
