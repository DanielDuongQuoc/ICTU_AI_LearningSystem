"""Documents: PDF/DOCX extraction, chunking, ChromaDB repository, vector indexing, upload limits,
get/delete/search, embedding-model compatibility, persistence and backup/restore."""

from __future__ import annotations

import chromadb
import io
import json
import pytest
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from app.core.exceptions import AppError, ErrorCode
from app.services.document_service import process_pdf
from app.services.pdf_service import extract_pages, validate_pdf_path
from app.utils.text_utils import clean_text
from tests.pdf_helpers import write_pdf, long_text
from app.core.config import get_settings
from app.schemas.document import ExtractedPage, Chunk, ChunkMetadata
from app.services.chunk_service import create_chunks
from typing import Any, Optional
from chromadb.config import Settings as ChromaSettings
from app.repositories.chroma_repository import ChromaRepository, check_embedding_compat
from unittest.mock import AsyncMock, patch
from app.services.embedding_service import GeminiEmbeddingService
from app.services.vector_service import VectorService
from app.schemas.evaluation import EvaluationResponse
from app.services import docx_service
from tests.fakes import RecordingLLM, Stack, LexicalEmbedding, sample_pdf_bytes
from app.repositories import chroma_repository
from scripts.backup import BackupError, create_backup
from scripts.restore import restore_backup
from datetime import datetime, timedelta
from scripts import backup, restore


# ====================================================================================================
# PDF extraction
# ====================================================================================================


def test_valid_pdf(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "valid.pdf", ["Machine learning is a field of AI."])
    pages = extract_pages(pdf_path)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "Machine learning" in pages[0].text


def test_file_is_not_pdf(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("this is not a pdf", encoding="utf-8")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(txt_path)
    assert exc.value.code == ErrorCode.INVALID_FILE


def test_file_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(AppError) as exc:
        validate_pdf_path(missing)
    assert exc.value.code == ErrorCode.FILE_NOT_FOUND


def test_empty_pdf_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(empty)
    assert exc.value.code == ErrorCode.EMPTY_FILE


def test_pdf_without_extractable_text(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "blank.pdf", [""])
    with pytest.raises(AppError) as exc:
        process_pdf(pdf_path, document_id="doc-blank")
    assert exc.value.code == ErrorCode.NO_EXTRACTABLE_TEXT


def test_extract_keeps_page_numbers(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "paged.pdf",
        ["Page one content.", "Page two content.", "Page three content."],
    )
    pages = extract_pages(pdf_path)
    assert [page.page for page in pages] == [1, 2, 3]
    assert "one" in pages[0].text
    assert "two" in pages[1].text
    assert "three" in pages[2].text


def test_text_cleaning() -> None:
    raw = "Machine   Learning\n\n\nis a branch\nof AI."
    assert clean_text(raw) == "Machine Learning\n\nis a branch of AI."


def test_invalid_pdf_bytes(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"%PDF-not-really")
    with pytest.raises(AppError) as exc:
        extract_pages(fake)
    assert exc.value.code in {ErrorCode.INVALID_PDF, ErrorCode.PROCESSING_ERROR}


def test_file_too_large(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "big.pdf", ["tiny"])
    with pytest.raises(AppError) as exc:
        validate_pdf_path(pdf_path, max_bytes=10)
    assert exc.value.code == ErrorCode.FILE_TOO_LARGE


def test_spoofed_pdf_extension(tmp_path: Path) -> None:
    spoofed = tmp_path / "malware.pdf"
    spoofed.write_bytes(b"MZ this is not a pdf")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(spoofed, declared_filename="malware.pdf")
    assert exc.value.code == ErrorCode.INVALID_FILE


def test_multi_page_pdf_processing(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "multi.pdf",
        [
            "Introduction to neural networks and supervised learning.",
            "Backpropagation updates weights using gradient descent.",
            "Regularisation reduces overfitting in deep models.",
        ],
    )
    result = process_pdf(pdf_path, document_id="doc-multi", source_filename="multi.pdf")
    assert result.pages == 3
    assert result.status == "processed"
    assert result.document_id == "doc-multi"
    assert len(result.chunks) >= 1


@pytest.mark.asyncio
async def test_upload_valid_pdf(client, tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    pdf_path = write_pdf(tmp_path / "upload.pdf", ["Supervised learning maps inputs to labels."])
    files = {"file": ("lecture.pdf", pdf_path.read_bytes(), "application/pdf")}

    fake_svc = MagicMock()
    fake_svc.index_chunks = AsyncMock(return_value={"chunk_count": 1})
    with patch("app.api.documents._get_vector_service", return_value=fake_svc):
        response = await client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "lecture.pdf"
    assert data["pages"] == 1
    assert data["chunks"] >= 1
    assert data["status"] == "indexed"
    assert data["document_id"]


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(client) -> None:
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    response = await client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == ErrorCode.INVALID_FILE
    assert "traceback" not in body["error"]["message"].lower()
    assert "Traceback" not in str(body)


# ====================================================================================================
# chunking
# ====================================================================================================


def test_chunking_splits_long_text() -> None:
    pages = [ExtractedPage(page=1, text=long_text(400))]
    chunks = create_chunks(pages, document_id="doc-1", source="sample.pdf", chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_size_respected() -> None:
    settings = get_settings()
    pages = [ExtractedPage(page=1, text=long_text(500))]
    chunks = create_chunks(
        pages,
        document_id="doc-size",
        source="sample.pdf",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    oversized = [chunk for chunk in chunks if len(chunk.text) > settings.chunk_size]
    assert oversized == []


def test_chunk_overlap() -> None:
    pages = [ExtractedPage(page=1, text=long_text(200, prefix="token"))]
    chunks = create_chunks(
        pages,
        document_id="doc-overlap",
        source="sample.pdf",
        chunk_size=80,
        chunk_overlap=25,
    )
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        prev_words = previous.text.split()
        next_words = current.text.split()
        overlap_words = set(prev_words[-8:]) & set(next_words[:12])
        assert overlap_words, "Expected shared tokens between consecutive chunks"


def test_chunk_metadata_single_page() -> None:
    pages = [ExtractedPage(page=5, text=long_text(40))]
    chunks = create_chunks(pages, document_id="doc-meta", source="example.pdf", chunk_size=120, chunk_overlap=20)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata.source == "example.pdf"
        assert chunk.metadata.page == 5
        assert chunk.metadata.pages is None


def test_chunk_metadata_multi_page() -> None:
    pages = [
        ExtractedPage(page=5, text=long_text(60, prefix="alpha")),
        ExtractedPage(page=6, text=long_text(60, prefix="beta")),
    ]
    chunks = create_chunks(pages, document_id="doc-pages", source="example.pdf", chunk_size=200, chunk_overlap=40)
    used_pages = set()
    for chunk in chunks:
        if chunk.metadata.page is not None:
            used_pages.add(chunk.metadata.page)
        if chunk.metadata.pages:
            used_pages.update(chunk.metadata.pages)
    assert used_pages == {5, 6}


def test_unique_chunk_ids() -> None:
    pages = [ExtractedPage(page=1, text=long_text(300))]
    chunks = create_chunks(pages, document_id="doc-ids", source="sample.pdf", chunk_size=90, chunk_overlap=20)
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids
    assert len(ids) == len(set(ids))


def test_document_id_is_preserved() -> None:
    document_id = "11111111-2222-3333-4444-555555555555"
    pages = [ExtractedPage(page=1, text="Gradient descent minimises a loss function.")]
    chunks = create_chunks(pages, document_id=document_id, source="lecture.pdf")
    assert chunks
    assert all(chunk.document_id == document_id for chunk in chunks)


def test_filename_is_not_used_as_document_id(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "example.pdf", ["Decision trees split features to reduce impurity."])
    result = process_pdf(
        pdf_path,
        document_id="kept-document-id",
        source_filename="example.pdf",
    )
    assert result.document_id == "kept-document-id"
    assert result.filename == "example.pdf"
    assert all(chunk.document_id == "kept-document-id" for chunk in result.chunks)
    assert all(chunk.document_id != "example.pdf" for chunk in result.chunks)


# ====================================================================================================
# ChromaDB repository
# ====================================================================================================


# ── Fixtures ────────────────────────────────────────────────────────
@pytest.fixture()
def temp_chroma_dir(tmp_path):
    """Create a temporary directory for persistent ChromaDB tests."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    return str(d)


@pytest.fixture()
def chroma_client():
    """Ephemeral in-memory ChromaDB client for unit tests."""
    return chromadb.Client(ChromaSettings(anonymized_telemetry=False))


@pytest.fixture()
def repo(chroma_client) -> ChromaRepository:
    """Repository backed by the ephemeral client with a unique collection."""
    coll_name = f"test_{uuid.uuid4().hex[:12]}"
    return ChromaRepository(client=chroma_client, collection_name=coll_name)


# ── Helpers ─────────────────────────────────────────────────────────
DIM = 5  # small embedding dimension for tests


def _fake_embedding(seed: float = 0.0) -> list[float]:
    return [seed + i * 0.1 for i in range(DIM)]


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise for cosine distance stability."""
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


def _add_sample_chunks(
    repo: ChromaRepository,
    document_id: str,
    texts: list[str],
    source: str = "test.pdf",
    base_seed: float = 0.0,
) -> None:
    """Helper to add multiple chunks quickly."""
    ids = []
    documents = []
    embeddings = []
    metadatas: list[dict[str, Any]] = []

    for i, text in enumerate(texts):
        chunk_id = f"chunk_{i}"
        record_id = f"{document_id}_{chunk_id}"
        ids.append(record_id)
        documents.append(text)
        embeddings.append(_normalise(_fake_embedding(base_seed + i)))
        metadatas.append({
            "document_id": document_id,
            "chunk_id": chunk_id,
            "page": i + 1,
            "source": source,
        })

    repo.add_documents(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


# ── Tests ───────────────────────────────────────────────────────────
class TestInitialise:
    def test_collection_created(self, repo: ChromaRepository) -> None:
        coll = repo.get_collection()
        assert coll is not None

    def test_initial_count_zero(self, repo: ChromaRepository) -> None:
        assert repo.count() == 0


class TestAddDocuments:
    def test_add_one_chunk(self, repo: ChromaRepository) -> None:
        repo.add_documents(
            ids=["doc1_c1"],
            documents=["Neural networks learn patterns."],
            embeddings=[_normalise(_fake_embedding(1.0))],
            metadatas=[{"document_id": "doc1", "chunk_id": "c1", "page": 1, "source": "a.pdf"}],
        )
        assert repo.count() == 1

    def test_add_multiple_chunks(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "docA", ["text1", "text2", "text3"])
        assert repo.count() == 3

    def test_empty_add_is_noop(self, repo: ChromaRepository) -> None:
        repo.add_documents(ids=[], documents=[], embeddings=[], metadatas=[])
        assert repo.count() == 0


class TestCount:
    def test_total_count(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "doc1", ["a", "b", "c"])
        _add_sample_chunks(repo, "doc2", ["x", "y"], base_seed=10.0)
        assert repo.count() == 5

    def test_count_by_document(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "doc1", ["a", "b", "c"])
        _add_sample_chunks(repo, "doc2", ["x", "y"], base_seed=10.0)
        assert repo.count_by_document("doc1") == 3
        assert repo.count_by_document("doc2") == 2
        assert repo.count_by_document("doc_missing") == 0


class TestDocumentExists:
    def test_exists_true(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "docX", ["data"])
        assert repo.document_exists("docX") is True

    def test_exists_false(self, repo: ChromaRepository) -> None:
        assert repo.document_exists("no_such_doc") is False


class TestQuery:
    def test_basic_query(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "doc1", ["ML is great", "Deep learning rocks"])
        results = repo.query(
            query_embedding=_normalise(_fake_embedding(0.0)),
            document_id="doc1",
            top_k=5,
            similarity_threshold=0.0,
        )
        assert len(results) > 0
        for r in results:
            assert r["document_id"] == "doc1"
            assert "text" in r
            assert "score" in r

    def test_metadata_preserved(self, repo: ChromaRepository) -> None:
        repo.add_documents(
            ids=["doc1_c1"],
            documents=["Backpropagation is used to train neural networks."],
            embeddings=[_normalise(_fake_embedding(1.0))],
            metadatas=[{
                "document_id": "doc1",
                "chunk_id": "c1",
                "page": 5,
                "source": "lecture.pdf",
            }],
        )
        results = repo.query(
            query_embedding=_normalise(_fake_embedding(1.0)),
            document_id="doc1",
            top_k=1,
            similarity_threshold=0.0,
        )
        assert len(results) == 1
        assert results[0]["page"] == 5
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["document_id"] == "doc1"

    def test_page_preservation(self, repo: ChromaRepository) -> None:
        pages = [3, 7, 12]
        ids = []
        docs = []
        embs = []
        metas: list[dict[str, Any]] = []

        for i, page in enumerate(pages):
            ids.append(f"doc1_chunk{i}")
            docs.append(f"Content on page {page}")
            embs.append(_normalise(_fake_embedding(float(i))))
            metas.append({
                "document_id": "doc1",
                "chunk_id": f"chunk{i}",
                "page": page,
                "source": "book.pdf",
            })

        repo.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

        results = repo.query(
            query_embedding=_normalise(_fake_embedding(0.0)),
            document_id="doc1",
            top_k=10,
            similarity_threshold=0.0,
        )
        returned_pages = {r["page"] for r in results}
        assert returned_pages.issubset(set(pages))

    def test_document_id_filtering(self, repo: ChromaRepository) -> None:
        """CRITICAL: Query for doc A must NOT return chunks from doc B."""
        _add_sample_chunks(repo, "docA", ["chunk A1", "chunk A2"], base_seed=0.0)
        _add_sample_chunks(repo, "docB", ["chunk B1", "chunk B2"], base_seed=5.0)

        results = repo.query(
            query_embedding=_normalise(_fake_embedding(0.0)),
            document_id="docA",
            top_k=10,
            similarity_threshold=0.0,
        )

        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {"docA"}, f"Expected only docA, got {doc_ids}"
        texts = {r["text"] for r in results}
        assert "chunk B1" not in texts
        assert "chunk B2" not in texts

    def test_similarity_threshold_filters(self, repo: ChromaRepository) -> None:
        repo.add_documents(
            ids=["d1_c1"],
            documents=["Gradient descent"],
            embeddings=[_normalise([1.0, 0.0, 0.0, 0.0, 0.0])],
            metadatas=[{"document_id": "d1", "chunk_id": "c1", "page": 1, "source": "a.pdf"}],
        )
        # Query with an orthogonal vector → low similarity
        results = repo.query(
            query_embedding=_normalise([0.0, 0.0, 0.0, 0.0, 1.0]),
            document_id="d1",
            top_k=10,
            similarity_threshold=0.99,  # very high threshold
        )
        assert len(results) == 0  # nothing should pass

    def test_top_k_limits(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "doc1", [f"chunk {i}" for i in range(10)])
        results = repo.query(
            query_embedding=_normalise(_fake_embedding(0.0)),
            document_id="doc1",
            top_k=3,
            similarity_threshold=0.0,
        )
        assert len(results) <= 3


class TestDocumentIsolation:
    """Dedicated test: Document A ↔ Document B isolation."""

    def test_query_a_returns_only_a(self, repo: ChromaRepository) -> None:
        # Document A with specific embeddings
        repo.add_documents(
            ids=["A_A1", "A_A2"],
            documents=["Chunk A1 about supervised learning", "Chunk A2 about regression"],
            embeddings=[
                _normalise([1.0, 0.2, 0.0, 0.0, 0.0]),
                _normalise([0.9, 0.3, 0.1, 0.0, 0.0]),
            ],
            metadatas=[
                {"document_id": "A", "chunk_id": "A1", "page": 1, "source": "docA.pdf"},
                {"document_id": "A", "chunk_id": "A2", "page": 2, "source": "docA.pdf"},
            ],
        )
        # Document B with VERY similar embeddings
        repo.add_documents(
            ids=["B_B1", "B_B2"],
            documents=["Chunk B1 about classification", "Chunk B2 about clustering"],
            embeddings=[
                _normalise([1.0, 0.2, 0.01, 0.0, 0.0]),  # almost identical to A1
                _normalise([0.95, 0.25, 0.05, 0.0, 0.0]),
            ],
            metadatas=[
                {"document_id": "B", "chunk_id": "B1", "page": 1, "source": "docB.pdf"},
                {"document_id": "B", "chunk_id": "B2", "page": 2, "source": "docB.pdf"},
            ],
        )

        results_a = repo.query(
            query_embedding=_normalise([1.0, 0.2, 0.0, 0.0, 0.0]),
            document_id="A",
            top_k=10,
            similarity_threshold=0.0,
        )
        for r in results_a:
            assert r["document_id"] == "A", f"Cross-contamination: got {r}"

        results_b = repo.query(
            query_embedding=_normalise([1.0, 0.2, 0.0, 0.0, 0.0]),
            document_id="B",
            top_k=10,
            similarity_threshold=0.0,
        )
        for r in results_b:
            assert r["document_id"] == "B", f"Cross-contamination: got {r}"


class TestDeleteDocument:
    def test_delete_removes_all_chunks(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "docDel", ["d1", "d2", "d3"])
        assert repo.count_by_document("docDel") == 3

        deleted = repo.delete_document("docDel")
        assert deleted == 3
        assert repo.count_by_document("docDel") == 0
        assert repo.document_exists("docDel") is False

    def test_delete_preserves_other_documents(self, repo: ChromaRepository) -> None:
        """Delete A must NOT touch B."""
        _add_sample_chunks(repo, "docA", ["A1", "A2", "A3"])
        _add_sample_chunks(repo, "docB", ["B1", "B2"], base_seed=10.0)
        assert repo.count() == 5

        repo.delete_document("docA")
        assert repo.count() == 2
        assert repo.count_by_document("docA") == 0
        assert repo.count_by_document("docB") == 2
        assert repo.document_exists("docB") is True

    def test_delete_nonexistent(self, repo: ChromaRepository) -> None:
        deleted = repo.delete_document("no_such_doc")
        assert deleted == 0


class TestDuplicateHandling:
    def test_upsert_does_not_create_duplicates(self, repo: ChromaRepository) -> None:
        """Re-indexing the same document must NOT increase count."""
        _add_sample_chunks(repo, "docU", ["text1", "text2"])
        assert repo.count() == 2

        # Index again with same IDs
        _add_sample_chunks(repo, "docU", ["text1 updated", "text2 updated"])
        assert repo.count() == 2  # must still be 2


class TestGetDocument:
    def test_get_existing_document(self, repo: ChromaRepository) -> None:
        _add_sample_chunks(repo, "docInfo", ["a", "b", "c"], source="info.pdf")
        info = repo.get_document("docInfo")
        assert info["document_id"] == "docInfo"
        assert info["chunk_count"] == 3
        assert info["source"] == "info.pdf"

    def test_get_nonexistent_document(self, repo: ChromaRepository) -> None:
        info = repo.get_document("no_doc")
        assert info["chunk_count"] == 0


class TestPersistence:
    """ChromaDB persistence test — data survives client reinitialisation."""

    def test_data_survives_reopen(self, temp_chroma_dir: str) -> None:
        # Create client, add data
        client1 = chromadb.PersistentClient(
            path=temp_chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        repo1 = ChromaRepository(client=client1, collection_name="persist_test")
        _add_sample_chunks(repo1, "persist_doc", ["chunk1", "chunk2", "chunk3"])
        assert repo1.count() == 3

        # Close & recreate client (simulates restart)
        del repo1
        del client1

        client2 = chromadb.PersistentClient(
            path=temp_chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        repo2 = ChromaRepository(client=client2, collection_name="persist_test")
        assert repo2.count() == 3
        assert repo2.document_exists("persist_doc") is True
        assert repo2.count_by_document("persist_doc") == 3

        # Verify query still works
        results = repo2.query(
            query_embedding=_normalise(_fake_embedding(0.0)),
            document_id="persist_doc",
            top_k=5,
            similarity_threshold=0.0,
        )
        assert len(results) > 0
        assert all(r["document_id"] == "persist_doc" for r in results)


# ====================================================================================================
# vector indexing
# ====================================================================================================


# ── Fixtures ────────────────────────────────────────────────────────
DIM_vector_service = 5


def _normalise_vector_service(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


def _fake_embedding_vector_service(seed: float = 0.0) -> list[float]:
    return _normalise_vector_service([seed + i * 0.1 for i in range(DIM_vector_service)])


@pytest.fixture()
def chroma_client_vector_service():
    return chromadb.Client(ChromaSettings(anonymized_telemetry=False))


@pytest.fixture()
def repo_vector_service(chroma_client_vector_service) -> ChromaRepository:
    return ChromaRepository(
        client=chroma_client_vector_service, collection_name=f"vs_test_{uuid.uuid4().hex[:12]}"
    )


@pytest.fixture()
def mock_embedding_service():
    """Return an GeminiEmbeddingService with mocked embed methods."""
    service = GeminiEmbeddingService(
        base_url="http://gemini.test/v1beta",
        model="test-embed",
        timeout=10,
    )
    return service


@pytest.fixture()
def vector_service(mock_embedding_service, repo_vector_service) -> VectorService:
    return VectorService(embedding_service=mock_embedding_service, chroma_repo=repo_vector_service)


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
        embeddings = [_fake_embedding_vector_service(float(i)) for i in range(3)]

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
        embeddings = [_fake_embedding_vector_service(float(i)) for i in range(2)]

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
        embeddings = [_fake_embedding_vector_service(float(i)) for i in range(2)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_search", "test.pdf")

        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _fake_embedding_vector_service(0.0)
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

        embeddings_a = [_fake_embedding_vector_service(float(i)) for i in range(2)]
        embeddings_b = [_fake_embedding_vector_service(10.0 + i) for i in range(2)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings_a
            await vector_service.index_chunks(chunks_a, "docA", "a.pdf")
            mock_embed.return_value = embeddings_b
            await vector_service.index_chunks(chunks_b, "docB", "b.pdf")

        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _fake_embedding_vector_service(0.0)
            result = await vector_service.search("query", "docA")

        for r in result["results"]:
            assert r["document_id"] == "docA", f"Cross-contamination: got {r}"


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_document(self, vector_service: VectorService, mock_embedding_service) -> None:
        chunks = _make_chunks("doc_del", ["x", "y", "z"])
        embeddings = [_fake_embedding_vector_service(float(i)) for i in range(3)]

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
        embeddings = [_fake_embedding_vector_service(0.0)]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = embeddings
            await vector_service.index_chunks(chunks, "doc_info", "info.pdf")

        info = vector_service.get_document_info("doc_info")
        assert info["document_id"] == "doc_info"
        assert info["chunk_count"] == 1


class TestIntegrationPipeline:
    """Integration test: chunks → Embedding → ChromaDB → Query → Correct results."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, vector_service: VectorService, mock_embedding_service) -> None:
        # Simulate chunking output
        chunks_a = _make_chunks("integ_A", [
            "Machine learning uses data to find patterns.",
            "Supervised learning requires labeled data.",
        ])
        chunks_b = _make_chunks("integ_B", [
            "Deep learning uses neural networks.",
            "CNNs are used for image recognition.",
        ])

        emb_a = [_normalise_vector_service([1.0, 0.0, 0.0, 0.0, 0.0]), _normalise_vector_service([0.9, 0.1, 0.0, 0.0, 0.0])]
        emb_b = [_normalise_vector_service([0.0, 1.0, 0.0, 0.0, 0.0]), _normalise_vector_service([0.0, 0.9, 0.1, 0.0, 0.0])]

        with patch.object(mock_embedding_service, "embed_documents", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = emb_a
            await vector_service.index_chunks(chunks_a, "integ_A", "ml.pdf")
            mock_embed.return_value = emb_b
            await vector_service.index_chunks(chunks_b, "integ_B", "dl.pdf")

        # Query document A
        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _normalise_vector_service([1.0, 0.0, 0.0, 0.0, 0.0])
            result = await vector_service.search("ML patterns", "integ_A", top_k=5)

        results = result["results"]
        assert len(results) > 0
        for r in results:
            assert r["document_id"] == "integ_A"

        # Query document B
        with patch.object(mock_embedding_service, "embed_text", new_callable=AsyncMock) as mock_et:
            mock_et.return_value = _normalise_vector_service([0.0, 1.0, 0.0, 0.0, 0.0])
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


# ====================================================================================================
# DOCX ingestion, exam matrix, no fraud API
# ====================================================================================================


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture()
def _settings_module3_requirements(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_docx(blocks: list[Any]) -> bytes:
    """blocks: str paragraph | ("table", [[cells]]) | "<PAGEBREAK>" | ("heading", text)."""
    import docx

    d = docx.Document()
    for b in blocks:
        if b == "<PAGEBREAK>":
            d.add_page_break()
        elif isinstance(b, tuple) and b[0] == "table":
            t = d.add_table(rows=len(b[1]), cols=len(b[1][0]))
            for r, row in enumerate(b[1]):
                for c, cell in enumerate(row):
                    t.cell(r, c).text = cell
        elif isinstance(b, tuple) and b[0] == "heading":
            d.add_heading(b[1], level=1)
        else:
            d.add_paragraph(b)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


async def upload(client, data: bytes, name: str = "lecture.docx", mime: str = DOCX_MIME):
    return await client.post("/api/documents/upload", files={"file": (name, data, mime)})


# ===================================================================== DOCX
@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_docx_upload_is_indexed_and_retrievable_with_logical_pages(client, monkeypatch):
    Stack().install(monkeypatch)
    data = make_docx([
        ("heading", "Chapter one"),
        " ".join(f"Zebracorn systems explain machine learning topic {i}. Zebracorn models learn patterns from data." for i in range(14)),
        "<PAGEBREAK>",
        "Quokkaplex indexes speed database queries. Quokkaplex structures reduce lookup time.",
        ("table", [["Narwhalon", "cache"], ["Hedgeflux", "queue"]]),
    ])
    r = await upload(client, data)
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["filename"] == "lecture.docx" and body["status"] == "indexed"
    assert body["pages"] == 2 and body["chunks"] >= 2
    doc = body["document_id"]

    hit = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "quokkaplex indexes database queries"})
    assert hit.status_code == 200, hit.text
    chunks = hit.json()["chunks"]
    assert chunks and all(c["document_id"] == doc for c in chunks)
    # A chunk that straddles a page break reports its first page (same rule as PDF), so check the exact page at extraction level.
    assert any("Quokkaplex" in c["text"] and c["page"] in (1, 2) for c in chunks), chunks
    assert all("filename" not in c or c["filename"] for c in chunks)
    tab = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "narwhalon cache hedgeflux queue"})
    assert any("Narwhalon | cache" in c["text"] for c in tab.json()["chunks"]), "table rows must be extracted"

    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    assert (await client.get(f"/api/documents/{doc}")).status_code == 404


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_docx_and_pdf_documents_stay_isolated(client, monkeypatch):
    from tests.fakes import sample_pdf_bytes

    Stack().install(monkeypatch)
    d = await upload(client, make_docx(["Zebracorn systems explain machine learning from data."]))
    p = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes(["Quokkaplex indexes speed database queries."]), "application/pdf")})
    assert d.status_code in (200, 201) and p.status_code in (200, 201)
    hit = await client.post("/api/rag/retrieve", json={"document_id": d.json()["document_id"], "question": "quokkaplex indexes"})
    assert all("Quokkaplex" not in c["text"] for c in hit.json()["chunks"])


@pytest.mark.usefixtures("_settings_module3_requirements")
def test_page_breaks_define_logical_pages(tmp_path):
    path = tmp_path / "pb.docx"
    path.write_bytes(make_docx(["First page text.", "<PAGEBREAK>", "Second page text.", ("table", [["a", "b"]]), "<PAGEBREAK>", "Third."]))
    pages = docx_service.extract_pages(path)
    assert [(p.page, p.text) for p in pages] == [(1, "First page text."), (2, "Second page text.\n\na | b"), (3, "Third.")]


@pytest.mark.usefixtures("_settings_module3_requirements")
def test_long_page_is_split_into_logical_pages(tmp_path):
    path = tmp_path / "long.docx"
    path.write_bytes(make_docx([f"Paragraph {i}: " + "word " * 40 for i in range(60)]))
    pages = docx_service.extract_pages(path)
    assert len(pages) > 1
    assert [p.page for p in pages] == list(range(1, len(pages) + 1))
    assert all(len(p.text) <= docx_service.LOGICAL_PAGE_CHARS + 400 for p in pages)


@pytest.mark.usefixtures("_settings_module3_requirements")
def test_blank_docx_has_no_extractable_text(tmp_path):
    path = tmp_path / "blank.docx"
    path.write_bytes(make_docx([]))
    with pytest.raises(AppError) as e:
        docx_service.extract_pages(path)
    assert e.value.code == ErrorCode.NO_EXTRACTABLE_TEXT


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_docx_content_is_checked_not_just_the_name(client, monkeypatch):
    Stack().install(monkeypatch)
    for data in (b"plain text pretending to be a document", b"PK\x03\x04 not really a zip"):
        r = await upload(client, data)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "INVALID_FILE"


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_other_formats_and_wrong_mime_are_rejected(client, monkeypatch):
    Stack().install(monkeypatch)
    ok = make_docx(["Zebracorn systems explain machine learning from data."])
    assert (await upload(client, ok, "x.docx", "image/png")).status_code == 400   # wrong MIME
    assert (await upload(client, ok, "x.doc", "application/msword")).status_code == 400   # legacy Word
    assert (await upload(client, ok, "x.docm", "application/vnd.ms-word.document.macroEnabled.12")).status_code == 400


def _zip(names_to_data: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in names_to_data.items():
            z.writestr(n, d)
    return buf.getvalue()


@pytest.mark.usefixtures("_settings_module3_requirements")
def test_macro_zip_bomb_and_non_word_zip_are_rejected(tmp_path):
    base = {"[Content_Types].xml": b"<x/>", "word/document.xml": b"<x/>"}
    cases = {
        "macro": _zip({**base, "word/vbaProject.bin": b"x"}),
        "notword": _zip({"hello.txt": b"hi"}),
        "bomb": _zip({**base, "word/big.xml": b"0" * 3_000_000}),
    }
    for name, data in cases.items():
        p = tmp_path / f"{name}.docx"
        p.write_bytes(data)
        with pytest.raises(AppError) as e:
            docx_service.validate_docx_path(p, max_bytes=200_000, declared_filename=p.name)
        assert e.value.code in (ErrorCode.INVALID_FILE, ErrorCode.DOCUMENT_LIMIT_EXCEEDED), name


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_oversized_docx_is_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    get_settings.cache_clear()
    Stack().install(monkeypatch)
    big = make_docx(["x"]) + b"0" * (1024 * 1024 + 10)
    assert (await upload(client, big)).status_code == 413


# ===================================================================== exam matrix
DOC_TEXT = " ".join([
    "Quokkaplex indexes speed database queries.", "Narwhalon caches store frequent results.",
    "Hedgeflux queues order background jobs.", "Zebracorn models learn patterns from data.",
    "Pangolith schedulers balance worker load.", "Marmotron logs record every request.",
    "Lemuroid routers forward network packets.", "Okapiton filters remove invalid input.",
])


TOPICS = ["quokkaplex indexes speed database queries", "narwhalon caches store frequent results",
          "hedgeflux queues order background jobs", "zebracorn models learn patterns from data",
          "pangolith schedulers balance worker load", "marmotron logs record every request",
          "lemuroid routers forward network packets", "okapiton filters remove invalid input"]


class MatrixLLM(RecordingLLM):
    """Honours the requested type/difficulty and yields a fresh grounded question each time."""

    def __init__(self, per_call: int = 6, fail_from_call: Optional[int] = None, same_text: bool = False) -> None:
        super().__init__()
        self.per_call, self.fail_from_call, self.same_text = per_call, fail_from_call, same_text
        self.gen_calls = 0
        self.counter = 0

    async def generate_json(self, system_prompt: str, user_prompt: str, *, required_keys=None, **kw: Any) -> Any:
        if "questions" not in (required_keys or []):
            return await super().generate_json(system_prompt, user_prompt, required_keys=required_keys, **kw)
        self.calls.append(("json", system_prompt, user_prompt))
        self.gen_calls += 1
        if self.fail_from_call is not None and self.gen_calls >= self.fail_from_call:
            raise AppError(ErrorCode.LLM_TIMEOUT, "model timed out")
        diff = re.search(r"Difficulty: (\w+)", user_prompt).group(1)
        qtype = re.search(r"Question type: (\w+)", user_prompt).group(1)
        out = []
        for _ in range(self.per_call):
            topic = TOPICS[0 if self.same_text else self.counter % len(TOPICS)]
            self.counter += 1
            q = {"question": f"What does the document explain about {topic}?",
                 "expected_answer": f"It states that {topic}.", "difficulty": diff, "type": qtype}
            if qtype == "multiple_choice":
                opts = [f"{topic}", f"Option B{self.counter}", f"Option C{self.counter}", f"Option D{self.counter}"]
                q.update(options=opts, correct_answer=opts[0])
            elif qtype == "true_false":
                q.update(correct_answer="true", options=["True", "False"])
            out.append(q)
        return {"questions": out}


async def matrix(client, doc, items, subject=None):
    body = {"document_id": doc, "items": items}
    if subject is not None:
        body["subject"] = subject
    return await client.post("/api/questions/generate-matrix", json=body)


async def indexed_doc(client, monkeypatch, llm):
    stack = Stack(llm).install(monkeypatch)
    r = await upload(client, make_docx([DOC_TEXT]))
    assert r.status_code in (200, 201), r.text
    return stack, r.json()["document_id"]


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_generates_exact_counts_types_and_difficulties(client, monkeypatch):
    llm = MatrixLLM()
    _, doc = await indexed_doc(client, monkeypatch, llm)
    items = [{"question_type": "multiple_choice", "difficulty": "medium", "count": 2},
             {"question_type": "true_false", "difficulty": "easy", "count": 1},
             {"question_type": "short_answer", "difficulty": "hard", "count": 2}]
    r = await matrix(client, doc, items, subject="  Databases   101 ")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["subject"] == "Databases 101" and body["requested_total"] == 5 == body["generated_total"]
    assert body["complete"] is True
    assert [i["status"] for i in body["items"]] == ["ok"] * 3
    qs = body["questions"]
    assert len(qs) == 5 and len({q["question_id"] for q in qs}) == 5
    for idx, it in enumerate(items):
        mine = [q for q in qs if q["matrix_item"] == idx]
        assert len(mine) == it["count"]
        assert all(q["type"] == it["question_type"] and q["difficulty"] == it["difficulty"] for q in mine)
    assert all(q["source_chunks"] for q in qs), "every question is traceable to document chunks"
    assert all(len(q["options"]) == 4 and q["correct_answer"] in q["options"] for q in qs if q["type"] == "multiple_choice")
    assert len({q["question"] for q in qs}) == 5, "no duplicate across rows"
    assert "Databases" not in llm.prompts, "subject is not sent to the model"
    assert llm.gen_calls == 3, "one gated LLM call per matrix row"


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_reports_shortfall_instead_of_padding(client, monkeypatch):
    llm = MatrixLLM(per_call=1)
    _, doc = await indexed_doc(client, monkeypatch, llm)
    r = await matrix(client, doc, [{"question_type": "short_answer", "difficulty": "easy", "count": 3},
                                    {"question_type": "short_answer", "difficulty": "hard", "count": 1}])
    body = r.json()
    assert r.status_code == 200
    assert body["complete"] is False and body["requested_total"] == 4 and body["generated_total"] == 2
    assert [(i["requested"], i["generated"], i["status"]) for i in body["items"]] == [(3, 1, "partial"), (1, 1, "ok")]


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_later_row_failure_is_reported_and_earlier_rows_kept(client, monkeypatch):
    llm = MatrixLLM(fail_from_call=2)
    _, doc = await indexed_doc(client, monkeypatch, llm)
    r = await matrix(client, doc, [{"question_type": "short_answer", "difficulty": "easy", "count": 2},
                                    {"question_type": "essay", "difficulty": "hard", "count": 2}])
    body = r.json()
    assert r.status_code == 200 and body["complete"] is False
    assert body["items"][0]["status"] == "ok"
    assert body["items"][1]["status"] == "failed" and body["items"][1]["error_code"] == "OLLAMA_TIMEOUT"
    assert body["generated_total"] == 2 and all(q["matrix_item"] == 0 for q in body["questions"])


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_first_row_failure_fails_the_request(client, monkeypatch):
    llm = MatrixLLM(fail_from_call=1)
    _, doc = await indexed_doc(client, monkeypatch, llm)
    r = await matrix(client, doc, [{"question_type": "short_answer", "difficulty": "easy", "count": 2}])
    assert r.status_code == 504 and r.json()["error"]["code"] == "OLLAMA_TIMEOUT"


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_removes_duplicates_across_rows(client, monkeypatch):
    llm = MatrixLLM(per_call=1, same_text=True)
    _, doc = await indexed_doc(client, monkeypatch, llm)
    r = await matrix(client, doc, [{"question_type": "short_answer", "difficulty": "easy", "count": 1},
                                    {"question_type": "short_answer", "difficulty": "medium", "count": 1}])
    body = r.json()
    assert body["generated_total"] == 1 and body["complete"] is False
    assert body["items"][1]["status"] == "failed"


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_unknown_document_is_404(client, monkeypatch):
    Stack(MatrixLLM()).install(monkeypatch)
    r = await matrix(client, "11111111-1111-1111-1111-111111111111",
                     [{"question_type": "short_answer", "difficulty": "easy", "count": 1}])
    assert r.status_code == 404


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
@pytest.mark.parametrize("items", [
    [],
    [{"question_type": "short_answer", "difficulty": "easy", "count": 0}],
    [{"question_type": "short_answer", "difficulty": "easy", "count": 21}],
    [{"question_type": "poem", "difficulty": "easy", "count": 1}],
    [{"question_type": "short_answer", "difficulty": "impossible", "count": 1}],
    [{"question_type": "short_answer", "difficulty": "easy", "count": 20}] * 3,                    # 60 > 50
    [{"question_type": "short_answer", "difficulty": "easy", "count": 1}] * 13,                    # > 12 rows
])
async def test_matrix_invalid_requests_are_422(client, monkeypatch, items):
    Stack(MatrixLLM()).install(monkeypatch)
    r = await matrix(client, "11111111-1111-1111-1111-111111111111", items)
    assert r.status_code == 422, r.text


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_matrix_total_limit_is_configurable(client, monkeypatch):
    monkeypatch.setenv("MAX_MATRIX_QUESTIONS", "3")
    get_settings.cache_clear()
    Stack(MatrixLLM()).install(monkeypatch)
    r = await matrix(client, "11111111-1111-1111-1111-111111111111",
                     [{"question_type": "short_answer", "difficulty": "easy", "count": 4}])
    assert r.status_code == 422


@pytest.mark.usefixtures("_settings_module3_requirements")
@pytest.mark.asyncio
async def test_existing_single_type_generation_is_unchanged(client, monkeypatch):
    llm = MatrixLLM()
    _, doc = await indexed_doc(client, monkeypatch, llm)
    r = await client.post("/api/questions/generate", json={"document_id": doc, "number_of_questions": 2,
                                                             "difficulty": "easy", "question_type": "short_answer"})
    assert r.status_code == 200 and len(r.json()["questions"]) == 2
    assert "matrix_item" not in r.json()["questions"][0]


# ===================================================================== scoring is not fraud detection
@pytest.mark.usefixtures("_settings_module3_requirements")
def test_evaluation_contract_has_no_fraud_or_cheating_fields():
    assert set(EvaluationResponse.model_fields) == {
        "score", "max_score", "percentage", "correct", "feedback", "strengths", "missing_points", "errors", "sources"}


@pytest.mark.usefixtures("_settings_module3_requirements")
def test_no_fraud_api_exists():
    from app.main import app

    paths = " ".join(r.path for r in app.routes if hasattr(r, "path")).lower()
    assert not re.search(r"fraud|cheat|plagiar|anti", paths)


# ====================================================================================================
# embedding-model compatibility
# ====================================================================================================


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
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "lexical-test-embedding")
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

    monkeypatch.setattr("app.api.health.check_provider_health", up)
    monkeypatch.setattr("app.api.health.check_chroma_health", lambda: True)
    assert (await client.get("/ready")).status_code == 200

    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")      # "new release" with another model
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


# ====================================================================================================
# restart persistence and backup/restore
# ====================================================================================================


ROOT = Path(__file__).resolve().parents[1]


_CHILD = """
import json, sys
import chromadb
from chromadb.config import Settings
mode, path = sys.argv[1], sys.argv[2]
col = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False)).get_or_create_collection(
    "drill", metadata={"hnsw:space": "cosine"})
if mode == "ingest":
    col.upsert(ids=["c0", "c1"], documents=["zebracorn data", "quokka index"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
               metadatas=[{"document_id": "d", "source": "drill.pdf"}] * 2)
res = col.query(query_embeddings=[[1.0, 0.0]], n_results=1, where={"document_id": "d"})
print(json.dumps({"chunks": col.count(), "results": len(res["ids"][0]),
                  "source": res["metadatas"][0][0]["source"] if res["ids"][0] else None}))
"""


def _spawn(mode: str, data: Path) -> dict:
    """Run ingest/query in a brand-new Python process (= an application restart)."""
    proc = subprocess.run([sys.executable, "-c", _CHILD, mode, str(data)], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_data_survives_process_restart(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    before = _spawn("ingest", data)
    assert before["chunks"] > 0
    after = _spawn("query", data)               # a brand-new Python process = a restart
    assert after["chunks"] == before["chunks"] and after["results"] > 0 and after["source"] == "drill.pdf"


def test_corrupted_backup_is_rejected_and_target_untouched(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    _spawn("ingest", data)
    archive = create_backup(data, tmp_path / "backups")
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(bytes(raw))
    before = sorted(p.name for p in data.rglob("*"))
    with pytest.raises(BackupError, match="(?i)checksum|cannot read"):
        restore_backup(archive, data, assume_yes=True)
    assert sorted(p.name for p in data.rglob("*")) == before            # nothing replaced


def test_restore_requires_confirmation_and_keeps_previous_data(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    _spawn("ingest", data)
    archive = create_backup(data, tmp_path / "backups")
    with pytest.raises(BackupError, match="--yes"):
        restore_backup(archive, data, assume_yes=False)
    moved = restore_backup(archive, data, assume_yes=True)               # target is non-empty
    assert moved is not None and moved.is_dir() and any(moved.iterdir()), "previous data must be kept aside"
    assert _spawn("query", data)["results"] > 0


def test_backup_refuses_missing_empty_and_existing(tmp_path):
    with pytest.raises(BackupError):
        create_backup(tmp_path / "missing", tmp_path / "b")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BackupError):
        create_backup(empty, tmp_path / "b")
    data = tmp_path / "chroma"
    data.mkdir()
    (data / "f").write_text("x")
    from datetime import datetime

    stamp = datetime(2026, 1, 1, 1, 1, 1)
    create_backup(data, tmp_path / "b", now=stamp)
    with pytest.raises(BackupError, match="overwrite"):
        create_backup(data, tmp_path / "b", now=stamp)
    shutil.rmtree(tmp_path / "b")


# ====================================================================================================
# backup -> restore drill
# ====================================================================================================


# ------------------------------------------------------------------ helpers
def _make_backups(tmp_path: Path, n: int) -> list[Path]:
    src = tmp_path / "chroma"
    src.mkdir(exist_ok=True)
    (src / "f.bin").write_bytes(b"data")
    base = datetime(2026, 1, 1, 12, 0, 0)
    return [backup.create_backup(src, tmp_path / "b", now=base + timedelta(days=i)) for i in range(n)]


# ------------------------------------------------------------------ backup -> modify -> restore drill
def _repo(path: Path) -> ChromaRepository:
    return ChromaRepository(
        client=chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False)),
        collection_name="drill",
    )


@pytest.mark.asyncio
async def test_backup_restore_drill_retrieval_works_in_isolated_dir(tmp_path):
    """Real ChromaDB persistence under tmp_path only; production data is never touched."""
    emb = LexicalEmbedding()
    live = tmp_path / "live" / "chroma"
    live.mkdir(parents=True)
    repo = _repo(live)
    texts = ["Zebracorn models learn patterns from data.", "Zebracorn systems explain machine learning."]
    repo.add_documents(
        ids=["c0", "c1"], documents=texts, embeddings=await emb.embed_documents(texts),
        metadatas=[{"document_id": "doc-1", "page": 1, "chunk_index": i, "chunk_id": f"c{i}"} for i in range(2)],
    )
    assert repo.count_by_document("doc-1") == 2

    archive = backup.create_backup(live, tmp_path / "backups")
    assert archive.exists()
    assert backup.sha256_of(archive) in archive.with_name(archive.name + ".sha256").read_text()

    repo.delete_document("doc-1")                        # simulate data loss after the backup
    assert repo.count_by_document("doc-1") == 0

    restored_dir = tmp_path / "restored" / "chroma"
    restore.restore_backup(archive, restored_dir, assume_yes=True)
    restored = _repo(restored_dir)
    assert restored.count_by_document("doc-1") == 2      # integrity: chunks are back
    hits = restored.query(query_embedding=await emb.embed_text("zebracorn data"), document_id="doc-1",
                          top_k=2, similarity_threshold=0.0)
    assert hits                                          # retrieval works on restored data


def test_restore_refuses_without_confirmation_and_on_corruption(tmp_path):
    archives = _make_backups(tmp_path, 1)
    with pytest.raises(backup.BackupError):
        restore.restore_backup(archives[0], tmp_path / "t", assume_yes=False)
    archives[0].write_bytes(b"corrupted")
    with pytest.raises(backup.BackupError):
        restore.restore_backup(archives[0], tmp_path / "t", assume_yes=True)
