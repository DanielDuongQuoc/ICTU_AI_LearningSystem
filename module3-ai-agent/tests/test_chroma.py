"""Phase 3 — ChromaDB repository tests.

Uses a temporary in-memory ChromaDB client so tests are fast
and don't touch the real persistent database.
Each test gets a fresh collection via a unique name.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.repositories.chroma_repository import ChromaRepository


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
        # Phase 1: Create client, add data
        client1 = chromadb.PersistentClient(
            path=temp_chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        repo1 = ChromaRepository(client=client1, collection_name="persist_test")
        _add_sample_chunks(repo1, "persist_doc", ["chunk1", "chunk2", "chunk3"])
        assert repo1.count() == 3

        # Phase 2: Close & recreate client (simulates restart)
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
