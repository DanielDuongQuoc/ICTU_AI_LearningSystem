"""Phase 3 — ChromaDB repository.

Architecture
~~~~~~~~~~~~
API → Service → ChromaRepository → ChromaDB

This module is the ONLY place that communicates directly with ChromaDB.
Services/APIs must go through the repository interface.

Similarity notes
~~~~~~~~~~~~~~~~
The collection is created with ``hnsw:space=cosine``.  ChromaDB returns
*distances* for cosine space: ``distance = 1 - cosine_similarity``.
We convert to similarity via: ``similarity = 1 - distance``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode

logger = logging.getLogger(__name__)


def _chroma_guard(func: Callable[..., Any]) -> Callable[..., Any]:
    """Normalise unexpected ChromaDB exceptions into ``CHROMA_ERROR``.

    The raw exception is logged server-side only; clients never see it.
    ``AppError`` instances pass through unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "ChromaDB operation failed op=%s exception_type=%s",
                func.__name__,
                type(exc).__name__,
                exc_info=True,
            )
            raise AppError(ErrorCode.CHROMA_ERROR, "Vector database error.") from exc

    return wrapper

# ── Module-level client singleton ──────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    if _chroma_client is None:
        raise RuntimeError("ChromaDB client is not initialised. Call init_chroma() first.")
    return _chroma_client


def init_chroma() -> chromadb.PersistentClient:
    """Initialise the persistent ChromaDB client and ensure the collection exists."""
    global _chroma_client

    settings = get_settings()

    logger.info(
        "Initialising ChromaDB at %r with collection %r",
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
    )

    _chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    _chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("ChromaDB initialised successfully.")
    return _chroma_client


def check_chroma_health() -> bool:
    try:
        client = get_chroma_client()
        client.list_collections()
        return True
    except RuntimeError:
        logger.warning("ChromaDB health check: client not initialised.")
        return False
    except Exception as exc:
        logger.error("ChromaDB health check failed: %s", exc)
        return False


def check_embedding_compat() -> bool:
    """Phase 21: stored vectors were produced by the *configured* embedding model.

    True when the store is empty, uninitialised (that is check_chroma_health's job) or holds only legacy
    chunks without an ``embedding_model`` tag. False only on a proven mismatch.
    """
    try:
        client = get_chroma_client()
    except RuntimeError:
        return True
    try:
        collection = client.get_or_create_collection(
            name=get_settings().chroma_collection_name, metadata={"hnsw:space": "cosine"})
        got = collection.get(limit=50, include=["metadatas"])
    except Exception as exc:
        logger.error("Embedding compatibility check failed: %s", exc)
        return False
    models = {str(m["embedding_model"]) for m in (got.get("metadatas") or []) if m and m.get("embedding_model")}
    from app.core import runtime_config

    s = get_settings()
    wanted = runtime_config.current().embedding_model
    allowed = {wanted}
    if s.gemini_api_key.strip() and wanted == s.ollama_embedding_model:
        allowed.add(s.gemini_embedding_model)
    if models and not (models <= allowed):
        logger.error("Embedding model mismatch: stored=%s configured=%s (re-index required)", sorted(models), wanted)
        return False
    return True


# ── Repository class ───────────────────────────────────────────────


class ChromaRepository:
    """Abstraction over a single ChromaDB collection.

    Handles adding, querying, deleting, and counting document chunks.
    All methods operate on the collection specified in ``Settings``.
    """

    def __init__(
        self,
        *,
        client: Optional[chromadb.ClientAPI] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._client = client or get_chroma_client()
        self._collection_name = collection_name or settings.chroma_collection_name
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Collection accessor ─────────────────────────────────────────

    def get_collection(self) -> Any:
        """Return the underlying chromadb Collection object."""
        return self._collection

    # ── Add / Upsert ────────────────────────────────────────────────

    @_chroma_guard
    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert document chunks into the collection.

        Uses ``upsert`` so that re-indexing the same document does not
        create duplicate records (deterministic IDs guarantee idempotency).
        """
        if not ids:
            logger.warning("add_documents called with empty list — skipping.")
            return

        # ChromaDB batch limit is ~41 666 items; chunk into smaller batches
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            "Upserted %d chunks into collection %r",
            len(ids),
            self._collection_name,
        )

    # ── Embedding-model compatibility (Phase 21) ────────────────────

    def stored_embedding_models(self, limit: int = 50) -> set[str]:
        """Embedding models recorded on (a sample of) the stored chunks.

        Chunks indexed before Phase 21 carry no ``embedding_model`` and contribute nothing: compatibility
        of such legacy data cannot be verified (documented in docs/RELEASE_ENGINEERING.md).
        """
        try:
            got = self._collection.get(limit=limit, include=["metadatas"])
        except Exception as exc:
            logger.error("Cannot read stored embedding models: %s", exc)
            raise AppError(ErrorCode.CHROMA_ERROR, "Vector database error.") from exc
        return {str(m["embedding_model"]) for m in (got.get("metadatas") or []) if m and m.get("embedding_model")}

    # ── Query (semantic search) ─────────────────────────────────────

    @_chroma_guard
    def query(
        self,
        query_embedding: list[float],
        document_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search filtered by ``document_id``.

        ChromaDB returns cosine *distances*.  We convert:
            ``similarity = 1.0 - distance``

        Results below ``similarity_threshold`` are excluded.
        """
        settings = get_settings()
        k = top_k if top_k is not None else settings.top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

        # Clamp top_k to a safe range
        k = max(1, min(k, 100))

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict[str, Any]] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        docs = raw["documents"][0] if raw["documents"] else [""] * len(ids)
        metas = raw["metadatas"][0] if raw["metadatas"] else [{}] * len(ids)
        distances = raw["distances"][0] if raw["distances"] else [0.0] * len(ids)

        for i, record_id in enumerate(ids):
            # cosine distance → similarity conversion
            similarity = 1.0 - distances[i]
            if similarity < threshold:
                continue

            meta = metas[i] if metas[i] else {}
            # Resolve page: prefer single 'page', fall back to first of 'pages'
            page = meta.get("page")
            if page is None and meta.get("pages"):
                page = meta["pages"][0] if isinstance(meta["pages"], list) else meta["pages"]

            results.append({
                "chunk_id": meta.get("chunk_id", record_id),
                "document_id": meta.get("document_id", document_id),
                "text": docs[i] if docs[i] else "",
                "page": page,
                "source": meta.get("source", ""),
                "score": round(similarity, 4),
            })

        logger.debug(
            "Query returned %d results (top_k=%d, threshold=%.2f) for document_id=%s",
            len(results),
            k,
            threshold,
            document_id,
        )
        return results

    # ── Get document info ───────────────────────────────────────────

    @_chroma_guard
    def get_document(self, document_id: str) -> dict[str, Any]:
        """Return metadata about a stored document.

        Returns dict with ``document_id``, ``chunk_count``, and ``source``.
        """
        result = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        chunk_count = len(result["ids"]) if result["ids"] else 0
        source = ""
        if chunk_count > 0 and result["metadatas"]:
            source = result["metadatas"][0].get("source", "")

        return {
            "document_id": document_id,
            "chunk_count": chunk_count,
            "source": source,
        }

    # ── Delete ──────────────────────────────────────────────────────

    @_chroma_guard
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to ``document_id``.

        Returns the number of deleted chunks.
        """
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids_to_delete = existing["ids"] if existing["ids"] else []

        if not ids_to_delete:
            logger.info("No chunks found for document_id=%s — nothing to delete.", document_id)
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks for document_id=%s from collection %r",
            len(ids_to_delete),
            document_id,
            self._collection_name,
        )
        return len(ids_to_delete)

    # ── Count ───────────────────────────────────────────────────────

    @_chroma_guard
    def count(self) -> int:
        """Total number of chunks across all documents."""
        return self._collection.count()

    @_chroma_guard
    def count_by_document(self, document_id: str) -> int:
        """Number of chunks belonging to a specific document."""
        result = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        return len(result["ids"]) if result["ids"] else 0

    # ── Existence check ─────────────────────────────────────────────

    def document_exists(self, document_id: str) -> bool:
        """Return True if at least one chunk for ``document_id`` exists."""
        return self.count_by_document(document_id) > 0
