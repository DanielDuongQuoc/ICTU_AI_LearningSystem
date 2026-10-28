"""Phase 3 — Vector operations service.

Orchestrates the pipeline:
    chunks → EmbeddingService → ChromaRepository

This service sits between the API layer and the lower-level
EmbeddingService / ChromaRepository.  It contains no HTTP or API logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.repositories.chroma_repository import ChromaRepository
from app.schemas.document import Chunk
from app.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class VectorService:
    """Index document chunks and perform semantic search."""

    def __init__(
        self,
        *,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_repo: Optional[ChromaRepository] = None,
    ) -> None:
        self._embedding: EmbeddingService = embedding_service or get_embedding_service()
        self._repo: ChromaRepository = chroma_repo or ChromaRepository()

    # ── Index chunks ────────────────────────────────────────────────

    async def index_chunks(
        self,
        chunks: list[Chunk],
        document_id: str,
        source: str,
    ) -> dict[str, Any]:
        """Embed and store chunks in ChromaDB.

        Returns timing / stats dict for logging/response.
        """
        if not chunks:
            logger.warning("index_chunks called with empty chunk list for document_id=%s", document_id)
            return {
                "document_id": document_id,
                "chunk_count": 0,
                "embedding_time_ms": 0.0,
                "chroma_insert_time_ms": 0.0,
                "total_time_ms": 0.0,
                "model_name": self._embedding.model_name,
            }

        total_start = time.perf_counter()

        # ── Build texts for batch embedding ─────────────────────────
        texts = [chunk.text for chunk in chunks]

        # ── Never mix vectors from different embedding models (Phase 21) ──
        try:
            stored_models = self._repo.stored_embedding_models()
        except AppError:
            raise
        except Exception:  # repository doubles in tests / older repositories: no guard possible
            stored_models = set()
        if isinstance(stored_models, set) and stored_models and stored_models != {self._embedding.model_name}:
            raise AppError(
                ErrorCode.INDEXING_ERROR,
                "The vector store was built with a different embedding model; re-index is required "
                "before adding documents (see docs/RELEASE_ENGINEERING.md).",
            )


        # ── Embed ───────────────────────────────────────────────────
        embed_start = time.perf_counter()
        try:
            embeddings = await self._embedding.embed_documents(texts)
        except AppError:
            raise
        except Exception as exc:
            logger.error("Embedding failed for document_id=%s: %s", document_id, exc)
            raise AppError(
                ErrorCode.INDEXING_ERROR,
                "Failed to create embeddings for document chunks.",
            ) from exc
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # ── Prepare ChromaDB records ────────────────────────────────
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            # Deterministic ID: document_id + "_" + chunk_id
            record_id = f"{document_id}_{chunk.chunk_id}"
            ids.append(record_id)
            documents.append(chunk.text)

            meta: dict[str, Any] = {
                "document_id": document_id,
                "chunk_id": chunk.chunk_id,
                "source": source,
                "embedding_model": self._embedding.model_name,

            }
            # Store page info
            if chunk.metadata.page is not None:
                meta["page"] = chunk.metadata.page
            elif chunk.metadata.pages:
                meta["page"] = chunk.metadata.pages[0]
                meta["pages"] = str(chunk.metadata.pages)

            metadatas.append(meta)

        # ── Insert into ChromaDB ────────────────────────────────────
        insert_start = time.perf_counter()
        try:
            self._repo.add_documents(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            metrics.observe_stage("index_write", time.perf_counter() - insert_start)
            logger.error("ChromaDB insert failed for document_id=%s: %s", document_id, exc)
            raise AppError(
                ErrorCode.INDEXING_ERROR,
                "Failed to store document chunks in vector database.",
            ) from exc
        insert_ms = (time.perf_counter() - insert_start) * 1000
        metrics.observe_stage("index_write", insert_ms / 1000.0)

        total_ms = (time.perf_counter() - total_start) * 1000

        stats = {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "embedding_time_ms": round(embed_ms, 1),
            "chroma_insert_time_ms": round(insert_ms, 1),
            "total_time_ms": round(total_ms, 1),
            "model_name": self._embedding.model_name,
        }

        logger.info(
            "Indexed document | document_id=%s chunk_count=%d "
            "embedding_time_ms=%.1f chroma_insert_time_ms=%.1f "
            "total_time_ms=%.1f model=%s",
            document_id,
            len(chunks),
            embed_ms,
            insert_ms,
            total_ms,
            self._embedding.model_name,
        )
        return stats

    # ── Search ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Semantic search within a specific document.

        Embeds the query, then delegates to ChromaRepository.query().
        Returns a dict with ``results`` list and timing info.
        """
        total_start = time.perf_counter()

        # ── Embed query ─────────────────────────────────────────────
        embed_start = time.perf_counter()
        query_embedding = await self._embedding.embed_text(query)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # ── Retrieve ────────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        results = self._repo.query(
            query_embedding=query_embedding,
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Search completed | document_id=%s query_len=%d results=%d "
            "embedding_time_ms=%.1f retrieval_time_ms=%.1f total_time_ms=%.1f",
            document_id,
            len(query),
            len(results),
            embed_ms,
            retrieval_ms,
            total_ms,
        )

        return {
            "results": results,
            "embedding_time_ms": round(embed_ms, 1),
            "retrieval_time_ms": round(retrieval_ms, 1),
            "total_time_ms": round(total_ms, 1),
        }

    # ── Delete ──────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> int:
        """Delete all indexed chunks for a document."""
        return self._repo.delete_document(document_id)

    # ── Info ────────────────────────────────────────────────────────

    def get_document_info(self, document_id: str) -> dict[str, Any]:
        """Return stored info about a document."""
        return self._repo.get_document(document_id)

    def document_exists(self, document_id: str) -> bool:
        """Check if a document has been indexed."""
        return self._repo.document_exists(document_id)

    def count(self) -> int:
        """Total chunks across all documents."""
        return self._repo.count()

    def count_by_document(self, document_id: str) -> int:
        """Chunk count for a specific document."""
        return self._repo.count_by_document(document_id)
