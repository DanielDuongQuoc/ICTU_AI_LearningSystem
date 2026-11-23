"""Versioned lesson knowledge service with visibility isolation (student_safe vs teacher_only)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core import runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.types import EvidenceRef, VisibilityScope
from app.repositories.chroma_repository import ChromaRepository
from app.schemas.document import ExtractedPage
from app.services.chunk_service import create_chunks
from app.services.embedding_service import GeminiEmbeddingService, get_embedding_service
from app.services.rag_service import build_context, deduplicate, filter_by_threshold, rank
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class LessonIndexResult(BaseModel):
    """Summary returned after indexing one versioned lesson into ChromaDB."""

    lesson_id: str
    knowledge_version: str
    approved: bool
    student_safe_chunks: int
    teacher_only_chunks: int
    total_chunks: int


class LessonRetrievalResult(BaseModel):
    """Grounded chunks and bounded context retrieved for one lesson version and visibility scope."""

    lesson_id: str
    knowledge_version: str
    visibility_scope: VisibilityScope
    insufficient_context: bool
    context: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class LessonKnowledgeService:
    """Manages versioned, approved lesson material and isolates teacher answer keys from students."""

    def __init__(
        self,
        *,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        chroma_repo: Optional[ChromaRepository] = None,
        vector_service: Optional[VectorService] = None,
        similarity_threshold: Optional[float] = None,
        max_context_chars: Optional[int] = None,
    ) -> None:
        self._embedding = embedding_service
        self._repo = chroma_repo
        self._vector = vector_service
        self._threshold = similarity_threshold
        self._max_chars = max_context_chars

    def _embedder(self) -> GeminiEmbeddingService:
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding

    def _repository(self) -> ChromaRepository:
        if self._repo is None:
            self._repo = ChromaRepository()
        return self._repo

    def _vector_service(self) -> VectorService:
        if self._vector is None:
            self._vector = VectorService(
                embedding_service=self._embedder(),
                chroma_repo=self._repository(),
            )
        return self._vector

    async def index_lesson_material(
        self,
        *,
        lesson_id: str,
        knowledge_version: str,
        title: str,
        student_safe_content: str,
        approved: bool,
        teacher_only_notes: Optional[str] = None,
    ) -> LessonIndexResult:
        """Index teacher-approved lesson content partitioned into student_safe and teacher_only chunks."""
        if not approved:
            raise AppError(
                ErrorCode.UNAPPROVED_KNOWLEDGE,
                "Only teacher-approved lesson material may be indexed for H12 assessment.",
            )
        clean_student = (student_safe_content or "").strip()
        if not clean_student:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "student_safe_content must not be empty.",
                status_code=422,
            )

        repo = self._repository()
        vector = self._vector_service()

        # Replace only the target (lesson_id, knowledge_version); keep all other versions intact.
        repo.delete_lesson_version(lesson_id, knowledge_version)

        source_label = f"{lesson_id}:{knowledge_version}:{title.strip() or 'lesson'}"
        student_doc_id = f"lesson:{lesson_id}:{knowledge_version}:student"
        student_chunks = create_chunks(
            [ExtractedPage(page=1, text=clean_student)],
            student_doc_id,
            source_label,
        )
        await vector.index_chunks(
            chunks=student_chunks,
            document_id=student_doc_id,
            source=source_label,
            extra_metadata={
                "lesson_id": lesson_id,
                "knowledge_version": knowledge_version,
                "approved": True,
                "visibility": "student_safe",
            },
        )

        teacher_chunks_count = 0
        clean_teacher = (teacher_only_notes or "").strip()
        if clean_teacher:
            teacher_doc_id = f"lesson:{lesson_id}:{knowledge_version}:teacher"
            teacher_chunks = create_chunks(
                [ExtractedPage(page=1, text=clean_teacher)],
                teacher_doc_id,
                f"{source_label}:teacher_only",
            )
            await vector.index_chunks(
                chunks=teacher_chunks,
                document_id=teacher_doc_id,
                source=f"{source_label}:teacher_only",
                extra_metadata={
                    "lesson_id": lesson_id,
                    "knowledge_version": knowledge_version,
                    "approved": True,
                    "visibility": "teacher_only",
                },
            )
            teacher_chunks_count = len(teacher_chunks)

        logger.info(
            "Indexed lesson knowledge | lesson_id=%s version=%s student_chunks=%d teacher_chunks=%d",
            lesson_id,
            knowledge_version,
            len(student_chunks),
            teacher_chunks_count,
        )
        return LessonIndexResult(
            lesson_id=lesson_id,
            knowledge_version=knowledge_version,
            approved=True,
            student_safe_chunks=len(student_chunks),
            teacher_only_chunks=teacher_chunks_count,
            total_chunks=len(student_chunks) + teacher_chunks_count,
        )

    async def retrieve_lesson_evidence(
        self,
        *,
        lesson_id: str,
        knowledge_version: str,
        query: str,
        visibility_scope: VisibilityScope,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> LessonRetrievalResult:
        """Retrieve version-scoped, visibility-enforced evidence for a lesson."""
        clean_query = (query or "").strip()
        if not clean_query:
            raise AppError(ErrorCode.EMPTY_QUERY, "Query must not be empty.")
        settings = get_settings()
        if len(clean_query) > settings.max_question_chars:
            raise AppError(
                ErrorCode.INPUT_TOO_LONG,
                f"Query must be at most {settings.max_question_chars} characters.",
            )

        eff = runtime_config.current()
        k = eff.top_k if top_k is None else top_k
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > settings.max_top_k:
            raise AppError(
                ErrorCode.INVALID_TOP_K,
                f"top_k must be between 1 and {settings.max_top_k}.",
            )

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else (self._threshold if self._threshold is not None else eff.similarity_threshold)
        )
        max_chars = self._max_chars if self._max_chars is not None else eff.max_context_chars

        query_embedding = await self._embedder().embed_text(clean_query)
        retrieved = self._repository().query_lesson(
            query_embedding=query_embedding,
            lesson_id=lesson_id,
            knowledge_version=knowledge_version,
            visibility_scope=visibility_scope,
            require_approved=True,
            top_k=k,
            similarity_threshold=threshold,
        )

        # Defense in depth: verify version, approval, and visibility boundaries in Python as well.
        scoped = []
        for chunk in retrieved:
            if chunk.get("lesson_id") != lesson_id or chunk.get("knowledge_version") != knowledge_version:
                continue
            if chunk.get("approved") is not True:
                continue
            if visibility_scope == "student_safe" and chunk.get("visibility") != "student_safe":
                raise AppError(
                    ErrorCode.SOURCE_INTEGRITY_ERROR,
                    "Teacher-only material leaked into student_safe retrieval.",
                )
            scoped.append(chunk)

        filtered = filter_by_threshold(scoped, threshold)
        ranked = rank(deduplicate(filtered))[:k]
        context, included = build_context(ranked, max_chars)

        refs = [
            EvidenceRef(
                chunk_id=c["chunk_id"],
                document_id=str(c["document_id"]),
                page=c.get("page"),
                score=float(c["score"]),
            )
            for c in included
        ]
        return LessonRetrievalResult(
            lesson_id=lesson_id,
            knowledge_version=knowledge_version,
            visibility_scope=visibility_scope,
            insufficient_context=(len(included) == 0),
            context=context,
            evidence_refs=refs,
            chunks=included,
        )

    def list_versions(self, lesson_id: str) -> list[dict[str, Any]]:
        return self._repository().list_lesson_versions(lesson_id)
