"""Phase 4 — RAG retrieval & context construction.

Pipeline
~~~~~~~~
question → validate → document check → EmbeddingService.embed_text
→ ChromaRepository.query (filtered by document_id)
→ similarity-threshold filter → de-duplicate by chunk_id → rank
→ context construction (size-limited) → structured result

``retrieve()`` performs retrieval only and never calls a generative LLM.
``answer_question()`` (Phase 5) adds generation on top via the abstract
:class:`LLMService`; RAGService never talks to Ollama directly.  The LLM only
produces the answer text — sources always come from retrieval (ChromaDB metadata).

Score semantics
~~~~~~~~~~~~~~~
The collection is created with ``hnsw:space=cosine``; ChromaDB returns
cosine *distance* (``1 - cosine_similarity``).  ``ChromaRepository.query``
converts it to ``score = 1 - distance`` = cosine similarity, so higher is
better.  RAGService uses that ``score`` unchanged.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from app.core import runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core import prompt_registry
from app.core.execution_trace import trace_execution
from app.core.output_guard import enforce_no_prompt_leak
from app.core.prompts import INSUFFICIENT_CONTEXT_ANSWER
from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

CONTEXT_SEPARATOR = "\n\n"


# ── Pure helpers (unit-testable without any I/O) ────────────────────


def filter_by_threshold(chunks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Keep chunks whose similarity score is >= threshold."""
    return [c for c in chunks if c.get("score", 0.0) >= threshold]


def deduplicate(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated chunk_ids, keeping the highest-scoring occurrence."""
    best: dict[str, dict[str, Any]] = {}
    for c in chunks:
        cid = c["chunk_id"]
        if cid not in best or c["score"] > best[cid]["score"]:
            best[cid] = c
    return list(best.values())


def rank(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by score descending (stable; ties keep retrieval order)."""
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


def format_chunk(chunk: dict[str, Any]) -> str:
    page = chunk.get("page")
    page_label = f"page {page}" if page is not None else "page unknown"
    return f"[Source: {page_label} | chunk: {chunk['chunk_id']}]\n{chunk['text']}"


def build_context(
    ranked_chunks: list[dict[str, Any]], max_chars: int
) -> tuple[str, list[dict[str, Any]]]:
    """Build the context string from ranked chunks within ``max_chars``.

    Chunk boundaries are preserved: a chunk is either included whole or not
    at all.  Selection stops at the first chunk that does not fit, so
    lower-scored chunks never displace higher-scored ones.  Only if the very
    first (best) chunk alone exceeds the limit is its text truncated, so that
    the context is never empty when relevant material exists.

    Returns ``(context, included_chunks)``.
    """
    parts: list[str] = []
    included: list[dict[str, Any]] = []
    used = 0

    for chunk in ranked_chunks:
        block = format_chunk(chunk)
        extra = len(block) + (len(CONTEXT_SEPARATOR) if parts else 0)

        if used + extra > max_chars:
            if not parts:
                header = format_chunk({**chunk, "text": ""})
                room = max_chars - len(header)
                if room > 0:
                    truncated = {**chunk, "text": chunk["text"][:room]}
                    parts.append(format_chunk(truncated))
                    included.append(truncated)
            break

        parts.append(block)
        included.append(chunk)
        used += extra

    return CONTEXT_SEPARATOR.join(parts), included


# ── Service ─────────────────────────────────────────────────────────


class RAGService:
    """Retrieval orchestration on top of EmbeddingService + ChromaRepository."""

    def __init__(
        self,
        *,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_repo: Optional[ChromaRepository] = None,
        llm_service: Optional[LLMService] = None,
        similarity_threshold: Optional[float] = None,
        max_context_chars: Optional[int] = None,
        max_top_k: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._embedding = embedding_service or get_embedding_service()
        self._repo = chroma_repo or ChromaRepository()
        self._llm = llm_service  # resolved lazily; retrieval-only use never needs it
        # Phase 18: None = follow the effective (active AI version) configuration on every call;
        # an explicit constructor value (tests, evaluation harness) always wins.
        self._threshold: Optional[float] = similarity_threshold
        self._max_chars: Optional[int] = max_context_chars
        self._max_top_k = max_top_k if max_top_k is not None else settings.max_top_k
        self._default_top_k: Optional[int] = None

    def _validate(self, question: str, top_k: Optional[int]) -> tuple[str, int]:
        if question is None or not question.strip():
            raise AppError(ErrorCode.EMPTY_QUERY, "Question must not be empty.")
        max_chars = get_settings().max_question_chars
        if len(question.strip()) > max_chars:
            raise AppError(ErrorCode.INPUT_TOO_LONG, f"Question must be at most {max_chars} characters.")
        k = (self._default_top_k or runtime_config.current().top_k) if top_k is None else top_k
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > self._max_top_k:
            raise AppError(
                ErrorCode.INVALID_TOP_K,
                f"top_k must be between 1 and {self._max_top_k}.",
            )
        return question.strip(), k

    async def retrieve(
        self,
        document_id: str,
        question: str,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and build a context for ``question``."""
        request_id = request_id or str(uuid.uuid4())
        total_start = time.perf_counter()
        eff = runtime_config.current()
        threshold = self._threshold if self._threshold is not None else eff.similarity_threshold
        max_chars = self._max_chars if self._max_chars is not None else eff.max_context_chars

        # 1. Validate (no embedding / Chroma calls on invalid input)
        question, k = self._validate(question, top_k)

        try:
            # 2. Document existence (before any embedding work)
            if not self._repo.document_exists(document_id):
                raise AppError(
                    ErrorCode.DOCUMENT_NOT_FOUND,
                    f"Document '{document_id}' not found.",
                )

            # 3. Embed question
            t0 = time.perf_counter()
            try:
                query_embedding = await self._embedding.embed_text(question)
            except AppError:
                raise
            except Exception as exc:
                logger.error("Question embedding failed request_id=%s: %s", request_id, exc)
                raise AppError(ErrorCode.EMBEDDING_ERROR, "Failed to embed the question.") from exc
            embed_ms = (time.perf_counter() - t0) * 1000

            # 4. Chroma query (filtered by document_id inside the repository)
            t0 = time.perf_counter()
            try:
                retrieved = self._repo.query(
                    query_embedding=query_embedding,
                    document_id=document_id,
                    top_k=k,
                    similarity_threshold=threshold,
                )
            except AppError:
                raise
            except Exception as exc:
                logger.error("ChromaDB query failed request_id=%s: %s", request_id, exc)
                raise AppError(ErrorCode.CHROMA_ERROR, "Vector database query failed.") from exc
            query_ms = (time.perf_counter() - t0) * 1000
            metrics.observe_stage("chroma_query", query_ms / 1000.0)

            # 5. Filter / de-duplicate / rank
            t0 = time.perf_counter()
            # Defence in depth: never trust cross-document hits, whatever the repo returns.
            scoped = [c for c in retrieved if c.get("document_id", document_id) == document_id]
            passed = filter_by_threshold(scoped, threshold)
            unique = deduplicate(passed)
            ranked = rank(unique)[:k]
            filter_ms = (time.perf_counter() - t0) * 1000

            # 6. Context construction
            t0 = time.perf_counter()
            context, included = build_context(ranked, max_chars)
            context_ms = (time.perf_counter() - t0) * 1000
            metrics.observe_stage("context_build", context_ms / 1000.0)
        except AppError as exc:
            if exc.status_code >= 500:
                metrics.event("rag_retrieval", "error")
            raise
        except Exception as exc:
            logger.error("RAG retrieval failed request_id=%s: %s", request_id, exc, exc_info=True)
            metrics.event("rag_retrieval", "error")
            raise AppError(ErrorCode.RAG_RETRIEVAL_ERROR, "Retrieval failed.") from exc

        insufficient = len(included) == 0
        metrics.event("rag_retrieval", "no_context" if insufficient else "ok")
        chunks = [
            {
                "chunk_id": c["chunk_id"],
                "document_id": document_id,
                "page": c.get("page"),
                "source": c.get("source", ""),
                "score": c["score"],
                "text": c["text"],
            }
            for c in included
        ]
        sources = [{k_: v for k_, v in c.items() if k_ != "text"} for c in chunks]

        total_ms = (time.perf_counter() - total_start) * 1000
        scores = [c["score"] for c in included]
        highest = max(scores) if scores else None
        lowest = min(scores) if scores else None
        logger.info(
            "RAG retrieval | request_id=%s document_id=%s query_len=%d top_k=%d threshold=%s "
            "retrieved=%d filtered=%d retrieved_chunks=%d context_chars=%d "
            "highest_score=%s lowest_score=%s insufficient_context=%s "
            "embedding_ms=%.1f chroma_query_ms=%.1f filter_ms=%.2f context_build_ms=%.2f "
            "retrieval_time_ms=%.1f total_rag_time_ms=%.1f total_ms=%.1f",
            request_id, document_id, len(question), k, threshold,
            len(retrieved), len(passed), len(included), len(context),
            highest, lowest, insufficient,
            embed_ms, query_ms, filter_ms, context_ms, total_ms, total_ms, total_ms,
        )

        return {
            "document_id": document_id,
            "question": question,
            "insufficient_context": insufficient,
            "chunks": chunks,
            "context": context,
            "sources": sources,
        }


    # ── Phase 5: retrieval + generation ─────────────────────────────

    async def answer_question(
        self,
        document_id: str,
        question: str,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve context, then generate a grounded answer with the LLM.

        * Insufficient context → fixed answer, the LLM is NOT called.
        * The LLM produces only ``answer``; ``sources`` come from retrieval and
          any other fields the model returns are ignored.
        """
        request_id = request_id or str(uuid.uuid4())
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        retrieval = await self.retrieve(document_id, question, top_k, request_id=request_id)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if retrieval["insufficient_context"]:
            logger.info(
                "RAG answer | request_id=%s document_id=%s insufficient_context=True "
                "llm_called=False retrieval_ms=%.1f total_ms=%.1f",
                request_id, document_id, retrieval_ms, (time.perf_counter() - total_start) * 1000,
            )
            trace_execution(prompt_registry.TASK_RAG_ANSWER, request_id, document_id,
                            "insufficient_context", (time.perf_counter() - total_start) * 1000, llm_called=False)
            return {
                "document_id": document_id,
                "question": retrieval["question"],
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "insufficient_context": True,
                "sources": [],
            }

        if self._llm is None:
            self._llm = get_llm_service()

        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_RAG_ANSWER,
            runtime_config.current().prompt_version(prompt_registry.TASK_RAG_ANSWER),
            retrieval["context"], retrieval["question"],
        )

        t0 = time.perf_counter()
        try:
            parsed = await self._llm.generate_json(
                system_prompt, user_prompt, required_keys=["answer"]
            )
        except AppError:
            raise
        except Exception as exc:
            logger.error("LLM generation failed request_id=%s: %s", request_id, exc, exc_info=True)
            raise AppError(
                ErrorCode.OLLAMA_GENERATION_ERROR, "Answer generation failed."
            ) from exc
        llm_ms = (time.perf_counter() - t0) * 1000

        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty answer.")
        enforce_no_prompt_leak(answer)  # Phase 19: model output is untrusted

        logger.info(
            "RAG answer | request_id=%s document_id=%s insufficient_context=False llm_called=True "
            "sources=%d retrieval_ms=%.1f llm_ms=%.1f total_ms=%.1f",
            request_id, document_id, len(retrieval["sources"]),
            retrieval_ms, llm_ms, (time.perf_counter() - total_start) * 1000,
        )

        trace_execution(
            prompt_registry.TASK_RAG_ANSWER, request_id, document_id, "ok",
            (time.perf_counter() - total_start) * 1000,
            source_chunk_ids=[s["chunk_id"] for s in retrieval["sources"]], llm_called=True,
        )
        return {
            "document_id": document_id,
            "question": retrieval["question"],
            "answer": answer.strip(),
            "insufficient_context": False,
            "sources": retrieval["sources"],  # from retrieval only — never from the LLM
        }


# ── Module-level singleton ──────────────────────────────────────────

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return a lazily-created RAGService (reuses embedding service & Chroma client)."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
