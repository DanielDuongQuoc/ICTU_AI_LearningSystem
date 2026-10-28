"""RAG API — Phase 4 retrieval only (no LLM)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.request_context import get_request_id
from app.schemas.rag import RAGAnswerResponse, RAGRetrievalRequest, RAGRetrievalResponse
from app.services.rag_service import RAGService, get_rag_service
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit

router = APIRouter(prefix="/api/rag", tags=["RAG"])


def _get_rag_service() -> RAGService:
    return get_rag_service()


@router.post(
    "/retrieve",
    response_model=RAGRetrievalResponse,
    summary="Retrieve relevant chunks and build context (no answer generation)",
    description=(
        "Embeds the question, searches ONLY the given document, filters by SIMILARITY_THRESHOLD, "
        "de-duplicates, ranks by score (best first) and builds a context capped at MAX_CONTEXT_CHARS. "
        "`insufficient_context` is true when nothing relevant was found. No LLM is called."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def retrieve(body: RAGRetrievalRequest, request: Request) -> RAGRetrievalResponse:
    result = await _get_rag_service().retrieve(
        document_id=body.document_id,
        question=body.question,
        top_k=body.top_k,
        request_id=get_request_id(request),
    )
    return RAGRetrievalResponse(**result)


@router.post(
    "/answer",
    response_model=RAGAnswerResponse,
    summary="Answer a question from the document (RAG + LLM)",
    description=(
        "Retrieves context then asks Qwen for a grounded answer. If the context is insufficient a fixed "
        "answer is returned and the LLM is not called. `sources` come from retrieval, never from the model."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def answer(body: RAGRetrievalRequest, request: Request) -> RAGAnswerResponse:
    result = await _get_rag_service().answer_question(
        document_id=body.document_id,
        question=body.question,
        top_k=body.top_k,
        request_id=get_request_id(request),
    )
    return RAGAnswerResponse(**result)
