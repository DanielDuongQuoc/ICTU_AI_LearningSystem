"""Evaluations API — Phase 7 (JSON) and Phase 8 (SSE).  Thin router: validate, delegate, respond."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.request_context import get_request_id
from app.schemas.evaluation import EvaluationRequest, EvaluationResponse, EvaluationStreamRequest
from app.services.evaluation_service import EvaluationService, get_evaluation_service
from app.utils.sse import SSE_HEADERS, SSE_MEDIA_TYPE
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])


def _get_evaluation_service() -> EvaluationService:
    return get_evaluation_service()


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate a student answer (RAG + LLM)",
    description=(
        "Scores a student answer from 0 to `max_score` (<= 100) against the document context. The "
        "student answer is treated as untrusted data. `percentage` and `sources` are computed by the "
        "backend; the model only supplies score, correctness and feedback text."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def evaluate_answer(body: EvaluationRequest, request: Request) -> EvaluationResponse:
    return await _get_evaluation_service().evaluate_answer(
        document_id=body.document_id,
        question=body.question,
        student_answer=body.student_answer,
        expected_answer=body.expected_answer,
        max_score=body.max_score,
        request_id=get_request_id(request),
    )


@router.post(
    "/evaluate/stream",
    summary="Evaluate a student answer with Server-Sent Events (RAG + streaming LLM)",
    description=(
        "Same evaluation as `/evaluate`, streamed as `text/event-stream`. Frames are `data: <json>` with "
        "`type` in status | token | result | error | done. `result` appears at most once and before the "
        "single final `done`; failures are `error` events (code + message) followed by `done`. Field "
        "validation problems are also reported as `error` events, not HTTP 4xx."
    ),
    responses={
        200: {
            "description": "SSE stream.",
            "content": {"text/event-stream": {"schema": {"type": "string"}, "example": 'data: {"type":"status","message":"Validating request"}\n\ndata: {"type":"done"}\n\n'}},
        },
        **error_responses(422, 429),
    },
    dependencies=[Depends(rate_limit())],
    response_class=StreamingResponse,
)
async def evaluate_answer_stream(
    body: EvaluationStreamRequest, request: Request
) -> StreamingResponse:
    generator = _get_evaluation_service().evaluate_answer_stream(
        document_id=body.document_id,
        question=body.question,
        student_answer=body.student_answer,
        expected_answer=body.expected_answer,
        max_score=body.max_score,
        request_id=get_request_id(request),
    )
    return StreamingResponse(generator, media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
