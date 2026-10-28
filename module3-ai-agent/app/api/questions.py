"""Questions API — Phase 6.  Thin router: validate, delegate, respond."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.request_context import get_request_id
from app.schemas.question import (
    ExamMatrixRequest,
    ExamMatrixResponse,
    QuestionGenerateRequest,
    QuestionGenerationResponse,
)
from app.services.question_service import QuestionGenerationService, get_question_service
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit

router = APIRouter(prefix="/api/questions", tags=["Questions"])


def _get_question_service() -> QuestionGenerationService:
    return get_question_service()


@router.post(
    "/generate",
    response_model=QuestionGenerationResponse,
    response_model_exclude_none=True,
    summary="Generate questions from a document (RAG + LLM)",
    description=(
        "Generates up to `number_of_questions` (1-20) questions grounded in the document. Supported "
        "`question_type`: short_answer, multiple_choice, true_false. `difficulty`: easy, medium, hard. "
        "`question_id` and `source_chunks` are derived by the backend; duplicates, malformed items and "
        "questions without document support are dropped (fewer questions may be returned)."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def generate_questions(
    body: QuestionGenerateRequest, request: Request
) -> QuestionGenerationResponse:
    return await _get_question_service().generate_questions(
        document_id=body.document_id,
        number_of_questions=body.number_of_questions,
        difficulty=body.difficulty,
        question_type=body.question_type,
        request_id=get_request_id(request),
    )


@router.post(
    "/generate-matrix",
    response_model=ExamMatrixResponse,
    response_model_exclude_none=True,
    summary="Generate an exam from a matrix (type x difficulty x count) for instructor review",
    description=(
        "Module 1 sends the document, an optional `subject` label and `items`, each a "
        "`{question_type, difficulty, count}` row. Rows are generated one after another and de-duplicated across "
        "rows. The response is for instructor review before saving to the question bank: every question carries its "
        "`matrix_item` index, backend-derived `source_chunks` and `question_id`, and `items` reports per row "
        "`requested` / `generated` / `status` (`ok`, `partial`, `failed` + `error_code`). Nothing is padded or "
        "invented to reach the requested number; `complete` is true only when every row is satisfied. The total is "
        "limited by MAX_MATRIX_QUESTIONS and each row by 20. Nothing is stored by Module 3."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def generate_exam_matrix(body: ExamMatrixRequest, request: Request) -> ExamMatrixResponse:
    return await _get_question_service().generate_exam(
        document_id=body.document_id,
        items=body.items,
        subject=body.subject,
        request_id=get_request_id(request),
    )
