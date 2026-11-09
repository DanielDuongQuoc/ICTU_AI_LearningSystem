"""Rubrics API: AI rubric generation and code-only rubric validation (for lecturer-edited rubrics)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit
from app.core.request_context import get_request_id
from app.grading.rubric_validator import validate_rubric
from app.schemas.grading import RubricGenerateRequest, RubricGenerateResponse, RubricValidateRequest
from app.schemas.rubric import RubricValidationResponse
from app.services.rubric_service import RubricService, get_rubric_service

router = APIRouter(prefix="/api/rubrics", tags=["Rubrics"])


def _get_rubric_service() -> RubricService:
    return get_rubric_service()


@router.post(
    "/generate",
    response_model=RubricGenerateResponse,
    summary="Generate an analytic rubric for a question",
    description=(
        "Drafts a rubric (criteria x levels) from the question, the reference answer and optional reference "
        "documents (`document_ids`) / structured `course_context`. Code rules R1, R2, R4, R6 are enforced (one "
        "repair attempt) and an optional AI review improves coverage (R3) and observable descriptors (R5). "
        "`homework` rubrics mark essential criteria `required`; `in_class` rubrics never do. The result is a "
        "`draft`: the lecturer must review it and set `status=approved` before grading."
    ),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def generate_rubric(body: RubricGenerateRequest, request: Request) -> RubricGenerateResponse:
    return await _get_rubric_service().generate(body, request_id=get_request_id(request))


@router.post(
    "/validate",
    response_model=RubricValidationResponse,
    summary="Validate a rubric (no AI call)",
    description=(
        "Checks a (lecturer-edited) rubric against the code rules: R1 points sum to max_score, R2 level "
        "structure, R4 no duplicate criteria, R6 grounded criteria carry sources, unique ids and, for "
        "`homework`, a pass rule that can actually fail a submission. Returns every violation at once."
    ),
    responses=error_responses(422, 429),
    dependencies=[Depends(rate_limit("cheap"))],
)
async def validate_rubric_endpoint(body: RubricValidateRequest) -> RubricValidationResponse:
    violations = validate_rubric(body.rubric, body.assessment_context)
    return RubricValidationResponse(valid=not violations, violations=violations)
