"""Grading API: rubric-based grading. The grading mode is derived from ``assessment_context``:
homework -> PASS/FAIL (final unless flagged), in_class -> points pending lecturer review."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit
from app.core.request_context import get_request_id
from app.schemas.grading import GradeBatchRequest, GradeBatchResponse, GradeRequest, GradeResult
from app.services.grading_service import GradingService, get_grading_service

router = APIRouter(prefix="/api/grading", tags=["Grading"])

_POLICY = (
    "`assessment_context=homework`: the model judges each criterion, the backend decides PASS/FAIL "
    "(all `required` criteria met and met weight >= `pass_rule.min_percentage`); `status` is `final`, or "
    "`needs_review` when a review flag is raised (e.g. `injection_suspected`). No points are returned. "
    "`assessment_context=in_class`: the model picks a level per criterion, the backend sums the points into "
    "`suggested_total`; `status` is always `pending_review` - the lecturer decides the final score. "
    "Any `grading_mode` sent by the caller is ignored. The rubric must have `status=approved`."
)


def _get_grading_service() -> GradingService:
    return get_grading_service()


@router.post(
    "/grade",
    response_model=GradeResult,
    summary="Grade one submission against an approved rubric",
    description="Grades a typed answer, a handwriting transcript or concatenated source code. " + _POLICY,
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def grade(body: GradeRequest, request: Request) -> GradeResult:
    return await _get_grading_service().grade(body, request_id=get_request_id(request))


@router.post(
    "/batch",
    response_model=GradeBatchResponse,
    summary="Grade several submissions of the same question",
    description=("Same rules as `/grade` for up to `MAX_BATCH_SUBMISSIONS` submissions sharing one question and "
                 "rubric. Each item reports its own result or error; one failure does not fail the batch. "
                 + _POLICY),
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def grade_batch(body: GradeBatchRequest, request: Request) -> GradeBatchResponse:
    return await _get_grading_service().grade_batch(body, request_id=get_request_id(request))
