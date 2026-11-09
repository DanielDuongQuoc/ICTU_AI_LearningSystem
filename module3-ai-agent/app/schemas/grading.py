"""Request/response schemas for rubric generation and rubric-based grading."""

from __future__ import annotations

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.course import CourseContext
from app.schemas.rubric import AssessmentContext, Rubric, RubricViolation

SubmissionKind = Literal["typed", "handwritten_transcript", "code"]
MAX_DOCUMENT_IDS = 5


# ── Rubric generation ──────────────────────────────────────────────


class RubricGenerateRequest(BaseModel):
    assessment_context: AssessmentContext
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1, description="Reference answer / key ideas")
    max_score: float = Field(10, gt=0, le=100)
    document_ids: List[UUID] = Field(default_factory=list, max_length=MAX_DOCUMENT_IDS)
    course_context: Optional[CourseContext] = None
    language: str = Field("vi", pattern=r"^[a-z]{2}$")


class RubricGenerateResponse(BaseModel):
    rubric: Rubric
    review_applied: bool = Field(..., description="True when the AI review step revised the rubric")
    review_issues: List[RubricViolation] = Field(
        default_factory=list, description="R3/R4/R5 issues reported by the AI review (informational)")
    model: str
    prompt_version: str


class RubricValidateRequest(BaseModel):
    rubric: Rubric
    assessment_context: Optional[AssessmentContext] = None


# ── Grading ────────────────────────────────────────────────────────


class GradeRequest(BaseModel):
    """The grading mode is NOT a field: it is derived from ``assessment_context`` by the backend."""

    assessment_context: AssessmentContext
    submission_kind: SubmissionKind = "typed"
    question: str = Field(..., min_length=1)
    rubric: Rubric
    answer: str = Field(..., description="Typed answer, handwriting transcript, or concatenated source files")
    expected_answer: Optional[str] = None
    document_ids: List[UUID] = Field(default_factory=list, max_length=MAX_DOCUMENT_IDS)
    submission_ref: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._-]{1,64}$",
                                          description="Anonymous reference; never a student name")
    language: str = Field("vi", pattern=r"^[a-z]{2}$")
    test_results: Optional[str] = Field(None, max_length=10000, description="Sandbox test summary (code)")
    extra_flags: List[str] = Field(default_factory=list, max_length=10,
                                   description="Review flags raised before grading (e.g. by the Agent)")


class CriterionResult(BaseModel):
    criterion_id: str
    name: str
    required: bool
    level: str
    met: bool
    points: Optional[float] = Field(None, description="Only for in_class (score mode)")
    max_points: Optional[float] = Field(None, description="Only for in_class (score mode)")
    reason: str
    evidence: str


class GradeTrace(BaseModel):
    model: str
    prompt_version: str
    request_id: str
    grounded: bool


class GradeResult(BaseModel):
    submission_ref: Optional[str] = None
    assessment_context: AssessmentContext
    grading_mode: Literal["pass_fail", "score"]
    status: Literal["final", "needs_review", "pending_review"]
    result: Optional[Literal["PASS", "FAIL"]] = Field(None, description="Only for homework")
    unmet_required: List[str] = Field(default_factory=list)
    suggested_total: Optional[float] = Field(None, description="Only for in_class; the lecturer decides")
    max_score: Optional[float] = Field(None, description="Only for in_class")
    criteria: List[CriterionResult]
    feedback: str
    flags: List[str] = Field(default_factory=list, description="Raise review: homework -> needs_review")
    warnings: List[str] = Field(default_factory=list, description="Informational only")
    trace: GradeTrace


class BatchSubmission(BaseModel):
    submission_ref: str = Field(..., pattern=r"^[A-Za-z0-9._-]{1,64}$")
    answer: str
    submission_kind: SubmissionKind = "typed"
    test_results: Optional[str] = Field(None, max_length=10000)
    extra_flags: List[str] = Field(default_factory=list, max_length=10)


class GradeBatchRequest(BaseModel):
    assessment_context: AssessmentContext
    question: str = Field(..., min_length=1)
    rubric: Rubric
    expected_answer: Optional[str] = None
    document_ids: List[UUID] = Field(default_factory=list, max_length=MAX_DOCUMENT_IDS)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")
    submissions: List[BatchSubmission] = Field(..., min_length=1)


class BatchError(BaseModel):
    code: str
    message: str


class BatchItemResult(BaseModel):
    submission_ref: str
    ok: bool
    result: Optional[GradeResult] = None
    error: Optional[BatchError] = None


class GradeBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[BatchItemResult]
