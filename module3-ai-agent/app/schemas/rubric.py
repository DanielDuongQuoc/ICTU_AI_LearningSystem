"""Rubric schema (analytic rubric: criteria x levels) shared by rubric generation, validation and grading.

Structural rules that need the whole rubric (sum of points, level ordering, duplicates, grounding) are NOT
Pydantic errors: they are reported as a list of violations by :func:`app.grading.rubric_validator.validate_rubric`
so a lecturer-edited rubric gets every problem at once instead of a single 422.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

AssessmentContext = Literal["homework", "in_class"]
RubricStatus = Literal["draft", "approved"]

MAX_CRITERIA = 20
MAX_LEVELS = 6
MAX_KEY_POINTS = 20


def _clean(v: str) -> str:
    return " ".join(v.split())


class RubricLevel(BaseModel):
    label: str = Field(..., min_length=1, max_length=60, description="e.g. Đầy đủ / Phần lớn / Một phần / Không đạt")
    points: float = Field(..., ge=0, le=100)
    descriptor: str = Field(..., min_length=1, max_length=600, description="Observable condition for this level")

    @field_validator("label", "descriptor")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = _clean(v)
        if not v:
            raise ValueError("must not be blank")
        return v


class RubricCriterion(BaseModel):
    criterion_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,16}$")
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    max_points: float = Field(..., gt=0, le=100)
    required: bool = Field(False, description="Must be met for a homework submission to PASS")
    key_points: List[str] = Field(default_factory=list, max_length=MAX_KEY_POINTS)
    levels: List[RubricLevel] = Field(..., min_length=1, max_length=MAX_LEVELS,
                                      description="Ordered from the highest to the lowest level")
    clo: Optional[str] = Field(None, max_length=32)
    bloom_level: Optional[str] = Field(None, max_length=32)
    source_chunks: List[str] = Field(default_factory=list, description="Set by the backend, never by the model")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = _clean(v)
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("key_points")
    @classmethod
    def _strip_points(cls, v: List[str]) -> List[str]:
        return [p for p in (_clean(x)[:500] for x in v) if p]


class PassRule(BaseModel):
    """Homework verdict: PASS iff (every required criterion is met, when enabled) AND met weight >= min_percentage."""

    all_required_met: bool = True
    min_percentage: float = Field(50.0, ge=0, le=100)


class Rubric(BaseModel):
    max_score: float = Field(10, gt=0, le=100)
    criteria: List[RubricCriterion] = Field(..., min_length=1, max_length=MAX_CRITERIA)
    pass_rule: PassRule = Field(default_factory=PassRule)
    grounded: bool = True
    status: RubricStatus = "draft"


class RubricViolation(BaseModel):
    rule: str = Field(..., description="R1 | R2 | R3 | R4 | R5 | R6 | ID | PASS_RULE")
    criterion_id: Optional[str] = None
    message: str


class RubricValidationResponse(BaseModel):
    valid: bool
    violations: List[RubricViolation] = Field(default_factory=list)
