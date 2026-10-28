from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, field_validator


class EvaluationSource(BaseModel):
    """Source metadata derived by the backend from RAG retrieval (never from the LLM)."""

    page: Optional[int] = None
    chunk_id: str
    score: float = Field(..., description="Similarity score from ChromaDB")


# Backward-compatible alias (name used before Phase 7).
SourceReference = EvaluationSource


class EvaluationRequest(BaseModel):
    document_id: UUID = Field(..., description="UUID of the indexed document")
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    max_score: float = Field(10, gt=0, le=100)

    @field_validator("question", "student_answer", "expected_answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class EvaluationResponse(BaseModel):
    score: float
    max_score: float
    percentage: float
    correct: bool
    feedback: str
    strengths: List[str] = Field(default_factory=list)
    missing_points: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    sources: List[EvaluationSource] = Field(default_factory=list)


# ── Internal schema (validates untrusted LLM JSON) ──────────────────


class LLMEvaluationResult(BaseModel):
    """Strict shape of the LLM output.  Unknown keys (e.g. ``sources``) are ignored.

    ``score`` is a strict number (no bool / numeric strings); ``correct`` a strict bool.
    """

    score: StrictInt | StrictFloat
    correct: StrictBool
    feedback: str
    strengths: List[str]
    missing_points: List[str]
    errors: List[str]

    @field_validator("feedback")
    @classmethod
    def _feedback_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("feedback must not be empty")
        return v.strip()


class EvaluationStreamRequest(BaseModel):
    """Lenient request model for the SSE endpoint.

    Field problems (empty answer, bad score, malformed id...) must be reported as SSE
    ``error`` events rather than an HTTP 4xx JSON body, so they are validated by
    ``EvaluationService`` instead of by Pydantic constraints.
    """

    document_id: str = ""
    question: str = ""
    student_answer: str = ""
    expected_answer: str = ""
    max_score: float = 10
