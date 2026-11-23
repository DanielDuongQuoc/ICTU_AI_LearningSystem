"""Typed contracts for the H12 AI Agent module (PI 1.1 - PI 3.4)."""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

AgentRole = Literal["question_drafting", "grading_support", "dialogue_assessment"]
CallerScope = Literal["teacher", "student", "lcms_service"]
AssessmentMode = Literal[
    "teacher_question_drafting",
    "teacher_grading_support",
    "home_knowing",
    "home_understanding",
    "classroom_exam",
]
ToolName = Literal[
    "fetch_authorized_lesson",
    "fetch_approved_questions_rubric",
    "fetch_assessment_context",
    "retrieve_lesson_evidence",
    "record_question_draft",
    "record_grading_suggestion",
    "record_dialogue_outcome",
]
VisibilityScope = Literal["student_safe", "teacher_authorized"]
PublicationStatus = Literal["draft_suggestion"]
StoppingReason = Literal[
    "criteria_covered",
    "max_turns_reached",
    "student_requested_closure",
    "repeated_injection_escalation",
    "insufficient_lesson_material",
]


class InstitutionalRef(BaseModel):
    """Stable LCMS/institutional identifiers carried across tools and session memory."""

    lesson_id: str = Field(..., pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    lesson_version: str = Field("v1", pattern=r"^[A-Za-z0-9._:-]{1,32}$")
    course_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    assignment_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    rubric_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    student_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    attempt_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")


class EvidenceRef(BaseModel):
    """Backend-derived reference to an indexed lesson chunk."""

    chunk_id: str
    document_id: str
    page: Optional[int] = None
    score: float


class ObservableStep(BaseModel):
    """One recorded step of the controlled agent graph (never hidden model reasoning)."""

    step: str
    action: str
    outcome: Literal["ok", "denied", "skipped", "mocked"]
    detail: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class AgentTrace(BaseModel):
    """Observable execution metadata distinguishing mock vs live model execution."""

    request_id: str
    framework: str = "langgraph"
    framework_version: str
    adapter_mode: Literal["mock", "live"]
    is_mock: bool
    provider: str
    model_name: str
    agent_role: AgentRole
    caller_scope: CallerScope
    assessment_mode: AssessmentMode
    visibility_scope: VisibilityScope
    allowed_tools: list[ToolName]
    selected_tools: list[ToolName]
    publication_status: PublicationStatus = "draft_suggestion"
    released: bool = False
    teacher_review_required: bool = True
    observable_steps: list[ObservableStep]


class AgentSkeletonRequest(BaseModel):
    """Controlled H12 agent skeleton invocation request."""

    agent_role: AgentRole
    caller_scope: CallerScope
    assessment_mode: AssessmentMode
    institutional_ref: InstitutionalRef
    prompt_input: str = Field(..., min_length=1, max_length=4000)
    document_id: Optional[UUID] = None
    requested_tools: list[ToolName] = Field(default_factory=list)
    format_hint: Optional[str] = Field(None, max_length=32)
    channel: str = Field("text_chat", max_length=32)
    idempotency_key: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")

    @field_validator("prompt_input")
    @classmethod
    def _strip_prompt_input(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("prompt_input must not be blank.")
        return cleaned


class AgentSkeletonResponse(BaseModel):
    """Controlled H12 agent output (always draft suggestion, never silent grade publication)."""

    status: Literal["ok"] = "ok"
    institutional_ref: InstitutionalRef
    publication_status: PublicationStatus = "draft_suggestion"
    released: bool = False
    teacher_review_required: bool = True
    idempotent_replay: bool = False
    draft_output: dict[str, Any]
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    stopping_conditions: list[StoppingReason]
    trace: AgentTrace


class RoleCapabilityView(BaseModel):
    """Published role boundary, permitted tools and stopping conditions."""

    role: AgentRole
    allowed_caller_scopes: list[CallerScope]
    allowed_assessment_modes: list[AssessmentMode]
    allowed_tools: list[ToolName]
    visibility_scope: VisibilityScope
    stopping_conditions: list[StoppingReason]
    output_publication_status: PublicationStatus = "draft_suggestion"
    auto_release_allowed: bool = False


class AgentCapabilitiesResponse(BaseModel):
    """Runtime capability manifest of the H12 AI agent service."""

    framework: str = "langgraph"
    framework_version: str
    adapter_mode: Literal["mock", "live"]
    is_mock: bool
    lcms_mode: Literal["mock", "http"]
    auth_required: bool
    forbidden_modes: list[AssessmentMode]
    forbidden_formats: list[str]
    forbidden_channels: list[str]
    roles: list[RoleCapabilityView]
