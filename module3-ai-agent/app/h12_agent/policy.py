"""Role permissions, assessment mode boundaries, format checks and service auth for H12."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.types import (
    AgentRole,
    AssessmentMode,
    CallerScope,
    InstitutionalRef,
    RoleCapabilityView,
    StoppingReason,
    ToolName,
    VisibilityScope,
)

FORBIDDEN_ASSESSMENT_MODES: tuple[AssessmentMode, ...] = ("classroom_exam",)
FORBIDDEN_QUIZ_FORMATS: tuple[str, ...] = ("multiple_choice", "mcq", "true_false", "quiz")
FORBIDDEN_CHANNELS: tuple[str, ...] = ("voice", "audio", "speech", "oral")


@dataclass(frozen=True)
class RolePolicy:
    """Immutable boundary definition for one H12 agent role."""

    role: AgentRole
    allowed_scopes: frozenset[CallerScope]
    allowed_modes: frozenset[AssessmentMode]
    allowed_tools: tuple[ToolName, ...]
    visibility_scope: VisibilityScope
    stopping_conditions: tuple[StoppingReason, ...]


ROLE_POLICIES: dict[AgentRole, RolePolicy] = {
    "question_drafting": RolePolicy(
        role="question_drafting",
        allowed_scopes=frozenset({"teacher", "lcms_service"}),
        allowed_modes=frozenset({"teacher_question_drafting"}),
        allowed_tools=(
            "fetch_authorized_lesson",
            "retrieve_lesson_evidence",
            "record_question_draft",
        ),
        visibility_scope="teacher_authorized",
        stopping_conditions=("insufficient_lesson_material",),
    ),
    "grading_support": RolePolicy(
        role="grading_support",
        allowed_scopes=frozenset({"teacher", "lcms_service"}),
        allowed_modes=frozenset({"teacher_grading_support", "home_knowing"}),
        allowed_tools=(
            "fetch_authorized_lesson",
            "fetch_approved_questions_rubric",
            "fetch_assessment_context",
            "retrieve_lesson_evidence",
            "record_grading_suggestion",
        ),
        visibility_scope="teacher_authorized",
        stopping_conditions=("insufficient_lesson_material", "repeated_injection_escalation"),
    ),
    "dialogue_assessment": RolePolicy(
        role="dialogue_assessment",
        allowed_scopes=frozenset({"student", "lcms_service"}),
        allowed_modes=frozenset({"home_understanding"}),
        allowed_tools=(
            "fetch_authorized_lesson",
            "fetch_approved_questions_rubric",
            "fetch_assessment_context",
            "retrieve_lesson_evidence",
            "record_dialogue_outcome",
        ),
        visibility_scope="student_safe",
        stopping_conditions=(
            "criteria_covered",
            "max_turns_reached",
            "student_requested_closure",
            "repeated_injection_escalation",
            "insufficient_lesson_material",
        ),
    ),
}


def verify_service_auth(provided_key: Optional[str]) -> None:
    """Enforce X-Agent-Service-Key when AGENT_AUTH_REQUIRED=true or a secret is supplied."""
    settings = get_settings()
    secret = settings.agent_service_secret.strip()
    if settings.agent_auth_required:
        if not secret or not provided_key or not hmac.compare_digest(provided_key.strip(), secret):
            raise AppError(ErrorCode.AGENT_UNAUTHORIZED, "Invalid or missing agent service credential.")
    elif secret and provided_key is not None and not hmac.compare_digest(provided_key.strip(), secret):
        raise AppError(ErrorCode.AGENT_UNAUTHORIZED, "Invalid agent service credential.")


def _validate_format_and_channel(
    *,
    agent_role: AgentRole,
    assessment_mode: AssessmentMode,
    format_hint: Optional[str],
    channel: str,
) -> None:
    norm_channel = (channel or "").strip().lower()
    if norm_channel != "text_chat":
        raise AppError(
            ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT,
            "Voice/oral interaction is prohibited; Understanding requires interactive text_chat.",
        )

    if format_hint is None:
        return

    fmt = format_hint.strip().lower()
    if fmt in FORBIDDEN_QUIZ_FORMATS:
        raise AppError(
            ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT,
            "Multiple-choice substitution is prohibited; Knowing requires fixed lesson-based essay assessment.",
        )

    if agent_role in {"question_drafting", "grading_support"} or assessment_mode in {
        "home_knowing",
        "teacher_question_drafting",
        "teacher_grading_support",
    }:
        if fmt != "essay":
            raise AppError(
                ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT,
                f"Format '{fmt}' is not permitted for {assessment_mode}; only 'essay' is allowed.",
            )
    elif fmt not in {"adaptive_text_chat", "text_chat"}:
        raise AppError(
            ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT,
            f"Format '{fmt}' is not permitted for Understanding dialogue; only 'adaptive_text_chat' is allowed.",
        )


def enforce_mode_and_scope(
    *,
    agent_role: AgentRole,
    caller_scope: CallerScope,
    assessment_mode: AssessmentMode,
    institutional_ref: InstitutionalRef,
    format_hint: Optional[str],
    channel: str,
) -> RolePolicy:
    """Validate permitted assessment mode, format/channel rules, caller scope and session identifiers."""
    if assessment_mode in FORBIDDEN_ASSESSMENT_MODES:
        raise AppError(
            ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
            "AI assistance is strictly prohibited during controlled classroom examinations.",
        )

    _validate_format_and_channel(
        agent_role=agent_role,
        assessment_mode=assessment_mode,
        format_hint=format_hint,
        channel=channel,
    )

    policy = ROLE_POLICIES[agent_role]
    if caller_scope not in policy.allowed_scopes:
        raise AppError(
            ErrorCode.AGENT_SCOPE_FORBIDDEN,
            f"Caller scope '{caller_scope}' is not authorized for agent role '{agent_role}'.",
        )
    if assessment_mode not in policy.allowed_modes:
        raise AppError(
            ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
            f"Assessment mode '{assessment_mode}' is not permitted for agent role '{agent_role}'.",
        )
    if agent_role == "dialogue_assessment" and (
        not institutional_ref.student_id or not institutional_ref.attempt_id
    ):
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "student_id and attempt_id are required for isolated Understanding dialogue sessions.",
            status_code=422,
        )
    return policy


def resolve_allowed_tools(
    policy: RolePolicy,
    requested_tools: list[ToolName],
    *,
    has_document: bool,
) -> list[ToolName]:
    """Return permitted tools for this invocation, rejecting any out-of-scope requested tool."""
    allowed_set = set(policy.allowed_tools)
    if requested_tools:
        selected: list[ToolName] = []
        for tool in requested_tools:
            if tool not in allowed_set:
                raise AppError(
                    ErrorCode.AGENT_SCOPE_FORBIDDEN,
                    f"Tool '{tool}' is not permitted for role '{policy.role}'.",
                )
            if tool not in selected:
                selected.append(tool)
        return selected

    return [
        tool
        for tool in policy.allowed_tools
        if tool != "retrieve_lesson_evidence" or has_document
    ]


def list_role_capabilities() -> list[RoleCapabilityView]:
    """Serialize role policies for GET /api/agent/capabilities."""
    return [
        RoleCapabilityView(
            role=p.role,
            allowed_caller_scopes=sorted(p.allowed_scopes),  # type: ignore[arg-type]
            allowed_assessment_modes=sorted(p.allowed_modes),  # type: ignore[arg-type]
            allowed_tools=list(p.allowed_tools),
            visibility_scope=p.visibility_scope,
            stopping_conditions=list(p.stopping_conditions),
        )
        for p in ROLE_POLICIES.values()
    ]
