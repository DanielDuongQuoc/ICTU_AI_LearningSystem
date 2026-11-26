"""LangGraph workflow 3: Understanding adaptive text chat & dialogue security (Milestone 4)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core import prompt_registry, runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.output_guard import clamp_text
from app.h12_agent.dialogue_guards import (
    MAX_DIALOGUE_REPLY_CHARS,
    SecurityInspection,
    build_refusal_reply,
    guard_assistant_reply,
    inspect_student_message,
)
from app.h12_agent.knowledge import LessonKnowledgeService
from app.h12_agent.lcms_tools import (
    AssessmentContextRecord,
    DialogueOutcomeRecordInput,
    DraftRecordReceipt,
    LCMSToolClient,
    LessonMaterialRecord,
    LessonQuestionsRubricBundle,
    ToolOutcome,
)
from app.h12_agent.model_adapter import AgentModelAdapter
from app.h12_agent.policy import RolePolicy, enforce_mode_and_scope
from app.h12_agent.session_store import SessionMemoryStore, SessionRecord
from app.h12_agent.tool_executor import AuthorizedToolExecutor
from app.h12_agent.types import (
    AgentTrace,
    AssessmentMode,
    CallerScope,
    EvidenceRef,
    InstitutionalRef,
    ObservableStep,
    StoppingReason,
    ToolName,
)

logger = logging.getLogger("app.ai_trace")

MAX_UNDERSTANDING_MODEL_ATTEMPTS = 2
MAX_REPEATED_INJECTIONS = 2
_VALID_STATUSES = frozenset({"uncovered", "partial", "covered"})
_VALID_DECISIONS = frozenset({"follow_up", "close_session"})
_FORBIDDEN_LEAK_MARKERS = (
    "answer_key_confidential",
    "official answer key is",
    "confidential answer key",
    "model answer key:",
    "expected_answer:",
    "đáp án chính thức là",
    "bài giải mẫu chính thức",
)


class CriterionUnderstandingStatus(BaseModel):
    """Per-criterion comprehension state tracked across turns of an Understanding session."""

    criterion_id: str
    name: str
    description: str = ""
    required: bool = False
    status: Literal["uncovered", "partial", "covered"] = "uncovered"
    reason: str = ""
    evidence_quote: str = ""


class StartUnderstandingSessionRequest(BaseModel):
    """Request to open or resume an isolated Understanding adaptive text chat session."""

    caller_scope: CallerScope = "student"
    assessment_mode: AssessmentMode = "home_understanding"
    channel: str = Field("text_chat", max_length=32)
    institutional_ref: InstitutionalRef
    topic_focus: Optional[str] = Field(None, max_length=500)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")
    max_turns: Optional[int] = Field(None, ge=1, le=20)


class StartUnderstandingSessionResponse(BaseModel):
    """Session bootstrap payload with opening adaptive question and rubric criteria."""

    status: Literal["active", "closed", "escalated"] = "active"
    session_id: str
    created: bool
    institutional_ref: InstitutionalRef
    lesson_title: str
    rubric_id: str
    max_turns: int
    turn_count: int = 0
    opening_prompt: str
    rubric_progress: list[CriterionUnderstandingStatus]
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    trace: AgentTrace


class UnderstandingTurnRequest(BaseModel):
    """Student text turn inside an active Understanding adaptive dialogue session."""

    caller_scope: CallerScope = "student"
    assessment_mode: AssessmentMode = "home_understanding"
    channel: str = Field("text_chat", max_length=32)
    student_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=8000)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")
    idempotency_key: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")


class UnderstandingTurnResponse(BaseModel):
    """Adaptive Understanding turn output with updated rubric tracking and security status."""

    status: Literal["active", "closed", "escalated"]
    session_id: str
    turn_index: int
    turn_count: int
    max_turns: int
    decision: Literal["follow_up", "close_session"]
    stopping_reason: Optional[StoppingReason] = None
    assistant_reply: str
    turn_feedback: str
    rubric_progress: list[CriterionUnderstandingStatus]
    flags: list[str] = Field(default_factory=list)
    answer_leak_prevented: bool = False
    idempotent_replay: bool = False
    teacher_review_required: bool = True
    publication_status: Literal["draft_suggestion"] = "draft_suggestion"
    released: bool = False
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    lcms_receipt: Optional[DraftRecordReceipt] = None
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    trace: AgentTrace


class CloseUnderstandingSessionRequest(BaseModel):
    """Explicit student/LCMS request to conclude an active Understanding session."""

    caller_scope: CallerScope = "student"
    assessment_mode: AssessmentMode = "home_understanding"
    student_id: str = Field(..., min_length=1, max_length=64)
    reason: Optional[str] = Field(None, max_length=300)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")


class CloseUnderstandingSessionResponse(BaseModel):
    """Recorded Understanding session closure payload."""

    status: Literal["closed", "escalated"] = "closed"
    session_id: str
    turn_count: int
    stopping_reason: StoppingReason
    closing_message: str
    rubric_progress: list[CriterionUnderstandingStatus]
    flags: list[str] = Field(default_factory=list)
    teacher_review_required: bool = True
    publication_status: Literal["draft_suggestion"] = "draft_suggestion"
    released: bool = False
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    lcms_receipt: Optional[DraftRecordReceipt] = None
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    trace: AgentTrace


class _UnderstandingTurnState(TypedDict, total=False):
    session_id: str
    request: UnderstandingTurnRequest
    request_id: str
    model_adapter: AgentModelAdapter
    knowledge_service: LessonKnowledgeService
    session_store: SessionMemoryStore
    executor: AuthorizedToolExecutor
    policy: RolePolicy
    session: SessionRecord
    assessment_context: AssessmentContextRecord
    lesson: LessonMaterialRecord
    bundle: LessonQuestionsRubricBundle
    criteria_progress: list[CriterionUnderstandingStatus]
    evidence_refs: list[EvidenceRef]
    lesson_context_text: str
    inspection: SecurityInspection
    prior_injection_count: int
    new_injection_count: int
    accumulated_flags: list[str]
    assistant_reply: str
    turn_feedback: str
    answer_leak_prevented: bool
    decision: Literal["follow_up", "close_session"]
    stopping_reason: Optional[StoppingReason]
    escalate: bool
    turn_index: int
    final_turn_count: int
    final_status: Literal["active", "closed", "escalated"]
    lcms_receipt: Optional[DraftRecordReceipt]
    observable_steps: list[ObservableStep]


def _append_step(
    state: _UnderstandingTurnState,
    *,
    step: str,
    action: str,
    outcome: Literal["ok", "denied", "skipped", "mocked"],
    detail: str,
    selected_tools: Optional[list[str]] = None,
    duration_ms: float = 0.0,
) -> list[ObservableStep]:
    steps = list(state.get("observable_steps", []))
    steps.append(
        ObservableStep(
            step=step,
            action=action,
            outcome=outcome,
            detail=detail,
            selected_tools=selected_tools or [],
            duration_ms=round(duration_ms, 2),
        )
    )
    return steps


def _criteria_from_bundle(bundle: LessonQuestionsRubricBundle) -> list[CriterionUnderstandingStatus]:
    raw_criteria = bundle.rubric.get("criteria") if isinstance(bundle.rubric, dict) else []
    out: list[CriterionUnderstandingStatus] = []
    if isinstance(raw_criteria, list):
        for idx, item in enumerate(raw_criteria, start=1):
            if not isinstance(item, dict):
                continue
            cid = str(item.get("criterion_id") or f"C{idx}").strip()
            name = str(item.get("name") or f"Criterion {idx}").strip()
            desc = str(item.get("description") or "").strip()
            out.append(
                CriterionUnderstandingStatus(
                    criterion_id=cid,
                    name=name,
                    description=desc,
                    required=bool(item.get("required", False)),
                    status="uncovered",
                )
            )
    if not out:
        out.append(
            CriterionUnderstandingStatus(
                criterion_id="C1",
                name="Core Lesson Concept",
                description="Explains the core concept using approved lesson material.",
                required=True,
                status="uncovered",
            )
        )
    return out


def _hydrate_criteria(
    rubric_progress_dict: dict[str, Any],
    bundle: LessonQuestionsRubricBundle,
) -> tuple[list[CriterionUnderstandingStatus], int, list[str]]:
    raw_list = rubric_progress_dict.get("criteria") if isinstance(rubric_progress_dict, dict) else None
    criteria: list[CriterionUnderstandingStatus] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, dict) and item.get("criterion_id"):
                status = str(item.get("status") or "uncovered").lower()
                if status not in _VALID_STATUSES:
                    status = "uncovered"
                criteria.append(
                    CriterionUnderstandingStatus(
                        criterion_id=str(item["criterion_id"]),
                        name=str(item.get("name") or item["criterion_id"]),
                        description=str(item.get("description") or ""),
                        required=bool(item.get("required", False)),
                        status=status,  # type: ignore[arg-type]
                        reason=str(item.get("reason") or ""),
                        evidence_quote=str(item.get("evidence_quote") or ""),
                    )
                )
    if not criteria:
        criteria = _criteria_from_bundle(bundle)

    sec = rubric_progress_dict.get("_security") if isinstance(rubric_progress_dict, dict) else {}
    inj_count = int(sec.get("injection_count", 0)) if isinstance(sec, dict) else 0
    raw_flags = sec.get("flags", []) if isinstance(sec, dict) else []
    flags = [str(f) for f in raw_flags if isinstance(f, str) and f]
    return criteria, inj_count, flags


def _merge_flags(existing: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*existing, *additions]:
        if item and item not in merged:
            merged.append(item)
    return merged


def _build_opening_prompt(
    *,
    lesson_title: str,
    criteria: list[CriterionUnderstandingStatus],
    topic_focus: Optional[str],
    language: str,
    adapter_mode: Literal["mock", "live"],
) -> str:
    first = criteria[0] if criteria else CriterionUnderstandingStatus(criterion_id="C1", name=lesson_title)
    focus = (topic_focus or first.name).strip()
    if language == "vi":
        text = (
            f"Chào bạn, chúng ta sẽ cùng trao đổi ở mức Hiểu cho bài '{lesson_title}'. "
            f"Trước hết, bạn hãy giải thích bằng lời của mình về '{focus}' "
            "và nêu một ví dụ minh họa hoặc lý do hoạt động nhé?"
        )
    else:
        text = (
            f"Welcome to this Understanding dialogue for '{lesson_title}'. "
            f"To begin, please explain '{focus}' in your own words "
            "along with a brief example of why it matters."
        )
    return f"[MOCK] {text}" if adapter_mode == "mock" else text


def _build_turn_prompts(
    *,
    req: UnderstandingTurnRequest,
    lesson: LessonMaterialRecord,
    criteria: list[CriterionUnderstandingStatus],
    rendered_history: str,
    lesson_context: str,
) -> tuple[str, str]:
    criteria_payload = [
        {
            "criterion_id": c.criterion_id,
            "name": c.name,
            "description": c.description,
            "status": c.status,
        }
        for c in criteria
    ]
    cfg = runtime_config.current()
    version = cfg.prompt_version(prompt_registry.TASK_H12_UNDERSTANDING_TURN)
    return prompt_registry.build(
        prompt_registry.TASK_H12_UNDERSTANDING_TURN,
        version,
        language=req.language,
        course_id=lesson.course_id or "(unspecified)",
        lesson_id=lesson.lesson_id,
        lesson_version=lesson.lesson_version,
        lesson_title=lesson.title,
        criteria_json=json.dumps(criteria_payload, ensure_ascii=False),
        rendered_history=rendered_history or "(first student turn)",
        lesson_context=lesson_context,
        student_message=req.message,
    )


def _parse_model_turn_output(
    raw: dict[str, Any],
    *,
    valid_ids: set[str],
) -> tuple[list[dict[str, str]], Literal["follow_up", "close_session"], str, str]:
    decision_raw = str(raw.get("decision") or "").strip().lower()
    if decision_raw not in _VALID_DECISIONS:
        raise AppError(
            ErrorCode.EVALUATION_INVALID_RESPONSE,
            "Model returned invalid decision for Understanding turn.",
        )

    reply_raw = raw.get("assistant_reply")
    if not isinstance(reply_raw, str) or not reply_raw.strip():
        raise AppError(
            ErrorCode.EVALUATION_INVALID_RESPONSE,
            "Model returned empty assistant_reply for Understanding turn.",
        )

    updates_raw = raw.get("criterion_updates")
    if updates_raw is None:
        updates_list: list[Any] = []
    elif isinstance(updates_raw, list):
        updates_list = updates_raw
    else:
        raise AppError(
            ErrorCode.EVALUATION_INVALID_RESPONSE,
            "Model returned invalid criterion_updates structure.",
        )

    clean_updates: list[dict[str, str]] = []
    for item in updates_list:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("criterion_id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if cid not in valid_ids or status not in _VALID_STATUSES:
            # Safely ignore hallucinated criterion IDs or invalid statuses.
            continue
        clean_updates.append(
            {
                "criterion_id": cid,
                "status": status,
                "reason": clamp_text(str(item.get("reason") or "Evaluated from dialogue turn.").strip(), 400),
                "evidence_quote": clamp_text(str(item.get("evidence_quote") or "").strip(), 300),
            }
        )

    feedback_raw = raw.get("turn_feedback")
    feedback = clamp_text(
        str(feedback_raw).strip() if isinstance(feedback_raw, str) and feedback_raw.strip() else "Turn evaluated.",
        500,
    )
    return clean_updates, decision_raw, reply_raw.strip(), feedback  # type: ignore[return-value]


def _apply_criterion_updates(
    criteria: list[CriterionUnderstandingStatus],
    updates: list[dict[str, str]],
) -> list[CriterionUnderstandingStatus]:
    rank = {"uncovered": 0, "partial": 1, "covered": 2}
    by_id = {u["criterion_id"]: u for u in updates}
    updated: list[CriterionUnderstandingStatus] = []
    for crit in criteria:
        u = by_id.get(crit.criterion_id)
        if u and rank.get(u["status"], 0) >= rank.get(crit.status, 0):
            updated.append(
                crit.model_copy(
                    update={
                        "status": u["status"],
                        "reason": u["reason"] or crit.reason,
                        "evidence_quote": u["evidence_quote"] or crit.evidence_quote,
                    }
                )
            )
        else:
            updated.append(crit)
    return updated


async def _ensure_student_safe_indexed(
    *,
    knowledge_service: LessonKnowledgeService,
    lesson: LessonMaterialRecord,
) -> None:
    existing = {v["knowledge_version"] for v in knowledge_service.list_versions(lesson.lesson_id)}
    if lesson.lesson_version not in existing:
        await knowledge_service.index_lesson_material(
            lesson_id=lesson.lesson_id,
            knowledge_version=lesson.lesson_version,
            title=lesson.title,
            student_safe_content=lesson.student_safe_content,
            approved=lesson.approved,
            teacher_only_notes=None,
        )


def _record_all_tool_audits(
    *,
    session_store: SessionMemoryStore,
    request_id: str,
    session_id: Optional[str],
    caller_scope: CallerScope,
    outcomes: list[ToolOutcome],
) -> None:
    for out in outcomes:
        session_store.record_tool_audit(
            request_id=request_id,
            session_id=session_id,
            agent_role="dialogue_assessment",
            caller_scope=caller_scope,
            outcome=out,
        )


def _build_trace(
    *,
    request_id: str,
    adapter: AgentModelAdapter,
    caller_scope: CallerScope,
    assessment_mode: AssessmentMode,
    policy: RolePolicy,
    outcomes: list[ToolOutcome],
    observable_steps: list[ObservableStep],
    extra_tools: Optional[list[ToolName]] = None,
) -> AgentTrace:
    from app.h12_agent.graph import get_framework_version

    selected_tools: list[ToolName] = []
    for out in outcomes:
        if out.outcome == "ok" and out.tool_name not in selected_tools:
            selected_tools.append(out.tool_name)
    for t in extra_tools or []:
        if t not in selected_tools:
            selected_tools.append(t)

    return AgentTrace(
        request_id=request_id,
        framework="langgraph",
        framework_version=get_framework_version(),
        adapter_mode=adapter.adapter_mode,
        is_mock=adapter.is_mock,
        provider=adapter.provider,
        model_name=adapter.model_name,
        agent_role="dialogue_assessment",
        caller_scope=caller_scope,
        assessment_mode=assessment_mode,
        visibility_scope=policy.visibility_scope,
        allowed_tools=list(policy.allowed_tools),
        selected_tools=selected_tools,
        observable_steps=observable_steps,
    )


async def _node_validate_mode_and_session(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    req = state["request"]
    session_id = state["session_id"]
    session_store = state["session_store"]

    session = session_store.get_session_for_owner(session_id, student_id=req.student_id)
    if session.status != "active":
        raise AppError(
            ErrorCode.SESSION_STATE_CONFLICT,
            f"Session '{session_id}' is already {session.status} and cannot accept new turns.",
        )
    if session.turn_count >= session.max_turns:
        raise AppError(
            ErrorCode.SESSION_STATE_CONFLICT,
            f"Session '{session_id}' reached max_turns ({session.max_turns}).",
        )

    institutional_ref = InstitutionalRef(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
        course_id=session.course_id,
        assignment_id=session.assignment_id,
        student_id=session.student_id,
        attempt_id=session.attempt_id,
        rubric_id=session.rubric_id,
    )
    policy = enforce_mode_and_scope(
        agent_role="dialogue_assessment",
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        institutional_ref=institutional_ref,
        format_hint=req.channel,
        channel=req.channel,
    )
    ctx = await state["executor"].fetch_assessment_context(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
        student_id=session.student_id,
        attempt_id=session.attempt_id,
    )

    return {
        "session": session,
        "policy": policy,
        "assessment_context": ctx,
        "observable_steps": _append_step(
            state,
            step="validate_mode_and_session",
            action=f"verify:{session_id}:{req.student_id}:{req.assessment_mode}",
            outcome="ok",
            detail=f"Session active ({session.turn_count}/{session.max_turns}) and mode allowed.",
            selected_tools=["fetch_assessment_context"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_fetch_safe_lesson_and_rubric(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    session = state["session"]
    executor = state["executor"]

    lesson = await executor.fetch_authorized_lesson(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
    )
    bundle = await executor.fetch_approved_questions_rubric(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
        rubric_id=session.rubric_id,
    )
    criteria_progress, inj_count, flags = _hydrate_criteria(session.rubric_progress, bundle)

    return {
        "lesson": lesson,
        "bundle": bundle,
        "criteria_progress": criteria_progress,
        "prior_injection_count": inj_count,
        "accumulated_flags": flags,
        "observable_steps": _append_step(
            state,
            step="fetch_safe_lesson_and_rubric",
            action="fetch_authorized_lesson+fetch_approved_questions_rubric",
            outcome="ok",
            detail=f"Loaded {len(criteria_progress)} Understanding criteria in student_safe visibility.",
            selected_tools=["fetch_authorized_lesson", "fetch_approved_questions_rubric"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_retrieve_student_safe_evidence(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    req = state["request"]
    lesson = state["lesson"]
    knowledge = state["knowledge_service"]
    policy = state["policy"]

    await _ensure_student_safe_indexed(knowledge_service=knowledge, lesson=lesson)
    query = f"{req.message[:350]} {lesson.title}".strip()
    retrieval = await knowledge.retrieve_lesson_evidence(
        lesson_id=lesson.lesson_id,
        knowledge_version=lesson.lesson_version,
        query=query,
        visibility_scope=policy.visibility_scope,
    )
    if retrieval.insufficient_context:
        retrieval = await knowledge.retrieve_lesson_evidence(
            lesson_id=lesson.lesson_id,
            knowledge_version=lesson.lesson_version,
            query=lesson.title,
            visibility_scope=policy.visibility_scope,
        )
    if retrieval.insufficient_context:
        raise AppError(
            ErrorCode.INSUFFICIENT_CONTEXT,
            "Insufficient approved student-safe lesson material to guide Understanding dialogue.",
        )

    return {
        "lesson_context_text": retrieval.context,
        "evidence_refs": retrieval.evidence_refs,
        "observable_steps": _append_step(
            state,
            step="retrieve_student_safe_evidence",
            action="retrieve_lesson_evidence:student_safe",
            outcome="ok",
            detail=f"Retrieved {len(retrieval.evidence_refs)} student_safe chunks.",
            selected_tools=["retrieve_lesson_evidence"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_inspect_security_and_evaluate_turn(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    req = state["request"]
    session = state["session"]
    lesson = state["lesson"]
    adapter = state["model_adapter"]
    criteria = state["criteria_progress"]
    session_store = state["session_store"]

    inspection = inspect_student_message(req.message)
    new_injection_count = state["prior_injection_count"] + (1 if inspection.injection_detected else 0)
    all_flags = _merge_flags(state["accumulated_flags"], inspection.flags)

    context_window = session_store.build_model_context(session.session_id, student_id=req.student_id)
    system_prompt, user_prompt = _build_turn_prompts(
        req=req,
        lesson=lesson,
        criteria=criteria,
        rendered_history=context_window.rendered_context,
        lesson_context=state["lesson_context_text"],
    )

    valid_ids = {c.criterion_id for c in criteria}
    next_open = next((c for c in criteria if c.status != "covered"), criteria[0])
    max_tokens = get_settings().grading_max_output_tokens

    parsed: Optional[tuple[list[dict[str, str]], Literal["follow_up", "close_session"], str, str]] = None
    last_err: Optional[AppError] = None
    total_ms = 0.0

    for _attempt in range(1, MAX_UNDERSTANDING_MODEL_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            raw, elapsed = await adapter.generate_structured(
                agent_role="dialogue_assessment",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                required_keys=["criterion_updates", "decision", "assistant_reply", "turn_feedback"],
                max_tokens=max_tokens,
            )
            total_ms += elapsed
            parsed = _parse_model_turn_output(raw, valid_ids=valid_ids)
            break
        except AppError as exc:
            total_ms += (time.perf_counter() - started) * 1000
            if exc.code in {ErrorCode.LLM_INVALID_RESPONSE, ErrorCode.EVALUATION_INVALID_RESPONSE}:
                last_err = exc
                continue
            if inspection.injection_detected or inspection.closure_requested:
                last_err = exc
                break
            raise

    if parsed is None:
        if inspection.injection_detected:
            refusal_fallback = build_refusal_reply(
                inspection=inspection,
                next_criterion_name=next_open.name,
                language=req.language,
            )
            parsed = ([], "follow_up", refusal_fallback, "Security guard blocked adversarial input.")
        elif inspection.closure_requested:
            close_fallback = (
                "Cảm ơn bạn đã tham gia trao đổi. Phiên đánh giá mức Hiểu kết thúc theo yêu cầu của bạn."
                if req.language == "vi"
                else "Thank you for participating. This Understanding session is now closed as requested."
            )
            parsed = ([], "close_session", close_fallback, "Session closed as requested by student.")
        else:
            raise AppError(
                ErrorCode.EVALUATION_INVALID_RESPONSE,
                last_err.message if last_err else "Model returned invalid Understanding turn JSON.",
            )

    updates, model_decision, raw_reply, turn_feedback = parsed

    if inspection.block_rubric_progress:
        updated_criteria = list(criteria)
        turn_feedback = (
            "[MOCK] Security boundary preserved; no rubric progress was updated."
            if adapter.is_mock
            else "Security guard blocked rubric advancement for adversarial input."
        )
    else:
        updated_criteria = _apply_criterion_updates(criteria, updates)

    forbidden_markers = list(_FORBIDDEN_LEAK_MARKERS)
    for q in state["bundle"].questions:
        if q.answer_key:
            forbidden_markers.append(q.answer_key)

    safe_reply, answer_leak_prevented = guard_assistant_reply(
        raw_reply,
        system_prompt=system_prompt,
        forbidden_markers=forbidden_markers,
        next_criterion_name=next_open.name,
        language=req.language,
    )

    if inspection.injection_detected and not answer_leak_prevented:
        refusal = build_refusal_reply(
            inspection=inspection,
            next_criterion_name=next_open.name,
            language=req.language,
        )
        safe_reply = clamp_text(
            f"[MOCK] {refusal}" if adapter.is_mock else refusal,
            MAX_DIALOGUE_REPLY_CHARS,
        )

    return {
        "inspection": inspection,
        "new_injection_count": new_injection_count,
        "accumulated_flags": all_flags,
        "criteria_progress": updated_criteria,
        "assistant_reply": safe_reply,
        "turn_feedback": turn_feedback,
        "answer_leak_prevented": answer_leak_prevented,
        "decision": model_decision,
        "observable_steps": _append_step(
            state,
            step="inspect_security_and_evaluate_turn",
            action=f"evaluate_turn:{adapter.adapter_mode}",
            outcome="mocked" if adapter.is_mock else "ok",
            detail=(
                f"injection={inspection.injection_detected} inj_count={new_injection_count} "
                f"leak_prevented={answer_leak_prevented}"
            ),
            duration_ms=total_ms,
        ),
    }


async def _node_decide_next_action(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    session = state["session"]
    criteria = state["criteria_progress"]
    inspection = state["inspection"]
    inj_count = state["new_injection_count"]
    flags = list(state["accumulated_flags"])
    req = state["request"]
    adapter = state["model_adapter"]
    reply = state["assistant_reply"]

    projected_turn_count = session.turn_count + 1
    all_covered = bool(criteria) and all(c.status == "covered" for c in criteria)

    decision: Literal["follow_up", "close_session"] = "follow_up"
    stopping_reason: Optional[StoppingReason] = None
    escalate = False

    if inj_count >= MAX_REPEATED_INJECTIONS:
        decision = "close_session"
        stopping_reason = "repeated_injection_escalation"
        escalate = True
        flags = _merge_flags(flags, ["REPEATED_INJECTION_ESCALATION"])
        esc_msg = (
            "Phiên đối thoại bị tạm dừng và chuyển giảng viên xem xét do phát hiện nhiều lượt tìm cách vượt rào kiểm soát."
            if req.language == "vi"
            else "This Understanding session has been paused and escalated for lecturer review due to repeated policy bypass attempts."
        )
        reply = f"[MOCK] {esc_msg}" if adapter.is_mock else esc_msg
    elif inspection.closure_requested:
        decision = "close_session"
        stopping_reason = "student_requested_closure"
    elif all_covered and not inspection.block_rubric_progress:
        decision = "close_session"
        stopping_reason = "criteria_covered"
    elif projected_turn_count >= session.max_turns:
        decision = "close_session"
        stopping_reason = "max_turns_reached"
        max_msg = (
            "Phiên đối thoại đã đạt số lượt trao đổi tối đa. Tiến độ tiêu chí hiện tại đã được ghi nhận để giảng viên xem xét."
            if req.language == "vi"
            else "We have reached the maximum turn limit for this session. Your current rubric progress has been recorded for lecturer review."
        )
        reply = f"[MOCK] {max_msg}" if adapter.is_mock else max_msg

    return {
        "decision": decision,
        "stopping_reason": stopping_reason,
        "escalate": escalate,
        "accumulated_flags": flags,
        "assistant_reply": reply,
        "final_turn_count": projected_turn_count,
        "observable_steps": _append_step(
            state,
            step="decide_next_action",
            action=f"route:{decision}",
            outcome="ok",
            detail=f"decision={decision} stopping_reason={stopping_reason} turns={projected_turn_count}/{session.max_turns}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


def _route_after_decision(state: _UnderstandingTurnState) -> str:
    return "close_and_record_outcome" if state["decision"] == "close_session" else "finalize_active_turn"


def _persist_turn_and_progress(state: _UnderstandingTurnState) -> int:
    req = state["request"]
    session = state["session"]
    session_store = state["session_store"]
    evidence_refs = state["evidence_refs"]

    session_store.append_turn(
        session.session_id,
        student_id=req.student_id,
        speaker="student",
        message_text=req.message,
        evidence_refs=evidence_refs,
    )
    assistant_turn = session_store.append_turn(
        session.session_id,
        student_id=req.student_id,
        speaker="agent",
        message_text=state["assistant_reply"],
        evidence_refs=evidence_refs,
        tool_outcomes=state["executor"].outcomes,
    )
    session_store.update_rubric_progress(
        session.session_id,
        student_id=req.student_id,
        rubric_progress={
            "criteria": [c.model_dump(mode="json") for c in state["criteria_progress"]],
            "_security": {
                "injection_count": state["new_injection_count"],
                "flags": state["accumulated_flags"],
            },
        },
    )
    return assistant_turn.turn_index


async def _node_close_and_record_outcome(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    req = state["request"]
    session = state["session"]
    session_store = state["session_store"]
    executor = state["executor"]
    adapter = state["model_adapter"]
    stopping_reason = state["stopping_reason"] or "criteria_covered"
    turn_index = _persist_turn_and_progress(state)

    institutional_ref = InstitutionalRef(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
        course_id=session.course_id,
        assignment_id=session.assignment_id,
        student_id=session.student_id,
        attempt_id=session.attempt_id,
        rubric_id=session.rubric_id,
    )
    covered_count = sum(1 for c in state["criteria_progress"] if c.status == "covered")
    summary_feedback = (
        f"Understanding dialogue concluded ({stopping_reason}): "
        f"{covered_count}/{len(state['criteria_progress'])} criteria covered in {state['final_turn_count']} student turns. "
        f"{state['turn_feedback']}"
    )
    receipt = await executor.record_dialogue_outcome(
        DialogueOutcomeRecordInput(
            institutional_ref=institutional_ref,
            session_id=session.session_id,
            stopping_reason=stopping_reason,
            turn_count=state["final_turn_count"],
            criterion_assessments=[
                {
                    "criterion_id": c.criterion_id,
                    "name": c.name,
                    "status": c.status,
                    "reason": c.reason,
                    "evidence_quote": c.evidence_quote,
                }
                for c in state["criteria_progress"]
            ],
            summary_feedback=clamp_text(summary_feedback, 3800),
            evidence_refs=state["evidence_refs"],
            flags=state["accumulated_flags"],
            adapter_mode=adapter.adapter_mode,
            idempotency_key=req.idempotency_key or f"dialogue-close:{session.session_id}:turn-{state['final_turn_count']}",
        )
    )
    closed_session = session_store.close_session(
        session.session_id,
        student_id=req.student_id,
        stopping_reason=stopping_reason,
        escalate=bool(state.get("escalate", False)),
    )

    return {
        "turn_index": turn_index,
        "final_status": closed_session.status,
        "lcms_receipt": receipt,
        "observable_steps": _append_step(
            state,
            step="close_and_record_outcome",
            action=f"record_dialogue_outcome:{stopping_reason}",
            outcome="ok",
            detail=f"status={closed_session.status} record_id={receipt.record_id}",
            selected_tools=["record_dialogue_outcome"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_finalize_active_turn(state: _UnderstandingTurnState) -> _UnderstandingTurnState:
    t0 = time.perf_counter()
    turn_index = _persist_turn_and_progress(state)
    return {
        "turn_index": turn_index,
        "final_status": "active",
        "lcms_receipt": None,
        "observable_steps": _append_step(
            state,
            step="finalize_active_turn",
            action="persist_turn_and_rubric_progress",
            outcome="ok",
            detail=f"turn={state['final_turn_count']}/{state['session'].max_turns} remains active",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


def _build_understanding_turn_graph():
    builder = StateGraph(_UnderstandingTurnState)
    builder.add_node("validate_mode_and_session", _node_validate_mode_and_session)
    builder.add_node("fetch_safe_lesson_and_rubric", _node_fetch_safe_lesson_and_rubric)
    builder.add_node("retrieve_student_safe_evidence", _node_retrieve_student_safe_evidence)
    builder.add_node("inspect_security_and_evaluate_turn", _node_inspect_security_and_evaluate_turn)
    builder.add_node("decide_next_action", _node_decide_next_action)
    builder.add_node("close_and_record_outcome", _node_close_and_record_outcome)
    builder.add_node("finalize_active_turn", _node_finalize_active_turn)

    builder.add_edge(START, "validate_mode_and_session")
    builder.add_edge("validate_mode_and_session", "fetch_safe_lesson_and_rubric")
    builder.add_edge("fetch_safe_lesson_and_rubric", "retrieve_student_safe_evidence")
    builder.add_edge("retrieve_student_safe_evidence", "inspect_security_and_evaluate_turn")
    builder.add_edge("inspect_security_and_evaluate_turn", "decide_next_action")
    builder.add_conditional_edges(
        "decide_next_action",
        _route_after_decision,
        {
            "close_and_record_outcome": "close_and_record_outcome",
            "finalize_active_turn": "finalize_active_turn",
        },
    )
    builder.add_edge("close_and_record_outcome", END)
    builder.add_edge("finalize_active_turn", END)
    return builder.compile()


_UNDERSTANDING_TURN_GRAPH = _build_understanding_turn_graph()


async def start_understanding_session(
    req: StartUnderstandingSessionRequest,
    *,
    request_id: str,
    model_adapter: AgentModelAdapter,
    lcms_client: LCMSToolClient,
    knowledge_service: LessonKnowledgeService,
    session_store: SessionMemoryStore,
) -> StartUnderstandingSessionResponse:
    """Validate Understanding mode, bootstrap student_safe lesson & rubric, and create isolated session."""
    started = time.perf_counter()
    steps: list[ObservableStep] = []

    t0 = time.perf_counter()
    policy = enforce_mode_and_scope(
        agent_role="dialogue_assessment",
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        institutional_ref=req.institutional_ref,
        format_hint=req.channel,
        channel=req.channel,
    )
    executor = AuthorizedToolExecutor(
        lcms_client=lcms_client,
        policy=policy,
        assessment_mode=req.assessment_mode,
        caller_scope=req.caller_scope,
        request_id=request_id,
    )
    session_id_for_audit: Optional[str] = None
    try:
        ctx = await executor.fetch_assessment_context(
            lesson_id=req.institutional_ref.lesson_id,
            lesson_version=req.institutional_ref.lesson_version,
            student_id=req.institutional_ref.student_id,
            attempt_id=req.institutional_ref.attempt_id,
        )
        steps.append(
            ObservableStep(
                step="validate_mode_and_context",
                action="enforce_mode_and_scope+fetch_assessment_context",
                outcome="ok",
                detail=f"Understanding dialogue permitted for student={req.institutional_ref.student_id}.",
                selected_tools=["fetch_assessment_context"],
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        )

        t1 = time.perf_counter()
        lesson = await executor.fetch_authorized_lesson(
            lesson_id=req.institutional_ref.lesson_id,
            lesson_version=req.institutional_ref.lesson_version,
        )
        bundle = await executor.fetch_approved_questions_rubric(
            lesson_id=req.institutional_ref.lesson_id,
            lesson_version=req.institutional_ref.lesson_version,
            rubric_id=req.institutional_ref.rubric_id,
        )
        steps.append(
            ObservableStep(
                step="fetch_safe_lesson_and_rubric",
                action="fetch_authorized_lesson+fetch_approved_questions_rubric",
                outcome="ok",
                detail=f"Loaded {lesson.lesson_id}@{lesson.lesson_version} and rubric {bundle.rubric_id}.",
                selected_tools=["fetch_authorized_lesson", "fetch_approved_questions_rubric"],
                duration_ms=round((time.perf_counter() - t1) * 1000, 2),
            )
        )

        t2 = time.perf_counter()
        await _ensure_student_safe_indexed(knowledge_service=knowledge_service, lesson=lesson)
        retrieval = await knowledge_service.retrieve_lesson_evidence(
            lesson_id=lesson.lesson_id,
            knowledge_version=lesson.lesson_version,
            query=(req.topic_focus or lesson.title).strip(),
            visibility_scope=policy.visibility_scope,
        )
        if retrieval.insufficient_context:
            raise AppError(
                ErrorCode.INSUFFICIENT_CONTEXT,
                "Insufficient approved student-safe lesson material to start Understanding session.",
            )
        steps.append(
            ObservableStep(
                step="retrieve_student_safe_evidence",
                action="retrieve_lesson_evidence:student_safe",
                outcome="ok",
                detail=f"Retrieved {len(retrieval.evidence_refs)} approved chunks.",
                selected_tools=["retrieve_lesson_evidence"],
                duration_ms=round((time.perf_counter() - t2) * 1000, 2),
            )
        )

        t3 = time.perf_counter()
        initial_criteria = _criteria_from_bundle(bundle)
        session, created = session_store.create_or_get_session(
            student_id=req.institutional_ref.student_id or "",
            lesson_id=lesson.lesson_id,
            lesson_version=lesson.lesson_version,
            attempt_id=req.institutional_ref.attempt_id or "",
            assessment_mode=req.assessment_mode,
            course_id=req.institutional_ref.course_id or ctx.course_id,
            assignment_id=req.institutional_ref.assignment_id or ctx.assignment_id,
            rubric_id=bundle.rubric_id,
            max_turns=req.max_turns or ctx.max_turns,
        )
        session_id_for_audit = session.session_id

        if created:
            opening_prompt = _build_opening_prompt(
                lesson_title=lesson.title,
                criteria=initial_criteria,
                topic_focus=req.topic_focus,
                language=req.language,
                adapter_mode=model_adapter.adapter_mode,
            )
            session_store.update_rubric_progress(
                session.session_id,
                student_id=session.student_id,
                rubric_progress={
                    "criteria": [c.model_dump(mode="json") for c in initial_criteria],
                    "_security": {"injection_count": 0, "flags": []},
                },
            )
            session_store.append_turn(
                session.session_id,
                student_id=session.student_id,
                speaker="agent",
                message_text=opening_prompt,
                evidence_refs=retrieval.evidence_refs,
                tool_outcomes=executor.outcomes,
            )
            rubric_progress = initial_criteria
        else:
            rubric_progress, _, _ = _hydrate_criteria(session.rubric_progress, bundle)
            turns = session_store.list_turns(session.session_id, student_id=session.student_id)
            assistant_msgs = [t.message_text for t in turns if t.speaker == "agent"]
            opening_prompt = (
                assistant_msgs[-1]
                if assistant_msgs
                else _build_opening_prompt(
                    lesson_title=lesson.title,
                    criteria=rubric_progress,
                    topic_focus=req.topic_focus,
                    language=req.language,
                    adapter_mode=model_adapter.adapter_mode,
                )
            )

        steps.append(
            ObservableStep(
                step="initialize_isolated_session",
                action="create_or_get_session",
                outcome="ok",
                detail=f"session_id={session.session_id} created={created} max_turns={session.max_turns}",
                duration_ms=round((time.perf_counter() - t3) * 1000, 2),
            )
        )
    finally:
        _record_all_tool_audits(
            session_store=session_store,
            request_id=request_id,
            session_id=session_id_for_audit,
            caller_scope=req.caller_scope,
            outcomes=executor.outcomes,
        )

    trace = _build_trace(
        request_id=request_id,
        adapter=model_adapter,
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        policy=policy,
        outcomes=executor.outcomes,
        observable_steps=steps,
        extra_tools=["retrieve_lesson_evidence"],
    )
    logger.info(
        "H12 understanding start | request_id=%s session_id=%s created=%s student=%s latency_ms=%.1f",
        request_id,
        session.session_id,
        created,
        session.student_id,
        (time.perf_counter() - started) * 1000,
    )
    return StartUnderstandingSessionResponse(
        status=session.status,
        session_id=session.session_id,
        created=created,
        institutional_ref=InstitutionalRef(
            lesson_id=session.lesson_id,
            lesson_version=session.lesson_version,
            course_id=session.course_id,
            assignment_id=session.assignment_id,
            student_id=session.student_id,
            attempt_id=session.attempt_id,
            rubric_id=bundle.rubric_id,
        ),
        lesson_title=lesson.title,
        rubric_id=bundle.rubric_id,
        max_turns=session.max_turns,
        turn_count=session.turn_count,
        opening_prompt=opening_prompt,
        rubric_progress=rubric_progress,
        evidence_refs=retrieval.evidence_refs,
        tool_outcomes=executor.outcomes,
        trace=trace,
    )


async def run_understanding_turn(
    session_id: str,
    req: UnderstandingTurnRequest,
    *,
    request_id: str,
    model_adapter: AgentModelAdapter,
    lcms_client: LCMSToolClient,
    knowledge_service: LessonKnowledgeService,
    session_store: SessionMemoryStore,
) -> UnderstandingTurnResponse:
    """Execute one turn of the conditional LangGraph Understanding adaptive dialogue workflow."""
    from app.h12_agent.policy import ROLE_POLICIES

    started = time.perf_counter()
    policy = ROLE_POLICIES["dialogue_assessment"]
    executor = AuthorizedToolExecutor(
        lcms_client=lcms_client,
        policy=policy,
        assessment_mode=req.assessment_mode,
        caller_scope=req.caller_scope,
        request_id=request_id,
    )
    try:
        final_state = await _UNDERSTANDING_TURN_GRAPH.ainvoke(
            {
                "session_id": session_id,
                "request": req,
                "request_id": request_id,
                "model_adapter": model_adapter,
                "knowledge_service": knowledge_service,
                "session_store": session_store,
                "executor": executor,
                "observable_steps": [],
            }
        )
    finally:
        _record_all_tool_audits(
            session_store=session_store,
            request_id=request_id,
            session_id=session_id,
            caller_scope=req.caller_scope,
            outcomes=executor.outcomes,
        )

    trace = _build_trace(
        request_id=request_id,
        adapter=model_adapter,
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        policy=final_state["policy"],
        outcomes=executor.outcomes,
        observable_steps=final_state["observable_steps"],
        extra_tools=["retrieve_lesson_evidence"],
    )
    logger.info(
        "H12 understanding turn | request_id=%s session_id=%s turn=%d/%d decision=%s reason=%s latency_ms=%.1f",
        request_id,
        session_id,
        final_state["final_turn_count"],
        final_state["session"].max_turns,
        final_state["decision"],
        final_state["stopping_reason"],
        (time.perf_counter() - started) * 1000,
    )
    return UnderstandingTurnResponse(
        status=final_state["final_status"],
        session_id=session_id,
        turn_index=final_state["turn_index"],
        turn_count=final_state["final_turn_count"],
        max_turns=final_state["session"].max_turns,
        decision=final_state["decision"],
        stopping_reason=final_state["stopping_reason"],
        assistant_reply=final_state["assistant_reply"],
        turn_feedback=final_state["turn_feedback"],
        rubric_progress=final_state["criteria_progress"],
        flags=final_state["accumulated_flags"],
        answer_leak_prevented=final_state["answer_leak_prevented"],
        evidence_refs=final_state["evidence_refs"],
        lcms_receipt=final_state.get("lcms_receipt"),
        tool_outcomes=executor.outcomes,
        trace=trace,
    )


async def close_understanding_session(
    session_id: str,
    req: CloseUnderstandingSessionRequest,
    *,
    request_id: str,
    model_adapter: AgentModelAdapter,
    lcms_client: LCMSToolClient,
    session_store: SessionMemoryStore,
) -> CloseUnderstandingSessionResponse:
    """Explicitly conclude an active Understanding session and hand off the recorded outcome to LCMS."""
    started = time.perf_counter()
    steps: list[ObservableStep] = []

    t0 = time.perf_counter()
    session = session_store.get_session_for_owner(session_id, student_id=req.student_id)
    if session.status != "active":
        raise AppError(
            ErrorCode.SESSION_STATE_CONFLICT,
            f"Session '{session_id}' is already {session.status} and cannot be closed again.",
        )

    institutional_ref = InstitutionalRef(
        lesson_id=session.lesson_id,
        lesson_version=session.lesson_version,
        course_id=session.course_id,
        assignment_id=session.assignment_id,
        student_id=session.student_id,
        attempt_id=session.attempt_id,
        rubric_id=session.rubric_id,
    )
    policy = enforce_mode_and_scope(
        agent_role="dialogue_assessment",
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        institutional_ref=institutional_ref,
        format_hint="text_chat",
        channel="text_chat",
    )
    executor = AuthorizedToolExecutor(
        lcms_client=lcms_client,
        policy=policy,
        assessment_mode=req.assessment_mode,
        caller_scope=req.caller_scope,
        request_id=request_id,
    )
    try:
        await executor.fetch_assessment_context(
            lesson_id=session.lesson_id,
            lesson_version=session.lesson_version,
            student_id=session.student_id,
            attempt_id=session.attempt_id,
        )
        bundle = await executor.fetch_approved_questions_rubric(
            lesson_id=session.lesson_id,
            lesson_version=session.lesson_version,
            rubric_id=session.rubric_id,
        )
        steps.append(
            ObservableStep(
                step="validate_and_load_session",
                action="get_session_for_owner+fetch_assessment_context",
                outcome="ok",
                detail=f"Closing active session {session_id} after {session.turn_count} turns.",
                selected_tools=["fetch_assessment_context", "fetch_approved_questions_rubric"],
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        )

        criteria_progress, _, flags = _hydrate_criteria(session.rubric_progress, bundle)
        covered_count = sum(1 for c in criteria_progress if c.status == "covered")
        reason_note = f" Reason: {req.reason.strip()}." if req.reason else ""
        summary_feedback = (
            f"Student closed Understanding session manually after {session.turn_count} turns "
            f"({covered_count}/{len(criteria_progress)} criteria covered).{reason_note}"
        )

        t1 = time.perf_counter()
        receipt = await executor.record_dialogue_outcome(
            DialogueOutcomeRecordInput(
                institutional_ref=institutional_ref,
                session_id=session.session_id,
                stopping_reason="student_requested_closure",
                turn_count=session.turn_count,
                criterion_assessments=[
                    {
                        "criterion_id": c.criterion_id,
                        "name": c.name,
                        "status": c.status,
                        "reason": c.reason,
                        "evidence_quote": c.evidence_quote,
                    }
                    for c in criteria_progress
                ],
                summary_feedback=clamp_text(summary_feedback, 3800),
                evidence_refs=session.evidence_refs,
                flags=flags,
                adapter_mode=model_adapter.adapter_mode,
                idempotency_key=f"dialogue-explicit-close:{session.session_id}",
            )
        )
        closed_session = session_store.close_session(
            session.session_id,
            student_id=req.student_id,
            stopping_reason="student_requested_closure",
            escalate=False,
        )
        steps.append(
            ObservableStep(
                step="close_and_record_outcome",
                action="record_dialogue_outcome+close_session",
                outcome="ok",
                detail=f"record_id={receipt.record_id}",
                selected_tools=["record_dialogue_outcome"],
                duration_ms=round((time.perf_counter() - t1) * 1000, 2),
            )
        )
    finally:
        _record_all_tool_audits(
            session_store=session_store,
            request_id=request_id,
            session_id=session_id,
            caller_scope=req.caller_scope,
            outcomes=executor.outcomes,
        )

    closing_msg = (
        "Phiên đối thoại đánh giá mức Hiểu đã được kết thúc theo yêu cầu của bạn và ghi nhận cho giảng viên xem xét."
        if req.language == "vi"
        else "This Understanding dialogue session has been closed at your request and recorded for lecturer review."
    )
    trace = _build_trace(
        request_id=request_id,
        adapter=model_adapter,
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        policy=policy,
        outcomes=executor.outcomes,
        observable_steps=steps,
    )
    logger.info(
        "H12 understanding close | request_id=%s session_id=%s turns=%d latency_ms=%.1f",
        request_id,
        session_id,
        closed_session.turn_count,
        (time.perf_counter() - started) * 1000,
    )
    return CloseUnderstandingSessionResponse(
        status=closed_session.status,  # type: ignore[arg-type]
        session_id=closed_session.session_id,
        turn_count=closed_session.turn_count,
        stopping_reason="student_requested_closure",
        closing_message=closing_msg,
        rubric_progress=criteria_progress,
        flags=flags,
        evidence_refs=closed_session.evidence_refs,
        lcms_receipt=receipt,
        tool_outcomes=executor.outcomes,
        trace=trace,
    )
