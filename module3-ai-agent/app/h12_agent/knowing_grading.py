"""LangGraph workflow 2: Knowing fixed essay rubric grading support & teacher handoff (Milestone 3)."""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core import prompt_registry, runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.grading.flags import detect_injection, is_empty_or_copy, split_flags
from app.grading.grader import CriterionJudgement, GradingJudgement, parse_judgement, rubric_prompt_json
from app.grading.policy import met_percentage, unmet_required
from app.grading.rubric_validator import validate_rubric
from app.h12_agent.knowledge import LessonKnowledgeService
from app.h12_agent.lcms_tools import (
    DraftRecordReceipt,
    GradingSuggestionRecordInput,
    LCMSToolClient,
    LessonMaterialRecord,
    LessonQuestionRecord,
    LessonQuestionsRubricBundle,
    ToolOutcome,
)
from app.h12_agent.model_adapter import AgentModelAdapter
from app.h12_agent.policy import RolePolicy, enforce_mode_and_scope
from app.h12_agent.session_store import SessionMemoryStore
from app.h12_agent.tool_executor import AuthorizedToolExecutor
from app.h12_agent.types import (
    AgentTrace,
    AssessmentMode,
    CallerScope,
    EvidenceRef,
    InstitutionalRef,
    ObservableStep,
    ToolName,
)
from app.schemas.rubric import Rubric

logger = logging.getLogger("app.ai_trace")

MAX_GRADING_ATTEMPTS = 2
_EPS = 1e-9
_EMPTY_REASON = "Không có nội dung bài làm tự luận để đánh giá."
_EMPTY_FEEDBACK = "Bài làm trống hoặc chỉ lặp lại đề bài."


class KnowingGradingWorkflowRequest(BaseModel):
    """Request to generate criterion-linked scoring suggestions for a fixed Knowing essay answer."""

    caller_scope: CallerScope
    assessment_mode: AssessmentMode = "home_knowing"
    institutional_ref: InstitutionalRef
    question_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    student_answer: str = Field(..., max_length=60_000)
    question_format: str = Field("essay", max_length=32)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")
    record_suggestion: bool = True
    idempotency_key: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")


class CriterionSuggestion(BaseModel):
    """Per-criterion provisional scoring suggestion derived deterministically from rubric points."""

    criterion_id: str
    name: str
    required: bool
    level: str
    met: bool
    suggested_points: float
    max_points: float
    reason: str
    evidence: str


class KnowingGradingWorkflowResponse(BaseModel):
    """Provisional Knowing essay grading suggestion handed off to the lecturer (never auto-published)."""

    status: Literal["pending_teacher_review", "needs_teacher_escalation"]
    publication_status: Literal["draft_suggestion"] = "draft_suggestion"
    released: bool = False
    teacher_review_required: bool = True
    teacher_handoff: bool = True
    idempotent_replay: bool = False
    institutional_ref: InstitutionalRef
    question_id: str
    question_text: str
    rubric_id: str
    suggested_total: float
    max_score: float
    met_percentage: float
    unmet_required: list[str] = Field(default_factory=list)
    criteria: list[CriterionSuggestion]
    feedback: str
    flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    lcms_receipt: Optional[DraftRecordReceipt] = None
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    trace: AgentTrace


class _KnowingGraphState(TypedDict, total=False):
    request: KnowingGradingWorkflowRequest
    request_id: str
    model_adapter: AgentModelAdapter
    knowledge_service: LessonKnowledgeService
    executor: AuthorizedToolExecutor
    policy: RolePolicy
    lesson: LessonMaterialRecord
    bundle: LessonQuestionsRubricBundle
    rubric: Rubric
    question: LessonQuestionRecord
    context_text: str
    evidence_refs: list[EvidenceRef]
    criteria: list[CriterionSuggestion]
    suggested_total: float
    met_pct: float
    unmet_req: list[str]
    feedback: str
    review_flags: list[str]
    warning_flags: list[str]
    status: Literal["pending_teacher_review", "needs_teacher_escalation"]
    lcms_receipt: Optional[DraftRecordReceipt]
    observable_steps: list[ObservableStep]


def _append_step(
    state: _KnowingGraphState,
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


async def _node_validate_mode_and_scope(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    req = state["request"]
    policy = enforce_mode_and_scope(
        agent_role="grading_support",
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        institutional_ref=req.institutional_ref,
        format_hint=req.question_format,
        channel="text_chat",
    )
    if not req.institutional_ref.student_id or not req.institutional_ref.attempt_id:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "student_id and attempt_id are required for Knowing essay grading support.",
            status_code=422,
        )
    if len(req.student_answer) > get_settings().max_submission_chars:
        raise AppError(
            ErrorCode.INPUT_TOO_LONG,
            f"student_answer must be at most {get_settings().max_submission_chars} characters.",
        )
    return {
        "policy": policy,
        "observable_steps": _append_step(
            state,
            step="validate_mode_and_scope",
            action=f"verify:{req.caller_scope}:{req.assessment_mode}",
            outcome="ok",
            detail="Essay grading support permitted with mandatory teacher handoff.",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_verify_assessment_context(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    ref = state["request"].institutional_ref
    ctx = await state["executor"].fetch_assessment_context(
        lesson_id=ref.lesson_id,
        lesson_version=ref.lesson_version,
        student_id=ref.student_id,
        attempt_id=ref.attempt_id,
    )
    return {
        "observable_steps": _append_step(
            state,
            step="verify_assessment_context",
            action="fetch_assessment_context",
            outcome="ok",
            detail=f"LCMS context permitted_mode={ctx.permitted_mode} classroom_exam_active=false",
            selected_tools=["fetch_assessment_context"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_fetch_lesson_and_rubric(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    req = state["request"]
    ref = req.institutional_ref
    executor = state["executor"]

    lesson = await executor.fetch_authorized_lesson(
        lesson_id=ref.lesson_id,
        lesson_version=ref.lesson_version,
    )
    bundle = await executor.fetch_approved_questions_rubric(
        lesson_id=ref.lesson_id,
        lesson_version=ref.lesson_version,
        rubric_id=ref.rubric_id,
    )

    rubric = Rubric.model_validate(bundle.rubric)
    if rubric.status != "approved":
        raise AppError(ErrorCode.RUBRIC_NOT_APPROVED, "Only an approved rubric can be used for grading support.")
    violations = validate_rubric(rubric)
    if violations:
        raise AppError(
            ErrorCode.RUBRIC_INVALID,
            "The rubric violates the rubric rules: " + "; ".join(f"[{v.rule}] {v.message}" for v in violations[:8]),
        )

    knowing_questions = [
        q for q in bundle.questions
        if q.approved and q.assessment_level == "knowing" and q.format == "essay"
    ]
    if req.question_id:
        question = next((q for q in knowing_questions if q.question_id == req.question_id), None)
    else:
        question = knowing_questions[0] if knowing_questions else None
    if question is None:
        raise AppError(
            ErrorCode.LESSON_NOT_FOUND,
            f"No approved fixed essay Knowing question found for lesson '{ref.lesson_id}'.",
        )

    return {
        "lesson": lesson,
        "bundle": bundle,
        "rubric": rubric,
        "question": question,
        "observable_steps": _append_step(
            state,
            step="fetch_lesson_and_rubric",
            action="fetch_lesson_and_approved_rubric",
            outcome="ok",
            detail=f"question={question.question_id} rubric={bundle.rubric_id}",
            selected_tools=["fetch_authorized_lesson", "fetch_approved_questions_rubric"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_retrieve_lesson_evidence(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    lesson = state["lesson"]
    question = state["question"]
    knowledge = state["knowledge_service"]
    policy = state["policy"]

    existing = {v["knowledge_version"] for v in knowledge.list_versions(lesson.lesson_id)}
    if lesson.lesson_version not in existing:
        await knowledge.index_lesson_material(
            lesson_id=lesson.lesson_id,
            knowledge_version=lesson.lesson_version,
            title=lesson.title,
            student_safe_content=lesson.student_safe_content,
            approved=lesson.approved,
            teacher_only_notes=lesson.teacher_only_notes,
        )

    query = f"{question.question_text}\n{question.answer_key or ''}".strip()[:1500]
    retrieval = await knowledge.retrieve_lesson_evidence(
        lesson_id=lesson.lesson_id,
        knowledge_version=lesson.lesson_version,
        query=query,
        visibility_scope=policy.visibility_scope,
    )

    return {
        "context_text": retrieval.context,
        "evidence_refs": retrieval.evidence_refs,
        "observable_steps": _append_step(
            state,
            step="retrieve_lesson_evidence",
            action="retrieve_lesson_evidence",
            outcome="skipped" if retrieval.insufficient_context else "ok",
            detail=(
                "insufficient_context"
                if retrieval.insufficient_context
                else f"Retrieved {len(retrieval.evidence_refs)} chunks."
            ),
            selected_tools=["retrieve_lesson_evidence"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_evaluate_essay_against_rubric(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    req = state["request"]
    rubric = state["rubric"]
    question = state["question"]
    adapter = state["model_adapter"]
    answer = req.student_answer.strip()

    raw_flags: list[str] = []
    if not rubric.grounded:
        raw_flags.append("ungrounded_rubric")

    if not answer or is_empty_or_copy(answer, question.question_text):
        judgement = GradingJudgement(
            criteria=[CriterionJudgement(c, c.levels[-1], _EMPTY_REASON, "") for c in rubric.criteria],
            feedback=_EMPTY_FEEDBACK,
            off_topic=True,
            prompt_version="none:empty-or-copy",
        )
        outcome: Literal["ok", "denied", "skipped", "mocked"] = "skipped"
        llm_ms = (time.perf_counter() - t0) * 1000
    else:
        version = runtime_config.current().prompt_version(prompt_registry.TASK_RUBRIC_GRADING)
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_RUBRIC_GRADING,
            version,
            question=question.question_text,
            rubric_json=rubric_prompt_json(rubric),
            answer=answer,
            expected_answer=question.answer_key or "",
            context=state.get("context_text", ""),
            submission_kind="typed",
            language=req.language,
            test_results="",
        )
        last_err: Optional[AppError] = None
        judgement_opt: Optional[GradingJudgement] = None
        llm_ms = 0.0
        max_tokens = get_settings().grading_max_output_tokens

        for _attempt in range(1, MAX_GRADING_ATTEMPTS + 1):
            raw, elapsed = await adapter.generate_structured(
                agent_role="grading_support",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                required_keys=["criteria", "feedback"],
                max_tokens=max_tokens,
            )
            llm_ms += elapsed
            try:
                judgement_opt = parse_judgement(raw, rubric)
                break
            except AppError as exc:
                last_err = exc

        if judgement_opt is None:
            assert last_err is not None
            raise last_err
        judgement = judgement_opt
        outcome = "mocked" if adapter.is_mock else "ok"

    if judgement.injection_suspected or detect_injection(answer):
        raw_flags.append("injection_suspected")
    if judgement.off_topic:
        raw_flags.append("empty_or_off_topic")

    ratio = get_settings().grading_criterion_met_ratio
    met_map = {
        j.criterion.criterion_id: (j.level.points + _EPS >= ratio * j.criterion.max_points)
        for j in judgement.criteria
    }
    criteria_out = [
        CriterionSuggestion(
            criterion_id=j.criterion.criterion_id,
            name=j.criterion.name,
            required=j.criterion.required,
            level=j.level.label,
            met=met_map[j.criterion.criterion_id],
            suggested_points=j.level.points,
            max_points=j.criterion.max_points,
            reason=j.reason,
            evidence=j.evidence,
        )
        for j in judgement.criteria
    ]
    suggested_total = round(sum(c.suggested_points for c in criteria_out), 2)
    met_pct = met_percentage(met_map, rubric)
    unmet_req = unmet_required(met_map, rubric)
    review_flags, warning_flags = split_flags(raw_flags)
    status: Literal["pending_teacher_review", "needs_teacher_escalation"] = (
        "needs_teacher_escalation" if review_flags else "pending_teacher_review"
    )

    return {
        "criteria": criteria_out,
        "suggested_total": suggested_total,
        "met_pct": met_pct,
        "unmet_req": unmet_req,
        "feedback": judgement.feedback,
        "review_flags": review_flags,
        "warning_flags": warning_flags,
        "status": status,
        "observable_steps": _append_step(
            state,
            step="evaluate_essay_against_rubric",
            action=f"rubric_suggest:{adapter.adapter_mode}",
            outcome=outcome,
            detail=f"suggested_total={suggested_total}/{rubric.max_score} status={status}",
            duration_ms=llm_ms,
        ),
    }


async def _node_record_grading_suggestion(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    req = state["request"]
    if not req.record_suggestion:
        return {
            "lcms_receipt": None,
            "observable_steps": _append_step(
                state,
                step="record_grading_suggestion",
                action="skip_record_suggestion",
                outcome="skipped",
                detail="record_suggestion=false",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ),
        }

    idem_key = req.idempotency_key or f"grade-{state['request_id']}"
    receipt = await state["executor"].record_grading_suggestion(
        GradingSuggestionRecordInput(
            institutional_ref=req.institutional_ref,
            suggested_total=state["suggested_total"],
            max_score=state["rubric"].max_score,
            criteria_suggestions=[c.model_dump(mode="json") for c in state["criteria"]],
            feedback=state["feedback"],
            flags=state["review_flags"],
            evidence_refs=state["evidence_refs"],
            adapter_mode=state["model_adapter"].adapter_mode,
            idempotency_key=idem_key,
        )
    )
    return {
        "lcms_receipt": receipt,
        "observable_steps": _append_step(
            state,
            step="record_grading_suggestion",
            action="record_grading_suggestion",
            outcome="ok",
            detail=f"record_id={receipt.record_id} released=false",
            selected_tools=["record_grading_suggestion"],
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


async def _node_finalize_handoff_boundary(state: _KnowingGraphState) -> _KnowingGraphState:
    t0 = time.perf_counter()
    return {
        "observable_steps": _append_step(
            state,
            step="finalize_handoff_boundary",
            action="enforce_teacher_handoff",
            outcome="ok",
            detail="publication_status=draft_suggestion released=false teacher_handoff=true",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ),
    }


def _build_knowing_grading_graph():
    builder = StateGraph(_KnowingGraphState)
    builder.add_node("validate_mode_and_scope", _node_validate_mode_and_scope)
    builder.add_node("verify_assessment_context", _node_verify_assessment_context)
    builder.add_node("fetch_lesson_and_rubric", _node_fetch_lesson_and_rubric)
    builder.add_node("retrieve_lesson_evidence", _node_retrieve_lesson_evidence)
    builder.add_node("evaluate_essay_against_rubric", _node_evaluate_essay_against_rubric)
    builder.add_node("record_grading_suggestion", _node_record_grading_suggestion)
    builder.add_node("finalize_handoff_boundary", _node_finalize_handoff_boundary)

    builder.add_edge(START, "validate_mode_and_scope")
    builder.add_edge("validate_mode_and_scope", "verify_assessment_context")
    builder.add_edge("verify_assessment_context", "fetch_lesson_and_rubric")
    builder.add_edge("fetch_lesson_and_rubric", "retrieve_lesson_evidence")
    builder.add_edge("retrieve_lesson_evidence", "evaluate_essay_against_rubric")
    builder.add_edge("evaluate_essay_against_rubric", "record_grading_suggestion")
    builder.add_edge("record_grading_suggestion", "finalize_handoff_boundary")
    builder.add_edge("finalize_handoff_boundary", END)
    return builder.compile()


_KNOWING_GRADING_GRAPH = _build_knowing_grading_graph()


async def run_knowing_grading_workflow(
    req: KnowingGradingWorkflowRequest,
    *,
    request_id: str,
    model_adapter: AgentModelAdapter,
    lcms_client: LCMSToolClient,
    knowledge_service: LessonKnowledgeService,
    session_store: Optional[SessionMemoryStore] = None,
) -> KnowingGradingWorkflowResponse:
    """Execute the LangGraph Knowing fixed-essay grading-support workflow with teacher handoff."""
    from app.h12_agent.graph import get_framework_version
    from app.h12_agent.policy import ROLE_POLICIES

    started = time.perf_counter()
    policy = ROLE_POLICIES["grading_support"]
    executor = AuthorizedToolExecutor(
        lcms_client=lcms_client,
        policy=policy,
        assessment_mode=req.assessment_mode,
        caller_scope=req.caller_scope,
        request_id=request_id,
    )

    final_state = await _KNOWING_GRADING_GRAPH.ainvoke(
        {
            "request": req,
            "request_id": request_id,
            "model_adapter": model_adapter,
            "knowledge_service": knowledge_service,
            "executor": executor,
            "observable_steps": [],
        }
    )

    if session_store is not None:
        for out in executor.outcomes:
            session_store.record_tool_audit(
                request_id=request_id,
                session_id=None,
                agent_role="grading_support",
                caller_scope=req.caller_scope,
                outcome=out,
            )

    selected_tools: list[ToolName] = []
    for out in executor.outcomes:
        if out.outcome == "ok" and out.tool_name not in selected_tools:
            selected_tools.append(out.tool_name)
    if "retrieve_lesson_evidence" not in selected_tools:
        selected_tools.append("retrieve_lesson_evidence")

    trace = AgentTrace(
        request_id=request_id,
        framework="langgraph",
        framework_version=get_framework_version(),
        adapter_mode=model_adapter.adapter_mode,
        is_mock=model_adapter.is_mock,
        provider=model_adapter.provider,
        model_name=model_adapter.model_name,
        agent_role="grading_support",
        caller_scope=req.caller_scope,
        assessment_mode=req.assessment_mode,
        visibility_scope=final_state["policy"].visibility_scope,
        allowed_tools=list(final_state["policy"].allowed_tools),
        selected_tools=selected_tools,
        observable_steps=final_state["observable_steps"],
    )

    logger.info(
        "H12 knowing grading support | request_id=%s lesson_id=%s status=%s total=%.2f/%.2f adapter=%s latency_ms=%.1f",
        request_id,
        req.institutional_ref.lesson_id,
        final_state["status"],
        final_state["suggested_total"],
        final_state["rubric"].max_score,
        model_adapter.adapter_mode,
        (time.perf_counter() - started) * 1000,
    )

    return KnowingGradingWorkflowResponse(
        status=final_state["status"],
        institutional_ref=req.institutional_ref,
        question_id=final_state["question"].question_id,
        question_text=final_state["question"].question_text,
        rubric_id=final_state["bundle"].rubric_id,
        suggested_total=final_state["suggested_total"],
        max_score=final_state["rubric"].max_score,
        met_percentage=final_state["met_pct"],
        unmet_required=final_state["unmet_req"],
        criteria=final_state["criteria"],
        feedback=final_state["feedback"],
        flags=final_state["review_flags"],
        warnings=final_state["warning_flags"],
        evidence_refs=final_state["evidence_refs"],
        lcms_receipt=final_state["lcms_receipt"],
        tool_outcomes=executor.outcomes,
        trace=trace,
    )
