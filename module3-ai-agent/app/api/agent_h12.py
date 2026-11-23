"""H12 AI Agent API router (Milestones 1 & 2: skeleton, LCMS tools, knowledge & session memory)."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Literal, Optional, TypeVar

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit
from app.core.request_context import get_request_id
from app.h12_agent.graph import execute_skeleton, get_framework_version
from app.h12_agent.idempotency import get_idempotency_store
from app.h12_agent.knowing_grading import (
    KnowingGradingWorkflowRequest,
    KnowingGradingWorkflowResponse,
    run_knowing_grading_workflow,
)
from app.h12_agent.understanding_dialogue import (
    CloseUnderstandingSessionRequest,
    CloseUnderstandingSessionResponse,
    StartUnderstandingSessionRequest,
    StartUnderstandingSessionResponse,
    UnderstandingTurnRequest,
    UnderstandingTurnResponse,
    close_understanding_session,
    run_understanding_turn,
    start_understanding_session,
)
from app.h12_agent.knowledge import LessonKnowledgeService
from app.h12_agent.lcms_tools import LCMSToolClient, ToolOutcome, get_lcms_tool_client
from app.h12_agent.model_adapter import get_agent_model_adapter
from app.h12_agent.policy import (
    FORBIDDEN_ASSESSMENT_MODES,
    FORBIDDEN_CHANNELS,
    FORBIDDEN_QUIZ_FORMATS,
    ROLE_POLICIES,
    enforce_mode_and_scope,
    list_role_capabilities,
    verify_service_auth,
)
from app.h12_agent.question_drafting import (
    QuestionDraftWorkflowRequest,
    QuestionDraftWorkflowResponse,
    run_question_drafting_workflow,
)
from app.h12_agent.session_store import (
    SessionContextWindow,
    SessionMemoryStore,
    SessionRecord,
    SessionTurnRecord,
    get_session_memory_store,
)
from app.h12_agent.tool_executor import AuthorizedToolExecutor
from app.h12_agent.types import (
    AgentCapabilitiesResponse,
    AgentRole,
    AgentSkeletonRequest,
    AgentSkeletonResponse,
    AssessmentMode,
    CallerScope,
    EvidenceRef,
    InstitutionalRef,
    VisibilityScope,
)
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/agent", tags=["H12 Agent"])

_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _get_rag_service() -> RAGService:
    from app.api.rag import _get_rag_service as get_shared_rag

    return get_shared_rag()


def _get_llm_service() -> LLMService:
    return get_llm_service()


def _get_lcms_client() -> LCMSToolClient:
    return get_lcms_tool_client()


def _get_knowledge_service() -> LessonKnowledgeService:
    from app.api.documents import _get_vector_service

    vector = _get_vector_service()
    return LessonKnowledgeService(
        embedding_service=vector._embedding,
        chroma_repo=vector._repo,
        vector_service=vector,
    )


def _get_session_store() -> SessionMemoryStore:
    return get_session_memory_store()


_RespT = TypeVar("_RespT", bound=BaseModel)


def _set_agent_headers(response: Response, result: Any, *, replayed: bool) -> None:
    response.headers["X-AI-Adapter-Mode"] = result.trace.adapter_mode
    response.headers["X-AI-Framework"] = result.trace.framework
    response.headers["X-Idempotent-Replay"] = "true" if replayed else "false"


async def _run_with_idempotency(
    *,
    endpoint: str,
    body: BaseModel,
    response_model: type[_RespT],
    response: Response,
    idempotency_key_header: Optional[str],
    runner: Callable[[Optional[str]], Awaitable[_RespT]],
) -> _RespT:
    body_key = getattr(body, "idempotency_key", None)
    eff_key = (idempotency_key_header or body_key or "").strip() or None
    if eff_key is not None and not _IDEMPOTENCY_RE.fullmatch(eff_key):
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "Idempotency-Key must match ^[A-Za-z0-9._:-]{1,128}$.",
            status_code=422,
        )

    settings = get_settings()
    store = get_idempotency_store()
    fingerprint = ""
    if eff_key is not None:
        fingerprint = store.compute_fingerprint(endpoint, body.model_dump(mode="json"))
        cached = store.lookup_or_conflict(
            endpoint=endpoint,
            key=eff_key,
            fingerprint=fingerprint,
            ttl_s=settings.agent_idempotency_ttl_s,
        )
        if cached is not None:
            replayed = response_model.model_validate({**cached, "idempotent_replay": True})
            _set_agent_headers(response, replayed, replayed=True)
            return replayed

    result = await runner(eff_key)
    if eff_key is not None:
        store.save(
            endpoint=endpoint,
            key=eff_key,
            fingerprint=fingerprint,
            response_data=result.model_dump(mode="json"),
            ttl_s=settings.agent_idempotency_ttl_s,
        )

    _set_agent_headers(response, result, replayed=False)
    return result


# ── Milestone 2 Request / Response Schemas ─────────────────────────────────


class LessonKnowledgeIndexRequest(BaseModel):
    """Index versioned, teacher-approved lesson knowledge (direct payload or pulled via LCMS tool)."""

    caller_scope: CallerScope
    lesson_id: str = Field(..., pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    knowledge_version: str = Field("v1", pattern=r"^[A-Za-z0-9._:-]{1,32}$")
    from_lcms: bool = False
    title: Optional[str] = Field(None, max_length=200)
    student_safe_content: Optional[str] = Field(None, max_length=200_000)
    teacher_only_notes: Optional[str] = Field(None, max_length=50_000)
    approved: bool = True


class LessonKnowledgeIndexResponse(BaseModel):
    """Result of indexing a versioned lesson with student_safe / teacher_only isolation."""

    lesson_id: str
    knowledge_version: str
    approved: bool
    student_safe_chunks: int
    teacher_only_chunks: int
    total_chunks: int
    source_mode: Literal["mock", "http", "direct"]


class LessonKnowledgeRetrieveRequest(BaseModel):
    """Retrieve version-scoped lesson chunks constrained by caller role and visibility scope."""

    agent_role: AgentRole
    caller_scope: CallerScope
    assessment_mode: AssessmentMode
    institutional_ref: InstitutionalRef
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class LessonKnowledgeRetrieveResponse(BaseModel):
    """Visibility-filtered retrieval response for a lesson version."""

    lesson_id: str
    knowledge_version: str
    visibility_scope: VisibilityScope
    insufficient_context: bool
    context: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    """Open or resume an isolated per-(student, lesson, attempt) session verified with LCMS."""

    caller_scope: CallerScope
    assessment_mode: Literal["home_knowing", "home_understanding", "classroom_exam"]
    institutional_ref: InstitutionalRef
    max_turns: Optional[int] = Field(None, ge=1, le=20)


class SessionDetailResponse(BaseModel):
    """Isolated session metadata, turn history, context window and tool outcomes."""

    session: SessionRecord
    created: bool
    turns: list[SessionTurnRecord] = Field(default_factory=list)
    context_window: SessionContextWindow
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)


# ── Milestone 1 Endpoints ──────────────────────────────────────────────────


@router.get(
    "/capabilities",
    response_model=AgentCapabilitiesResponse,
    summary="Inspect H12 AI Agent role boundaries, framework and adapter mode",
    description=(
        "Returns the active LangGraph agent framework version, whether the model adapter is in `mock` or "
        "`live` mode, forbidden examination modes (`classroom_exam`) and formats (multiple choice / voice), "
        "plus permitted tools and stopping conditions for each H12 agent role."
    ),
    responses=error_responses(500),
)
async def get_agent_capabilities() -> AgentCapabilitiesResponse:
    settings = get_settings()
    is_mock = settings.agent_model_adapter == "mock"
    return AgentCapabilitiesResponse(
        framework="langgraph",
        framework_version=get_framework_version(),
        adapter_mode="mock" if is_mock else "live",
        is_mock=is_mock,
        lcms_mode=settings.agent_lcms_mode,  # type: ignore[arg-type]
        auth_required=settings.agent_auth_required,
        forbidden_modes=list(FORBIDDEN_ASSESSMENT_MODES),
        forbidden_formats=list(FORBIDDEN_QUIZ_FORMATS),
        forbidden_channels=list(FORBIDDEN_CHANNELS),
        roles=list_role_capabilities(),
    )


@router.post(
    "/execute-skeleton",
    response_model=AgentSkeletonResponse,
    summary="Execute the controlled H12 LangGraph agent skeleton",
    description=(
        "Validates service authentication, assessment mode (forbidding `classroom_exam`), format/channel rules "
        "(essay for Knowing, interactive text_chat for Understanding), role/scope permissions, and idempotency; "
        "then runs the 5-step LangGraph workflow with optional ChromaDB retrieval and replaceable model adapter "
        "(`mock` vs `live`). All outputs remain `draft_suggestion` with `released=false`."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def execute_agent_skeleton(
    body: AgentSkeletonRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
    idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> AgentSkeletonResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()

    async def _run(_eff_key: Optional[str]) -> AgentSkeletonResponse:
        llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
        adapter = get_agent_model_adapter(llm_service=llm_svc)
        rag_svc = _get_rag_service() if body.document_id is not None else None
        return await execute_skeleton(
            body,
            request_id=get_request_id(request),
            model_adapter=adapter,
            rag_service=rag_svc,
        )

    return await _run_with_idempotency(
        endpoint="/api/agent/execute-skeleton",
        body=body,
        response_model=AgentSkeletonResponse,
        response=response,
        idempotency_key_header=idempotency_key_header,
        runner=_run,
    )


# ── Milestone 2 Endpoints: Versioned Knowledge & Isolated Sessions ─────────


@router.post(
    "/knowledge/index",
    response_model=LessonKnowledgeIndexResponse,
    summary="Index versioned, approved lesson knowledge with visibility isolation",
    description=(
        "Indexes a teacher-approved lesson version into ChromaDB, separating `student_safe` material from "
        "`teacher_only` answer-key notes. When `from_lcms=true`, pulls the lesson through the authorized "
        "LCMS tool layer (`mock` or `http`). Rejects student callers and unapproved content."
    ),
    responses=error_responses(400, 401, 403, 404, 422, 429, 502, 503),
    dependencies=[Depends(rate_limit())],
)
async def index_lesson_knowledge(
    body: LessonKnowledgeIndexRequest,
    request: Request,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> LessonKnowledgeIndexResponse:
    verify_service_auth(x_agent_service_key)
    if body.caller_scope == "student":
        raise AppError(
            ErrorCode.AGENT_SCOPE_FORBIDDEN,
            "Students are not authorized to index lesson knowledge.",
        )

    req_id = get_request_id(request)
    if body.from_lcms:
        executor = AuthorizedToolExecutor(
            lcms_client=_get_lcms_client(),
            policy=ROLE_POLICIES["question_drafting"],
            assessment_mode="teacher_question_drafting",
            caller_scope=body.caller_scope,
            request_id=req_id,
        )
        material = await executor.fetch_authorized_lesson(
            lesson_id=body.lesson_id,
            lesson_version=body.knowledge_version,
        )
        title = material.title
        student_safe_content = material.student_safe_content
        teacher_only_notes = material.teacher_only_notes
        approved = material.approved
        source_mode: Literal["mock", "http", "direct"] = material.source_mode
    else:
        title = (body.title or body.lesson_id).strip()
        student_safe_content = body.student_safe_content or ""
        teacher_only_notes = body.teacher_only_notes
        approved = body.approved
        source_mode = "direct"

    res = await _get_knowledge_service().index_lesson_material(
        lesson_id=body.lesson_id,
        knowledge_version=body.knowledge_version,
        title=title,
        student_safe_content=student_safe_content,
        approved=approved,
        teacher_only_notes=teacher_only_notes,
    )
    return LessonKnowledgeIndexResponse(
        lesson_id=res.lesson_id,
        knowledge_version=res.knowledge_version,
        approved=res.approved,
        student_safe_chunks=res.student_safe_chunks,
        teacher_only_chunks=res.teacher_only_chunks,
        total_chunks=res.total_chunks,
        source_mode=source_mode,
    )


@router.post(
    "/knowledge/retrieve",
    response_model=LessonKnowledgeRetrieveResponse,
    summary="Retrieve version-scoped lesson evidence honoring visibility isolation",
    description=(
        "Validates mode and caller scope (forbidding `classroom_exam`), then retrieves chunks strictly filtered "
        "by `lesson_id`, `lesson_version`, `approved=true`, and `visibility_scope` so `teacher_only` answer "
        "keys cannot leak into student-facing Understanding sessions."
    ),
    responses=error_responses(400, 401, 403, 422, 429, 500, 502, 503),
    dependencies=[Depends(rate_limit())],
)
async def retrieve_lesson_knowledge(
    body: LessonKnowledgeRetrieveRequest,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> LessonKnowledgeRetrieveResponse:
    verify_service_auth(x_agent_service_key)
    policy = enforce_mode_and_scope(
        agent_role=body.agent_role,
        caller_scope=body.caller_scope,
        assessment_mode=body.assessment_mode,
        institutional_ref=body.institutional_ref,
        format_hint=None,
        channel="text_chat",
    )
    res = await _get_knowledge_service().retrieve_lesson_evidence(
        lesson_id=body.institutional_ref.lesson_id,
        knowledge_version=body.institutional_ref.lesson_version,
        query=body.query,
        visibility_scope=policy.visibility_scope,
        top_k=body.top_k,
    )
    return LessonKnowledgeRetrieveResponse(
        lesson_id=res.lesson_id,
        knowledge_version=res.knowledge_version,
        visibility_scope=res.visibility_scope,
        insufficient_context=res.insufficient_context,
        context=res.context,
        evidence_refs=res.evidence_refs,
        chunks=res.chunks,
    )


@router.post(
    "/sessions",
    response_model=SessionDetailResponse,
    summary="Create or resume an isolated per-student, lesson and attempt session",
    description=(
        "Verifies with the authorized LCMS tool `fetch_assessment_context` that the student is not inside a "
        "controlled classroom examination (`classroom_exam`), then opens or resumes a SQLite session isolated "
        "strictly by `(student_id, lesson_id, attempt_id)`."
    ),
    responses=error_responses(400, 401, 403, 404, 422, 429, 502, 503),
    dependencies=[Depends(rate_limit())],
)
async def create_or_resume_session(
    body: CreateSessionRequest,
    request: Request,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> SessionDetailResponse:
    verify_service_auth(x_agent_service_key)
    if body.assessment_mode == "classroom_exam":
        raise AppError(
            ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
            "AI assistance is strictly prohibited during controlled classroom examinations.",
        )

    ref = body.institutional_ref
    if not ref.student_id or not ref.attempt_id:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "student_id and attempt_id are required to create an isolated assessment session.",
            status_code=422,
        )

    role: AgentRole = "dialogue_assessment" if body.assessment_mode == "home_understanding" else "grading_support"
    policy = ROLE_POLICIES[role]
    if body.caller_scope not in policy.allowed_scopes:
        raise AppError(
            ErrorCode.AGENT_SCOPE_FORBIDDEN,
            f"Caller scope '{body.caller_scope}' is not authorized for '{body.assessment_mode}'.",
        )

    req_id = get_request_id(request)
    executor = AuthorizedToolExecutor(
        lcms_client=_get_lcms_client(),
        policy=policy,
        assessment_mode=body.assessment_mode,
        caller_scope=body.caller_scope,
        request_id=req_id,
    )
    ctx = await executor.fetch_assessment_context(
        lesson_id=ref.lesson_id,
        lesson_version=ref.lesson_version,
        student_id=ref.student_id,
        attempt_id=ref.attempt_id,
    )

    store = _get_session_store()
    session, created = store.create_or_get_session(
        student_id=ref.student_id,
        lesson_id=ref.lesson_id,
        lesson_version=ref.lesson_version,
        attempt_id=ref.attempt_id,
        assessment_mode=body.assessment_mode,
        course_id=ref.course_id or ctx.course_id or None,
        assignment_id=ref.assignment_id or ctx.assignment_id,
        rubric_id=ref.rubric_id,
        max_turns=body.max_turns or ctx.max_turns,
    )
    for outcome in executor.outcomes:
        store.record_tool_audit(
            request_id=req_id,
            session_id=session.session_id,
            agent_role=role,
            caller_scope=body.caller_scope,
            outcome=outcome,
        )

    turns = store.list_turns(session.session_id, student_id=ref.student_id)
    window = store.build_model_context(session.session_id, student_id=ref.student_id)
    return SessionDetailResponse(
        session=session,
        created=created,
        turns=turns,
        context_window=window,
        tool_outcomes=executor.outcomes,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="Inspect an isolated assessment session owned by the given student",
    description=(
        "Returns metadata, turn history, and bounded context window for `session_id`, enforcing strict "
        "ownership isolation by `student_id` (and optionally `lesson_id` / `attempt_id`)."
    ),
    responses=error_responses(401, 403, 404, 422, 500),
)
async def get_isolated_session(
    session_id: str,
    student_id: str = Query(..., pattern=r"^[A-Za-z0-9._:-]{1,64}$"),
    lesson_id: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$"),
    attempt_id: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9._:-]{1,64}$"),
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> SessionDetailResponse:
    verify_service_auth(x_agent_service_key)
    store = _get_session_store()
    session = store.get_session_for_owner(
        session_id,
        student_id=student_id,
        lesson_id=lesson_id,
        attempt_id=attempt_id,
    )
    turns = store.list_turns(session_id, student_id=student_id)
    window = store.build_model_context(session_id, student_id=student_id)
    return SessionDetailResponse(
        session=session,
        created=False,
        turns=turns,
        context_window=window,
        tool_outcomes=[],
    )


# ── Milestone 3 Endpoints: Question Drafting & Knowing Essay Grading ───────


@router.post(
    "/questions/draft",
    response_model=QuestionDraftWorkflowResponse,
    summary="Draft lesson-grounded essay questions and hand off to LCMS for teacher review",
    description=(
        "Executes the 6-step LangGraph teacher question-drafting workflow: validates `essay`-only format, "
        "fetches approved lesson content from LCMS, retrieves version-scoped ChromaDB evidence, generates "
        "essay question candidates via the active model adapter (`mock` or `live`), and hands off unpublished "
        "drafts (`publication_status=draft_suggestion`, `released=false`) to LCMS."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def draft_lesson_essay_questions(
    body: QuestionDraftWorkflowRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
    idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> QuestionDraftWorkflowResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()

    async def _run(eff_key: Optional[str]) -> QuestionDraftWorkflowResponse:
        llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
        adapter = get_agent_model_adapter(llm_service=llm_svc)
        effective_req = body if eff_key is None else body.model_copy(update={"idempotency_key": eff_key})
        return await run_question_drafting_workflow(
            effective_req,
            request_id=get_request_id(request),
            model_adapter=adapter,
            lcms_client=_get_lcms_client(),
            knowledge_service=_get_knowledge_service(),
        )

    return await _run_with_idempotency(
        endpoint="/api/agent/questions/draft",
        body=body,
        response_model=QuestionDraftWorkflowResponse,
        response=response,
        idempotency_key_header=idempotency_key_header,
        runner=_run,
    )


@router.post(
    "/knowing/grade-suggestion",
    response_model=KnowingGradingWorkflowResponse,
    summary="Generate criterion-linked scoring suggestions for a fixed Knowing essay answer",
    description=(
        "Executes the 7-step LangGraph Knowing essay grading-support workflow: enforces `essay`-only format, "
        "checks LCMS assessment context (forbidding active `classroom_exam`), retrieves the approved rubric and "
        "lesson evidence, computes deterministic criterion points without auto-publishing grades, and hands off "
        "a `draft_suggestion` (`pending_teacher_review` or `needs_teacher_escalation`) to the lecturer."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def suggest_knowing_essay_grading(
    body: KnowingGradingWorkflowRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
    idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> KnowingGradingWorkflowResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()

    async def _run(eff_key: Optional[str]) -> KnowingGradingWorkflowResponse:
        llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
        adapter = get_agent_model_adapter(llm_service=llm_svc)
        effective_req = body if eff_key is None else body.model_copy(update={"idempotency_key": eff_key})
        return await run_knowing_grading_workflow(
            effective_req,
            request_id=get_request_id(request),
            model_adapter=adapter,
            lcms_client=_get_lcms_client(),
            knowledge_service=_get_knowledge_service(),
            session_store=_get_session_store(),
        )

    return await _run_with_idempotency(
        endpoint="/api/agent/knowing/grade-suggestion",
        body=body,
        response_model=KnowingGradingWorkflowResponse,
        response=response,
        idempotency_key_header=idempotency_key_header,
        runner=_run,
    )


@router.post(
    "/understanding/sessions/start",
    response_model=StartUnderstandingSessionResponse,
    summary="Start or resume an isolated Understanding adaptive text chat session",
    description=(
        "Validates `home_understanding` text-only mode and LCMS assessment context (denying `classroom_exam` "
        "and voice/oral channels), retrieves `student_safe` approved lesson evidence and rubric criteria without "
        "teacher notes, initializes a student/lesson/attempt isolated session, and returns the opening question."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def start_understanding_dialogue_session(
    body: StartUnderstandingSessionRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> StartUnderstandingSessionResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()
    llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
    adapter = get_agent_model_adapter(llm_service=llm_svc)
    result = await start_understanding_session(
        body,
        request_id=get_request_id(request),
        model_adapter=adapter,
        lcms_client=_get_lcms_client(),
        knowledge_service=_get_knowledge_service(),
        session_store=_get_session_store(),
    )
    _set_agent_headers(response, result, replayed=False)
    return result


@router.post(
    "/understanding/sessions/{session_id}/turns",
    response_model=UnderstandingTurnResponse,
    summary="Run one adaptive Understanding dialogue turn with security guards",
    description=(
        "Executes the conditional 7-node LangGraph Understanding dialogue workflow for an active session: "
        "verifies ownership, mode, and real-time `classroom_exam` status, retrieves `student_safe` evidence, "
        "blocks prompt-injection/redirect/answer-reveal/score-alter attempts, tracks rubric progression, and "
        "auto-closes and hands off the outcome when criteria are covered, max turns are reached, student stops, "
        "or repeated injection escalates."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def submit_understanding_dialogue_turn(
    session_id: str,
    body: UnderstandingTurnRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
    idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> UnderstandingTurnResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()

    async def _run(eff_key: Optional[str]) -> UnderstandingTurnResponse:
        llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
        adapter = get_agent_model_adapter(llm_service=llm_svc)
        effective_req = body if eff_key is None else body.model_copy(update={"idempotency_key": eff_key})
        return await run_understanding_turn(
            session_id,
            effective_req,
            request_id=get_request_id(request),
            model_adapter=adapter,
            lcms_client=_get_lcms_client(),
            knowledge_service=_get_knowledge_service(),
            session_store=_get_session_store(),
        )

    return await _run_with_idempotency(
        endpoint=f"/api/agent/understanding/sessions/{session_id}/turns",
        body=body,
        response_model=UnderstandingTurnResponse,
        response=response,
        idempotency_key_header=idempotency_key_header,
        runner=_run,
    )


@router.post(
    "/understanding/sessions/{session_id}/close",
    response_model=CloseUnderstandingSessionResponse,
    summary="Explicitly conclude an active Understanding dialogue session",
    description=(
        "Verifies ownership and active session state, ensures classroom examination lock is not active, "
        "closes the isolated Understanding session cleanly (`student_requested_closure`), and records the "
        "final criterion coverage and evidence references in LCMS without auto-publishing grades."
    ),
    responses=error_responses(400, 401, 403, 404, 409, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def close_understanding_dialogue_session(
    session_id: str,
    body: CloseUnderstandingSessionRequest,
    request: Request,
    response: Response,
    x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
) -> CloseUnderstandingSessionResponse:
    verify_service_auth(x_agent_service_key)
    settings = get_settings()
    llm_svc = None if settings.agent_model_adapter == "mock" else _get_llm_service()
    adapter = get_agent_model_adapter(llm_service=llm_svc)
    result = await close_understanding_session(
        session_id,
        body,
        request_id=get_request_id(request),
        model_adapter=adapter,
        lcms_client=_get_lcms_client(),
        session_store=_get_session_store(),
    )
    _set_agent_headers(response, result, replayed=False)
    return result
