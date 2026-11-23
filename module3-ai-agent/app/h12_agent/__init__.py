"""H12 AI Agent core package (Milestones 1 & 2: framework, tools, versioned knowledge, session memory)."""

from app.h12_agent.graph import execute_skeleton, get_framework_version
from app.h12_agent.idempotency import IdempotencyStore, get_idempotency_store
from app.h12_agent.knowledge import LessonIndexResult, LessonKnowledgeService, LessonRetrievalResult
from app.h12_agent.lcms_simulator import create_lcms_simulator_app
from app.h12_agent.lcms_tools import (
    HttpLcmsToolClient,
    LCMSToolClient,
    MockLcmsToolClient,
    ToolOutcome,
    close_lcms_tool_client,
    get_lcms_tool_client,
    get_mock_lcms_client,
)
from app.h12_agent.model_adapter import (
    AgentModelAdapter,
    DeterministicMockModelAdapter,
    LiveGeminiModelAdapter,
    get_agent_model_adapter,
)
from app.h12_agent.policy import (
    FORBIDDEN_ASSESSMENT_MODES,
    FORBIDDEN_CHANNELS,
    FORBIDDEN_QUIZ_FORMATS,
    ROLE_POLICIES,
    enforce_mode_and_scope,
    list_role_capabilities,
    verify_service_auth,
)
from app.h12_agent.session_store import (
    SessionContextWindow,
    SessionMemoryStore,
    SessionRecord,
    SessionTurnRecord,
    get_session_memory_store,
)
from app.h12_agent.tool_executor import AuthorizedToolExecutor
from app.h12_agent.question_drafting import (
    DraftedEssayQuestion,
    QuestionDraftWorkflowRequest,
    QuestionDraftWorkflowResponse,
    run_question_drafting_workflow,
)
from app.h12_agent.knowing_grading import (
    CriterionSuggestion,
    KnowingGradingWorkflowRequest,
    KnowingGradingWorkflowResponse,
    run_knowing_grading_workflow,
)
from app.h12_agent.dialogue_guards import (
    SecurityInspection,
    detect_answer_reveal_attempt,
    detect_redirect_attempt,
    detect_score_alter_attempt,
    detect_student_closure_request,
    guard_assistant_reply,
    inspect_student_message,
)
from app.h12_agent.understanding_dialogue import (
    CloseUnderstandingSessionRequest,
    CloseUnderstandingSessionResponse,
    CriterionUnderstandingStatus,
    StartUnderstandingSessionRequest,
    StartUnderstandingSessionResponse,
    UnderstandingTurnRequest,
    UnderstandingTurnResponse,
    close_understanding_session,
    run_understanding_turn,
    start_understanding_session,
)

__all__ = [
    "AgentModelAdapter",
    "AuthorizedToolExecutor",
    "CloseUnderstandingSessionRequest",
    "CloseUnderstandingSessionResponse",
    "CriterionSuggestion",
    "CriterionUnderstandingStatus",
    "DeterministicMockModelAdapter",
    "DraftedEssayQuestion",
    "FORBIDDEN_ASSESSMENT_MODES",
    "FORBIDDEN_CHANNELS",
    "FORBIDDEN_QUIZ_FORMATS",
    "HttpLcmsToolClient",
    "IdempotencyStore",
    "KnowingGradingWorkflowRequest",
    "KnowingGradingWorkflowResponse",
    "LCMSToolClient",
    "LessonIndexResult",
    "LessonKnowledgeService",
    "LessonRetrievalResult",
    "LiveGeminiModelAdapter",
    "MockLcmsToolClient",
    "QuestionDraftWorkflowRequest",
    "QuestionDraftWorkflowResponse",
    "ROLE_POLICIES",
    "SecurityInspection",
    "SessionContextWindow",
    "SessionMemoryStore",
    "SessionRecord",
    "SessionTurnRecord",
    "StartUnderstandingSessionRequest",
    "StartUnderstandingSessionResponse",
    "ToolOutcome",
    "UnderstandingTurnRequest",
    "UnderstandingTurnResponse",
    "close_lcms_tool_client",
    "close_understanding_session",
    "create_lcms_simulator_app",
    "detect_answer_reveal_attempt",
    "detect_redirect_attempt",
    "detect_score_alter_attempt",
    "detect_student_closure_request",
    "enforce_mode_and_scope",
    "execute_skeleton",
    "get_agent_model_adapter",
    "get_framework_version",
    "get_idempotency_store",
    "get_lcms_tool_client",
    "get_mock_lcms_client",
    "get_session_memory_store",
    "guard_assistant_reply",
    "inspect_student_message",
    "list_role_capabilities",
    "run_knowing_grading_workflow",
    "run_question_drafting_workflow",
    "run_understanding_turn",
    "start_understanding_session",
    "verify_service_auth",
]
