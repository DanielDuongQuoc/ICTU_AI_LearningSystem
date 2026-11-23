"""Role-scoped tool executor enforcing H12 boundaries, classroom-exam locks and outcome auditing."""

from __future__ import annotations

import time
from typing import Optional

from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.lcms_tools import (
    AssessmentContextRecord,
    DialogueOutcomeRecordInput,
    DraftRecordReceipt,
    GradingSuggestionRecordInput,
    LCMSToolClient,
    LessonMaterialRecord,
    LessonQuestionsRubricBundle,
    QuestionDraftRecordInput,
    ToolOutcome,
)
from app.h12_agent.policy import RolePolicy
from app.h12_agent.types import AssessmentMode, CallerScope, ToolName


class AuthorizedToolExecutor:
    """Executes only tools permitted by the active RolePolicy and records observable outcomes."""

    def __init__(
        self,
        *,
        lcms_client: LCMSToolClient,
        policy: RolePolicy,
        assessment_mode: AssessmentMode,
        caller_scope: CallerScope,
        request_id: str = "",
    ) -> None:
        self._lcms = lcms_client
        self._policy = policy
        self._assessment_mode = assessment_mode
        self._caller_scope = caller_scope
        self._request_id = request_id
        self.outcomes: list[ToolOutcome] = []

    def _authorize(self, tool_name: ToolName) -> None:
        if self._assessment_mode == "classroom_exam":
            self.outcomes.append(
                ToolOutcome(
                    tool_name=tool_name,
                    outcome="denied",
                    source_mode=self._lcms.source_mode,
                    detail="classroom_exam mode forbids AI tool execution",
                )
            )
            raise AppError(
                ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
                "AI assistance is strictly prohibited during controlled classroom examinations.",
            )
        if self._caller_scope not in self._policy.allowed_scopes:
            self.outcomes.append(
                ToolOutcome(
                    tool_name=tool_name,
                    outcome="denied",
                    source_mode=self._lcms.source_mode,
                    detail=f"caller_scope '{self._caller_scope}' denied for role '{self._policy.role}'",
                )
            )
            raise AppError(
                ErrorCode.AGENT_SCOPE_FORBIDDEN,
                f"Caller scope '{self._caller_scope}' is not authorized for '{self._policy.role}'.",
            )
        if tool_name not in self._policy.allowed_tools:
            self.outcomes.append(
                ToolOutcome(
                    tool_name=tool_name,
                    outcome="denied",
                    source_mode=self._lcms.source_mode,
                    detail=f"tool '{tool_name}' is outside role '{self._policy.role}' permissions",
                )
            )
            raise AppError(
                ErrorCode.AGENT_SCOPE_FORBIDDEN,
                f"Tool '{tool_name}' is not permitted for role '{self._policy.role}'.",
            )

    def _record_ok(self, tool_name: ToolName, started: float, detail: str) -> None:
        self.outcomes.append(
            ToolOutcome(
                tool_name=tool_name,
                outcome="ok",
                source_mode=self._lcms.source_mode,
                detail=detail,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

    def _record_error(self, tool_name: ToolName, started: float, exc: AppError) -> None:
        self.outcomes.append(
            ToolOutcome(
                tool_name=tool_name,
                outcome="denied" if exc.status_code == 403 else "error",
                source_mode=self._lcms.source_mode,
                detail=exc.code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

    async def fetch_authorized_lesson(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
    ) -> LessonMaterialRecord:
        tool: ToolName = "fetch_authorized_lesson"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            res = await self._lcms.fetch_authorized_lesson(
                lesson_id=lesson_id,
                lesson_version=lesson_version,
                visibility_scope=self._policy.visibility_scope,
                request_id=self._request_id,
            )
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        self._record_ok(tool, started, f"lesson={lesson_id}@{lesson_version} scope={self._policy.visibility_scope}")
        return res

    async def fetch_approved_questions_rubric(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        rubric_id: Optional[str] = None,
    ) -> LessonQuestionsRubricBundle:
        tool: ToolName = "fetch_approved_questions_rubric"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            res = await self._lcms.fetch_approved_questions_rubric(
                lesson_id=lesson_id,
                lesson_version=lesson_version,
                rubric_id=rubric_id,
                visibility_scope=self._policy.visibility_scope,
                request_id=self._request_id,
            )
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        self._record_ok(tool, started, f"rubric={res.rubric_id} scope={self._policy.visibility_scope}")
        return res

    async def fetch_assessment_context(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        student_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AssessmentContextRecord:
        tool: ToolName = "fetch_assessment_context"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            ctx = await self._lcms.fetch_assessment_context(
                lesson_id=lesson_id,
                lesson_version=lesson_version,
                student_id=student_id,
                attempt_id=attempt_id,
                requested_mode=self._assessment_mode,
                request_id=self._request_id,
            )
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        if ctx.classroom_exam_active or not ctx.allowed or ctx.permitted_mode == "classroom_exam":
            detail = ctx.denial_reason or "classroom_exam_active"
            self.outcomes.append(
                ToolOutcome(
                    tool_name=tool,
                    outcome="denied",
                    source_mode=self._lcms.source_mode,
                    detail=detail,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )
            raise AppError(
                ErrorCode.ASSESSMENT_MODE_FORBIDDEN,
                ctx.denial_reason or "AI assistance is strictly prohibited during controlled classroom examinations.",
            )
        self._record_ok(tool, started, f"mode={ctx.permitted_mode} allowed=true")
        return ctx

    async def record_question_draft(self, payload: QuestionDraftRecordInput) -> DraftRecordReceipt:
        tool: ToolName = "record_question_draft"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            res = await self._lcms.record_question_draft(payload, request_id=self._request_id)
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        self._record_ok(tool, started, f"record_id={res.record_id} replay={res.idempotent_replay}")
        return res

    async def record_grading_suggestion(self, payload: GradingSuggestionRecordInput) -> DraftRecordReceipt:
        tool: ToolName = "record_grading_suggestion"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            res = await self._lcms.record_grading_suggestion(payload, request_id=self._request_id)
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        self._record_ok(tool, started, f"record_id={res.record_id} replay={res.idempotent_replay}")
        return res

    async def record_dialogue_outcome(self, payload: DialogueOutcomeRecordInput) -> DraftRecordReceipt:
        tool: ToolName = "record_dialogue_outcome"
        self._authorize(tool)
        started = time.perf_counter()
        try:
            res = await self._lcms.record_dialogue_outcome(payload, request_id=self._request_id)
        except AppError as exc:
            self._record_error(tool, started, exc)
            raise
        self._record_ok(tool, started, f"record_id={res.record_id} replay={res.idempotent_replay}")
        return res
