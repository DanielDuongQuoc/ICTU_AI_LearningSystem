"""Authorized LCMS tool contracts, Mock LCMS client, and HTTP LCMS adapter (PI 2.1, PI 2.2, PI 3.3)."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.types import (
    AssessmentMode,
    EvidenceRef,
    InstitutionalRef,
    ToolName,
    VisibilityScope,
)


class ToolOutcome(BaseModel):
    """Observable outcome of one authorized tool invocation."""

    tool_name: ToolName
    outcome: Literal["ok", "denied", "error"]
    source_mode: Literal["mock", "http", "local"]
    detail: str = ""
    duration_ms: float = 0.0


class LessonMaterialRecord(BaseModel):
    """Lesson material fetched from LCMS with visibility separation."""

    lesson_id: str
    lesson_version: str = "v1"
    course_id: str = ""
    title: str
    approved: bool
    approved_by: Optional[str] = None
    student_safe_content: str
    teacher_only_notes: Optional[str] = None
    document_id: Optional[str] = None
    source_mode: Literal["mock", "http"] = "mock"


class LessonQuestionRecord(BaseModel):
    """Approved lesson question metadata (Knowing fixed essay or Understanding chat prompt)."""

    question_id: str
    lesson_id: str
    lesson_version: str = "v1"
    assessment_level: Literal["knowing", "understanding"]
    format: Literal["essay", "adaptive_text_chat"]
    question_text: str
    approved: bool
    answer_key: Optional[str] = None


class LessonQuestionsRubricBundle(BaseModel):
    """Approved questions and rubric bundle fetched from LCMS."""

    lesson_id: str
    lesson_version: str = "v1"
    rubric_id: str
    rubric_status: Literal["draft", "approved"]
    rubric: dict[str, Any]
    questions: list[LessonQuestionRecord] = Field(default_factory=list)
    source_mode: Literal["mock", "http"] = "mock"


class AssessmentContextRecord(BaseModel):
    """Institutional assessment context and classroom-exam lock status from LCMS."""

    lesson_id: str
    lesson_version: str = "v1"
    course_id: str = ""
    assignment_id: Optional[str] = None
    student_id: Optional[str] = None
    attempt_id: Optional[str] = None
    permitted_mode: AssessmentMode
    classroom_exam_active: bool = False
    max_turns: int = 6
    allowed: bool = True
    denial_reason: Optional[str] = None
    source_mode: Literal["mock", "http"] = "mock"


class QuestionDraftRecordInput(BaseModel):
    """Payload recorded back to LCMS as an unapproved teacher question draft."""

    institutional_ref: InstitutionalRef
    question_text: str = Field(..., min_length=1, max_length=4000)
    expected_answer: str = Field(..., min_length=1, max_length=6000)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    adapter_mode: Literal["mock", "live"]
    idempotency_key: str = Field(..., min_length=1, max_length=128)


class GradingSuggestionRecordInput(BaseModel):
    """Payload recorded back to LCMS as a provisional criterion-linked grading suggestion."""

    institutional_ref: InstitutionalRef
    suggested_total: float = Field(..., ge=0.0, le=100.0)
    max_score: float = Field(..., gt=0.0, le=100.0)
    criteria_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    feedback: str = Field(..., min_length=1, max_length=4000)
    flags: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    adapter_mode: Literal["mock", "live"]
    idempotency_key: str = Field(..., min_length=1, max_length=128)


class DialogueOutcomeRecordInput(BaseModel):
    """Payload recorded back to LCMS when an Understanding dialogue session closes."""

    institutional_ref: InstitutionalRef
    session_id: str = Field(..., min_length=1, max_length=64)
    stopping_reason: str = Field(..., min_length=1, max_length=64)
    turn_count: int = Field(..., ge=0)
    criterion_assessments: list[dict[str, Any]] = Field(default_factory=list)
    summary_feedback: str = Field(..., min_length=1, max_length=4000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    adapter_mode: Literal["mock", "live"]
    idempotency_key: str = Field(..., min_length=1, max_length=128)


class DraftRecordReceipt(BaseModel):
    """Receipt returned by LCMS confirming draft storage without silent grade publication."""

    record_id: str
    record_type: Literal["question_draft", "grading_suggestion", "dialogue_outcome"]
    publication_status: Literal["draft_suggestion"] = "draft_suggestion"
    released: bool = False
    teacher_review_required: bool = True
    idempotent_replay: bool = False
    source_mode: Literal["mock", "http"] = "mock"
    recorded_at: str


def sanitize_lesson_for_visibility(
    record: LessonMaterialRecord,
    visibility_scope: VisibilityScope,
) -> LessonMaterialRecord:
    """Strip teacher-only notes / answer keys when visibility is student_safe."""
    if visibility_scope == "student_safe":
        return record.model_copy(update={"teacher_only_notes": None})
    return record


def sanitize_bundle_for_visibility(
    bundle: LessonQuestionsRubricBundle,
    visibility_scope: VisibilityScope,
) -> LessonQuestionsRubricBundle:
    """Strip answer keys and detailed rubric key_points when visibility is student_safe."""
    if visibility_scope != "student_safe":
        return bundle

    cleaned_questions = [
        q.model_copy(update={"answer_key": None}) for q in bundle.questions
    ]
    cleaned_rubric = copy.deepcopy(bundle.rubric)
    for crit in cleaned_rubric.get("criteria", []):
        if isinstance(crit, dict):
            crit.pop("key_points", None)
    return bundle.model_copy(
        update={"questions": cleaned_questions, "rubric": cleaned_rubric}
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload_hash(payload: BaseModel) -> str:
    raw = payload.model_dump(mode="json")
    raw.pop("idempotency_key", None)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LCMSToolClient(ABC):
    """Abstract interface for authorized LCMS tool operations."""

    @property
    @abstractmethod
    def source_mode(self) -> Literal["mock", "http"]:
        """'mock' or 'http'."""

    @abstractmethod
    async def fetch_authorized_lesson(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonMaterialRecord:
        """Fetch an approved lesson version, sanitized for the caller's visibility scope."""

    @abstractmethod
    async def fetch_approved_questions_rubric(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        rubric_id: Optional[str] = None,
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonQuestionsRubricBundle:
        """Fetch approved lesson questions and rubric, stripping keys in student_safe scope."""

    @abstractmethod
    async def fetch_assessment_context(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        student_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        requested_mode: AssessmentMode = "home_understanding",
        request_id: str = "",
    ) -> AssessmentContextRecord:
        """Verify permitted assessment mode and whether a controlled classroom exam is active."""

    @abstractmethod
    async def record_question_draft(
        self,
        payload: QuestionDraftRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        """Persist an unapproved question draft in LCMS for teacher review."""

    @abstractmethod
    async def record_grading_suggestion(
        self,
        payload: GradingSuggestionRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        """Persist a provisional criterion-linked grading suggestion for teacher handoff."""

    @abstractmethod
    async def record_dialogue_outcome(
        self,
        payload: DialogueOutcomeRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        """Persist a completed Understanding dialogue outcome and criterion evidence."""


class MockLcmsToolClient(LCMSToolClient):
    """Documented in-memory mock LCMS client for isolated tests and independent demos."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lessons: dict[tuple[str, str], LessonMaterialRecord] = {}
        self._bundles: dict[tuple[str, str], LessonQuestionsRubricBundle] = {}
        self._classroom_exam_students: set[str] = set()
        self._idempotency: dict[str, tuple[str, DraftRecordReceipt]] = {}
        self._records: list[dict[str, Any]] = []
        self.reset()

    @property
    def source_mode(self) -> Literal["mock", "http"]:
        return "mock"

    def reset(self) -> None:
        with self._lock:
            self._classroom_exam_students = {"SV_IN_EXAM_01"}
            self._idempotency.clear()
            self._records.clear()
            self._seed_defaults()

    def register_lesson(
        self,
        lesson: LessonMaterialRecord,
        bundle: Optional[LessonQuestionsRubricBundle] = None,
    ) -> None:
        """Register an additional lesson (and optional rubric/question bundle) for evaluation or demos."""
        with self._lock:
            self._lessons[(lesson.lesson_id, lesson.lesson_version)] = lesson
            if bundle is not None:
                self._bundles[(bundle.lesson_id, bundle.lesson_version)] = bundle

    def _seed_defaults(self) -> None:
        self._lessons = {
            ("lesson-db-01", "v1"): LessonMaterialRecord(
                lesson_id="lesson-db-01",
                lesson_version="v1",
                course_id="CS301",
                title="Database Transactions & ACID Properties",
                approved=True,
                approved_by="lecturer@ictu.edu.vn",
                student_safe_content=(
                    "A database transaction is a single logical unit of work that accesses and possibly updates "
                    "data items. Transactions must guarantee four ACID properties: Atomicity (all operations "
                    "succeed or none are applied via rollback), Consistency (valid state transitions preserving "
                    "integrity constraints), Isolation (concurrent transactions do not expose intermediate dirty "
                    "states), and Durability (committed updates survive crashes through write-ahead logs)."
                ),
                teacher_only_notes=(
                    "ANSWER_KEY_CONFIDENTIAL_ACID: Full credit requires defining rollback for Atomicity, "
                    "constraint preservation for Consistency, serializability/isolation of dirty reads, and WAL "
                    "persistence for Durability."
                ),
                source_mode="mock",
            ),
            ("lesson-db-01", "v2"): LessonMaterialRecord(
                lesson_id="lesson-db-01",
                lesson_version="v2",
                course_id="CS301",
                title="Database Transactions, ACID & Isolation Levels (v2)",
                approved=True,
                approved_by="lecturer@ictu.edu.vn",
                student_safe_content=(
                    "Building on ACID fundamentals, SQL isolation levels balance concurrency against anomalies: "
                    "Read Committed prevents dirty reads, Repeatable Read prevents non-repeatable reads, and "
                    "Serializable prevents phantom reads. Write-Ahead Logging (WAL) guarantees Atomicity and "
                    "Durability during recovery."
                ),
                teacher_only_notes=(
                    "ANSWER_KEY_CONFIDENTIAL_V2: Expect explicit mapping from anomalies (dirty, non-repeatable, "
                    "phantom) to SQL isolation levels and WAL recovery."
                ),
                source_mode="mock",
            ),
            ("lesson-se-02", "v1"): LessonMaterialRecord(
                lesson_id="lesson-se-02",
                lesson_version="v1",
                course_id="SE202",
                title="Layered Architecture & Explicit Boundaries",
                approved=True,
                approved_by="lecturer@ictu.edu.vn",
                student_safe_content=(
                    "Layered architecture separates presentation, orchestration, tool execution, and storage "
                    "behind explicit contracts so external integrations can be replaced by documented mocks "
                    "without altering domain policies."
                ),
                teacher_only_notes=(
                    "ANSWER_KEY_CONFIDENTIAL_SE: Look for separation of concerns, boundary interfaces, and "
                    "dependency inversion."
                ),
                source_mode="mock",
            ),
            ("lesson-draft-99", "v1"): LessonMaterialRecord(
                lesson_id="lesson-draft-99",
                lesson_version="v1",
                course_id="CS301",
                title="Unapproved Draft Lesson",
                approved=False,
                approved_by=None,
                student_safe_content="Draft content not yet approved by the lecturer.",
                teacher_only_notes="Internal draft notes.",
                source_mode="mock",
            ),
        }

        acid_rubric = {
            "max_score": 10.0,
            "grounded": True,
            "status": "approved",
            "pass_rule": {"all_required_met": True, "min_percentage": 60.0},
            "criteria": [
                {
                    "criterion_id": "C1",
                    "name": "Transaction & Atomicity",
                    "description": "Explains transaction as an atomic all-or-nothing logical unit.",
                    "max_points": 5.0,
                    "required": True,
                    "key_points": [
                        "ANSWER_KEY_CONFIDENTIAL_ACID: Mention rollback on failure and all-or-nothing execution"
                    ],
                    "source_chunks": ["lesson-db-01:v1:c1"],
                    "levels": [
                        {"label": "Full", "points": 5.0, "descriptor": "Accurate definition and rollback role."},
                        {"label": "Partial", "points": 2.5, "descriptor": "Mentions concept without rollback."},
                        {"label": "Missing", "points": 0.0, "descriptor": "Incorrect or absent."},
                    ],
                },
                {
                    "criterion_id": "C2",
                    "name": "Consistency, Isolation & Durability",
                    "description": "Explains constraint preservation, concurrency isolation, and crash durability.",
                    "max_points": 5.0,
                    "required": False,
                    "key_points": [
                        "ANSWER_KEY_CONFIDENTIAL_ACID: Explain WAL / persistent commit and dirty read prevention"
                    ],
                    "source_chunks": ["lesson-db-01:v1:c1"],
                    "levels": [
                        {"label": "Full", "points": 5.0, "descriptor": "Covers C, I, D accurately."},
                        {"label": "Partial", "points": 2.5, "descriptor": "Covers at least one property."},
                        {"label": "Missing", "points": 0.0, "descriptor": "Incorrect or absent."},
                    ],
                },
            ],
        }

        self._bundles = {
            ("lesson-db-01", "v1"): LessonQuestionsRubricBundle(
                lesson_id="lesson-db-01",
                lesson_version="v1",
                rubric_id="rubric-db-01-v1",
                rubric_status="approved",
                rubric=acid_rubric,
                questions=[
                    LessonQuestionRecord(
                        question_id="q-db-01-knowing",
                        lesson_id="lesson-db-01",
                        lesson_version="v1",
                        assessment_level="knowing",
                        format="essay",
                        question_text="Explain what a database transaction is and describe the four ACID properties.",
                        approved=True,
                        answer_key="ANSWER_KEY_CONFIDENTIAL_ACID: A=rollback, C=constraints, I=isolation, D=WAL.",
                    ),
                    LessonQuestionRecord(
                        question_id="q-db-01-understanding",
                        lesson_id="lesson-db-01",
                        lesson_version="v1",
                        assessment_level="understanding",
                        format="adaptive_text_chat",
                        question_text="Why does Atomicity alone fail to protect concurrent transactions without Isolation?",
                        approved=True,
                        answer_key="ANSWER_KEY_CONFIDENTIAL_ACID: Dirty reads of uncommitted data before rollback.",
                    ),
                ],
                source_mode="mock",
            ),
            ("lesson-db-01", "v2"): LessonQuestionsRubricBundle(
                lesson_id="lesson-db-01",
                lesson_version="v2",
                rubric_id="rubric-db-01-v2",
                rubric_status="approved",
                rubric=copy.deepcopy(acid_rubric),
                questions=[
                    LessonQuestionRecord(
                        question_id="q-db-01-v2-knowing",
                        lesson_id="lesson-db-01",
                        lesson_version="v2",
                        assessment_level="knowing",
                        format="essay",
                        question_text="Compare Read Committed, Repeatable Read, and Serializable isolation levels.",
                        approved=True,
                        answer_key="ANSWER_KEY_CONFIDENTIAL_V2: Dirty, non-repeatable, and phantom read prevention.",
                    )
                ],
                source_mode="mock",
            ),
            ("lesson-se-02", "v1"): LessonQuestionsRubricBundle(
                lesson_id="lesson-se-02",
                lesson_version="v1",
                rubric_id="rubric-se-02-v1",
                rubric_status="approved",
                rubric={
                    "max_score": 10.0,
                    "grounded": True,
                    "status": "approved",
                    "pass_rule": {"all_required_met": True, "min_percentage": 60.0},
                    "criteria": [
                        {
                            "criterion_id": "C1",
                            "name": "Explicit Boundaries",
                            "description": "Explains separation between orchestration, tools, and storage.",
                            "max_points": 10.0,
                            "required": True,
                            "key_points": ["ANSWER_KEY_CONFIDENTIAL_SE: Mention contract interfaces and mocks"],
                            "source_chunks": ["lesson-se-02:v1:c1"],
                            "levels": [
                                {"label": "Full", "points": 10.0, "descriptor": "Clear separation and rationale."},
                                {"label": "Partial", "points": 5.0, "descriptor": "Partial explanation."},
                                {"label": "Missing", "points": 0.0, "descriptor": "Absent."},
                            ],
                        }
                    ],
                },
                questions=[
                    LessonQuestionRecord(
                        question_id="q-se-02-knowing",
                        lesson_id="lesson-se-02",
                        lesson_version="v1",
                        assessment_level="knowing",
                        format="essay",
                        question_text="Why should an AI agent separate LCMS tool interfaces from orchestration logic?",
                        approved=True,
                        answer_key="ANSWER_KEY_CONFIDENTIAL_SE: Swap mocks safely and enforce scope.",
                    )
                ],
                source_mode="mock",
            ),
        }

    def set_classroom_exam(self, student_id: str, active: bool) -> None:
        with self._lock:
            if active:
                self._classroom_exam_students.add(student_id)
            else:
                self._classroom_exam_students.discard(student_id)

    def list_records(self, record_type: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [
                copy.deepcopy(r)
                for r in self._records
                if record_type is None or r["record_type"] == record_type
            ]
            return items

    async def fetch_authorized_lesson(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonMaterialRecord:
        with self._lock:
            record = self._lessons.get((lesson_id, lesson_version))
        if record is None:
            raise AppError(
                ErrorCode.LESSON_NOT_FOUND,
                f"Lesson '{lesson_id}' version '{lesson_version}' was not found in LCMS.",
            )
        if not record.approved:
            raise AppError(
                ErrorCode.UNAPPROVED_KNOWLEDGE,
                f"Lesson '{lesson_id}' version '{lesson_version}' is not teacher-approved.",
            )
        return sanitize_lesson_for_visibility(record, visibility_scope)

    async def fetch_approved_questions_rubric(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        rubric_id: Optional[str] = None,
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonQuestionsRubricBundle:
        with self._lock:
            lesson = self._lessons.get((lesson_id, lesson_version))
            bundle = self._bundles.get((lesson_id, lesson_version))
        if lesson is None or bundle is None:
            raise AppError(
                ErrorCode.LESSON_NOT_FOUND,
                f"Approved assessment materials for '{lesson_id}' version '{lesson_version}' were not found.",
            )
        if not lesson.approved or bundle.rubric_status != "approved":
            raise AppError(
                ErrorCode.UNAPPROVED_KNOWLEDGE,
                f"Assessment materials for '{lesson_id}' version '{lesson_version}' are not approved.",
            )
        if rubric_id and bundle.rubric_id != rubric_id:
            raise AppError(
                ErrorCode.LESSON_NOT_FOUND,
                f"Rubric '{rubric_id}' does not match lesson '{lesson_id}'.",
            )
        return sanitize_bundle_for_visibility(bundle, visibility_scope)

    async def fetch_assessment_context(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        student_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        requested_mode: AssessmentMode = "home_understanding",
        request_id: str = "",
    ) -> AssessmentContextRecord:
        with self._lock:
            lesson = self._lessons.get((lesson_id, lesson_version))
            in_exam = bool(student_id and student_id in self._classroom_exam_students)
        if lesson is None:
            raise AppError(
                ErrorCode.LESSON_NOT_FOUND,
                f"Lesson '{lesson_id}' version '{lesson_version}' was not found in LCMS.",
            )
        if not lesson.approved:
            raise AppError(
                ErrorCode.UNAPPROVED_KNOWLEDGE,
                f"Lesson '{lesson_id}' version '{lesson_version}' is not teacher-approved.",
            )
        if in_exam or requested_mode == "classroom_exam":
            return AssessmentContextRecord(
                lesson_id=lesson_id,
                lesson_version=lesson_version,
                course_id=lesson.course_id,
                student_id=student_id,
                attempt_id=attempt_id,
                permitted_mode="classroom_exam",
                classroom_exam_active=True,
                max_turns=get_settings().agent_max_dialogue_turns,
                allowed=False,
                denial_reason="Student is currently in a controlled classroom examination.",
                source_mode="mock",
            )
        return AssessmentContextRecord(
            lesson_id=lesson_id,
            lesson_version=lesson_version,
            course_id=lesson.course_id,
            assignment_id=f"assign-{lesson_id}-{lesson_version}",
            student_id=student_id,
            attempt_id=attempt_id,
            permitted_mode=requested_mode,
            classroom_exam_active=False,
            max_turns=get_settings().agent_max_dialogue_turns,
            allowed=True,
            source_mode="mock",
        )

    def _record_idempotent(
        self,
        record_type: Literal["question_draft", "grading_suggestion", "dialogue_outcome"],
        payload: BaseModel,
        idempotency_key: str,
    ) -> DraftRecordReceipt:
        composite = f"{record_type}:{idempotency_key}"
        fp = _payload_hash(payload)
        with self._lock:
            existing = self._idempotency.get(composite)
            if existing is not None:
                prev_fp, prev_receipt = existing
                if prev_fp != fp:
                    raise AppError(
                        ErrorCode.IDEMPOTENCY_KEY_CONFLICT,
                        f"Idempotency-Key '{idempotency_key}' was reused for a different {record_type} payload.",
                    )
                return prev_receipt.model_copy(update={"idempotent_replay": True})

            receipt = DraftRecordReceipt(
                record_id=f"{record_type[:3]}-{uuid.uuid4().hex[:12]}",
                record_type=record_type,
                publication_status="draft_suggestion",
                released=False,
                teacher_review_required=True,
                idempotent_replay=False,
                source_mode="mock",
                recorded_at=_utc_now_iso(),
            )
            self._idempotency[composite] = (fp, receipt)
            self._records.append(
                {
                    "record_id": receipt.record_id,
                    "record_type": record_type,
                    "payload": payload.model_dump(mode="json"),
                    "receipt": receipt.model_dump(mode="json"),
                }
            )
            return receipt

    async def record_question_draft(
        self,
        payload: QuestionDraftRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return self._record_idempotent("question_draft", payload, payload.idempotency_key)

    async def record_grading_suggestion(
        self,
        payload: GradingSuggestionRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return self._record_idempotent("grading_suggestion", payload, payload.idempotency_key)

    async def record_dialogue_outcome(
        self,
        payload: DialogueOutcomeRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return self._record_idempotent("dialogue_outcome", payload, payload.idempotency_key)


_CAMEL_TO_SNAKE_KEYS: dict[str, str] = {
    "allRequiredMet": "all_required_met",
    "answerKey": "answer_key",
    "approvedBy": "approved_by",
    "assessmentLevel": "assessment_level",
    "assignmentId": "assignment_id",
    "attemptId": "attempt_id",
    "classroomExamActive": "classroom_exam_active",
    "courseId": "course_id",
    "criterionId": "criterion_id",
    "denialReason": "denial_reason",
    "documentId": "document_id",
    "idempotentReplay": "idempotent_replay",
    "keyPoints": "key_points",
    "lessonId": "lesson_id",
    "lessonVersion": "lesson_version",
    "maxPoints": "max_points",
    "maxScore": "max_score",
    "maxTurns": "max_turns",
    "minPercentage": "min_percentage",
    "passRule": "pass_rule",
    "permittedMode": "permitted_mode",
    "publicationStatus": "publication_status",
    "questionId": "question_id",
    "questionText": "question_text",
    "recordId": "record_id",
    "recordType": "record_type",
    "recordedAt": "recorded_at",
    "rubricId": "rubric_id",
    "rubricStatus": "rubric_status",
    "sourceChunks": "source_chunks",
    "sourceMode": "source_mode",
    "studentId": "student_id",
    "studentSafeContent": "student_safe_content",
    "teacherOnlyNotes": "teacher_only_notes",
    "teacherReviewRequired": "teacher_review_required",
}


def _normalize_lcms_keys(obj: Any) -> Any:
    """Recursively map documented C# ASP.NET camelCase JSON keys to snake_case."""
    if isinstance(obj, list):
        return [_normalize_lcms_keys(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _CAMEL_TO_SNAKE_KEYS.get(str(k), str(k)): _normalize_lcms_keys(v)
            for k, v in obj.items()
        }
    return obj


class HttpLcmsToolClient(LCMSToolClient):
    """HTTP adapter calling the institutional C# LCMS internal agent endpoints."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        service_secret: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.agent_lcms_base_url).rstrip("/")
        self._timeout_s = timeout_s or settings.agent_lcms_timeout_s
        self._service_secret = (
            service_secret if service_secret is not None else settings.agent_service_secret
        ).strip()
        self._client = client

    @property
    def source_mode(self) -> Literal["mock", "http"]:
        return "http"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=min(self._timeout_s, 10.0)),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _headers(self, request_id: str, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if request_id:
            headers["X-Request-ID"] = request_id
        if self._service_secret:
            headers["X-Agent-Service-Key"] = self._service_secret
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _raise_for_http_error(self, resp: httpx.Response) -> None:
        code_from_body = ""
        message_from_body = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                err = body.get("error") if isinstance(body.get("error"), dict) else body
                code_from_body = str(err.get("code") or body.get("code") or "")
                message_from_body = str(
                    err.get("message")
                    or err.get("detail")
                    or err.get("title")
                    or body.get("message")
                    or body.get("detail")
                    or body.get("title")
                    or ""
                )
        except ValueError:
            pass

        msg = message_from_body or f"LCMS tool request failed with HTTP {resp.status_code}."
        if resp.status_code == 401:
            raise AppError(ErrorCode.AGENT_UNAUTHORIZED, msg)
        if resp.status_code == 403:
            if code_from_body == ErrorCode.ASSESSMENT_MODE_FORBIDDEN:
                raise AppError(ErrorCode.ASSESSMENT_MODE_FORBIDDEN, msg)
            raise AppError(ErrorCode.AGENT_SCOPE_FORBIDDEN, msg)
        if resp.status_code == 404:
            raise AppError(ErrorCode.LESSON_NOT_FOUND, msg)
        if resp.status_code == 409:
            raise AppError(ErrorCode.IDEMPOTENCY_KEY_CONFLICT, msg)
        if resp.status_code == 422:
            if "UNAPPROVED" in code_from_body.upper():
                raise AppError(ErrorCode.UNAPPROVED_KNOWLEDGE, msg)
            raise AppError(ErrorCode.VALIDATION_ERROR, msg, status_code=422)
        raise AppError(ErrorCode.LCMS_TOOL_ERROR, msg)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        request_id: str = "",
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        try:
            resp = await client.request(
                method,
                path,
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=json_body,
                headers=self._headers(request_id, idempotency_key),
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as exc:
            raise AppError(ErrorCode.LCMS_UNAVAILABLE, "Institutional LCMS tool service is unavailable.") from exc

        if resp.status_code >= 400:
            self._raise_for_http_error(resp)

        try:
            data = resp.json()
        except ValueError as exc:
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned an unexpected response shape.")
        return data

    async def fetch_authorized_lesson(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonMaterialRecord:
        data = await self._request_json(
            "GET",
            f"/api/internal/agent/lessons/{lesson_id}",
            params={"version": lesson_version, "visibility": visibility_scope},
            request_id=request_id,
        )
        normalized = _normalize_lcms_keys(data)
        try:
            record = LessonMaterialRecord.model_validate({**normalized, "source_mode": "http"})
        except ValidationError as exc:
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned a malformed lesson payload.") from exc
        if not record.approved:
            raise AppError(
                ErrorCode.UNAPPROVED_KNOWLEDGE,
                f"Lesson '{lesson_id}' version '{lesson_version}' is not teacher-approved.",
            )
        return sanitize_lesson_for_visibility(record, visibility_scope)

    async def fetch_approved_questions_rubric(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        rubric_id: Optional[str] = None,
        visibility_scope: VisibilityScope = "student_safe",
        request_id: str = "",
    ) -> LessonQuestionsRubricBundle:
        data = await self._request_json(
            "GET",
            f"/api/internal/agent/lessons/{lesson_id}/assessment-materials",
            params={"version": lesson_version, "rubric_id": rubric_id, "visibility": visibility_scope},
            request_id=request_id,
        )
        normalized = _normalize_lcms_keys(data)
        try:
            bundle = LessonQuestionsRubricBundle.model_validate({**normalized, "source_mode": "http"})
        except ValidationError as exc:
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned a malformed rubric/question bundle.") from exc
        if bundle.rubric_status != "approved":
            raise AppError(ErrorCode.UNAPPROVED_KNOWLEDGE, "Lesson rubric is not teacher-approved.")
        return sanitize_bundle_for_visibility(bundle, visibility_scope)

    async def fetch_assessment_context(
        self,
        *,
        lesson_id: str,
        lesson_version: str = "v1",
        student_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        requested_mode: AssessmentMode = "home_understanding",
        request_id: str = "",
    ) -> AssessmentContextRecord:
        data = await self._request_json(
            "POST",
            "/api/internal/agent/assessment-context",
            json_body={
                "lesson_id": lesson_id,
                "lesson_version": lesson_version,
                "student_id": student_id,
                "attempt_id": attempt_id,
                "requested_mode": requested_mode,
            },
            request_id=request_id,
        )
        normalized = _normalize_lcms_keys(data)
        try:
            return AssessmentContextRecord.model_validate({**normalized, "source_mode": "http"})
        except ValidationError as exc:
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned a malformed assessment context.") from exc

    async def _post_draft(
        self,
        path: str,
        record_type: Literal["question_draft", "grading_suggestion", "dialogue_outcome"],
        payload: BaseModel,
        idempotency_key: str,
        request_id: str,
    ) -> DraftRecordReceipt:
        data = await self._request_json(
            "POST",
            path,
            json_body=payload.model_dump(mode="json"),
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        normalized = _normalize_lcms_keys(data)
        try:
            receipt = DraftRecordReceipt.model_validate(
                {
                    **normalized,
                    "record_type": record_type,
                    "publication_status": "draft_suggestion",
                    "released": False,
                    "teacher_review_required": True,
                    "source_mode": "http",
                }
            )
            return receipt
        except ValidationError as exc:
            raise AppError(ErrorCode.LCMS_TOOL_ERROR, "LCMS returned a malformed draft receipt.") from exc

    async def record_question_draft(
        self,
        payload: QuestionDraftRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return await self._post_draft(
            "/api/internal/agent/drafts/questions",
            "question_draft",
            payload,
            payload.idempotency_key,
            request_id,
        )

    async def record_grading_suggestion(
        self,
        payload: GradingSuggestionRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return await self._post_draft(
            "/api/internal/agent/suggestions/grading",
            "grading_suggestion",
            payload,
            payload.idempotency_key,
            request_id,
        )

    async def record_dialogue_outcome(
        self,
        payload: DialogueOutcomeRecordInput,
        *,
        request_id: str = "",
    ) -> DraftRecordReceipt:
        return await self._post_draft(
            "/api/internal/agent/outcomes/dialogue",
            "dialogue_outcome",
            payload,
            payload.idempotency_key,
            request_id,
        )


_mock_lcms_client: Optional[MockLcmsToolClient] = None
_http_lcms_client: Optional[HttpLcmsToolClient] = None
_http_lcms_client_cfg: Optional[tuple[str, float, str]] = None


def get_mock_lcms_client() -> MockLcmsToolClient:
    global _mock_lcms_client
    if _mock_lcms_client is None:
        _mock_lcms_client = MockLcmsToolClient()
    return _mock_lcms_client


def get_lcms_tool_client() -> LCMSToolClient:
    """Resolve the active LCMS tool client from AGENT_LCMS_MODE (mock | http)."""
    global _http_lcms_client, _http_lcms_client_cfg
    settings = get_settings()
    if settings.agent_lcms_mode == "mock":
        return get_mock_lcms_client()

    cfg = (
        settings.agent_lcms_base_url.rstrip("/"),
        float(settings.agent_lcms_timeout_s),
        settings.agent_service_secret.strip(),
    )
    if (
        _http_lcms_client is None
        or _http_lcms_client_cfg != cfg
        or (_http_lcms_client._client is not None and _http_lcms_client._client.is_closed)
    ):
        _http_lcms_client = HttpLcmsToolClient(
            base_url=cfg[0],
            timeout_s=cfg[1],
            service_secret=cfg[2],
        )
        _http_lcms_client_cfg = cfg
    return _http_lcms_client


async def close_lcms_tool_client() -> None:
    """Close the cached HTTP LCMS client cleanly on application shutdown."""
    global _http_lcms_client, _http_lcms_client_cfg
    client = _http_lcms_client
    _http_lcms_client = None
    _http_lcms_client_cfg = None
    if client is not None:
        await client.close()
