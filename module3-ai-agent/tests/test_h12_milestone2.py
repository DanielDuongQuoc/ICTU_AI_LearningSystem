"""H12 AI Agent Milestone 2 tests: authorized LCMS tools, versioned knowledge & isolated sessions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import AppError, ErrorCode
from app.h12_agent.knowledge import LessonKnowledgeService
from app.h12_agent.lcms_tools import (
    DialogueOutcomeRecordInput,
    GradingSuggestionRecordInput,
    HttpLcmsToolClient,
    MockLcmsToolClient,
    QuestionDraftRecordInput,
)
from app.h12_agent.policy import ROLE_POLICIES
from app.h12_agent.session_store import SessionMemoryStore
from app.h12_agent.tool_executor import AuthorizedToolExecutor
from app.h12_agent.types import EvidenceRef, InstitutionalRef
from tests.fakes import Stack


@pytest.fixture()
def mock_lcms():
    client = MockLcmsToolClient()
    yield client
    client.reset()


@pytest.fixture()
def mem_store(tmp_path):
    store = SessionMemoryStore(str(tmp_path / "test_sessions.db"))
    yield store
    store.close()


# ------------------------------------------------------------------ Authorized LCMS Tools


@pytest.mark.asyncio
async def test_mock_lcms_fetches_versions_and_enforces_approval_and_not_found(mock_lcms):
    v1 = await mock_lcms.fetch_authorized_lesson(
        lesson_id="lesson-db-01",
        lesson_version="v1",
        visibility_scope="teacher_authorized",
    )
    v2 = await mock_lcms.fetch_authorized_lesson(
        lesson_id="lesson-db-01",
        lesson_version="v2",
        visibility_scope="teacher_authorized",
    )
    assert "ACID properties" in v1.student_safe_content
    assert "Isolation levels" in v2.student_safe_content or "isolation levels" in v2.student_safe_content
    assert v1.student_safe_content != v2.student_safe_content

    with pytest.raises(AppError) as missing_exc:
        await mock_lcms.fetch_authorized_lesson(lesson_id="lesson-does-not-exist", lesson_version="v1")
    assert missing_exc.value.code == ErrorCode.LESSON_NOT_FOUND

    with pytest.raises(AppError) as draft_exc:
        await mock_lcms.fetch_authorized_lesson(lesson_id="lesson-draft-99", lesson_version="v1")
    assert draft_exc.value.code == ErrorCode.UNAPPROVED_KNOWLEDGE


@pytest.mark.asyncio
async def test_tool_executor_strips_answer_keys_for_student_safe_dialogue(mock_lcms):
    student_exec = AuthorizedToolExecutor(
        lcms_client=mock_lcms,
        policy=ROLE_POLICIES["dialogue_assessment"],
        assessment_mode="home_understanding",
        caller_scope="student",
        request_id="req-s-1",
    )
    lesson = await student_exec.fetch_authorized_lesson(lesson_id="lesson-db-01", lesson_version="v1")
    bundle = await student_exec.fetch_approved_questions_rubric(lesson_id="lesson-db-01", lesson_version="v1")

    assert lesson.teacher_only_notes is None
    assert "ANSWER_KEY_CONFIDENTIAL_ACID" not in lesson.model_dump_json()
    assert all(q.answer_key is None for q in bundle.questions)
    assert all("key_points" not in c for c in bundle.rubric["criteria"])
    assert "ANSWER_KEY_CONFIDENTIAL_ACID" not in bundle.model_dump_json()

    teacher_exec = AuthorizedToolExecutor(
        lcms_client=mock_lcms,
        policy=ROLE_POLICIES["grading_support"],
        assessment_mode="home_knowing",
        caller_scope="teacher",
        request_id="req-t-1",
    )
    t_lesson = await teacher_exec.fetch_authorized_lesson(lesson_id="lesson-db-01", lesson_version="v1")
    t_bundle = await teacher_exec.fetch_approved_questions_rubric(lesson_id="lesson-db-01", lesson_version="v1")
    assert t_lesson.teacher_only_notes and "ANSWER_KEY_CONFIDENTIAL_ACID" in t_lesson.teacher_only_notes
    assert any(q.answer_key and "ANSWER_KEY_CONFIDENTIAL_ACID" in q.answer_key for q in t_bundle.questions)
    assert any("key_points" in c for c in t_bundle.rubric["criteria"])


@pytest.mark.asyncio
async def test_tool_executor_denies_out_of_scope_tools_and_lcms_classroom_exam(mock_lcms):
    student_exec = AuthorizedToolExecutor(
        lcms_client=mock_lcms,
        policy=ROLE_POLICIES["dialogue_assessment"],
        assessment_mode="home_understanding",
        caller_scope="student",
        request_id="req-deny-1",
    )
    ref = InstitutionalRef(lesson_id="lesson-db-01", lesson_version="v1", student_id="SV001", attempt_id="att-1")

    with pytest.raises(AppError) as scope_exc:
        await student_exec.record_question_draft(
            QuestionDraftRecordInput(
                institutional_ref=ref,
                question_text="Draft?",
                expected_answer="Answer",
                adapter_mode="mock",
                idempotency_key="idem-out-of-scope",
            )
        )
    assert scope_exc.value.code == ErrorCode.AGENT_SCOPE_FORBIDDEN
    assert student_exec.outcomes[-1].outcome == "denied"

    with pytest.raises(AppError) as grade_exc:
        await student_exec.record_grading_suggestion(
            GradingSuggestionRecordInput(
                institutional_ref=ref,
                suggested_total=8.0,
                max_score=10.0,
                feedback="Good",
                adapter_mode="mock",
                idempotency_key="idem-out-of-scope-2",
            )
        )
    assert grade_exc.value.code == ErrorCode.AGENT_SCOPE_FORBIDDEN

    with pytest.raises(AppError) as exam_exc:
        await student_exec.fetch_assessment_context(
            lesson_id="lesson-db-01",
            lesson_version="v1",
            student_id="SV_IN_EXAM_01",
            attempt_id="att-1",
        )
    assert exam_exc.value.code == ErrorCode.ASSESSMENT_MODE_FORBIDDEN
    assert student_exec.outcomes[-1].outcome == "denied"


@pytest.mark.asyncio
async def test_lcms_record_tools_enforce_draft_boundary_and_idempotency(mock_lcms):
    teacher_exec = AuthorizedToolExecutor(
        lcms_client=mock_lcms,
        policy=ROLE_POLICIES["question_drafting"],
        assessment_mode="teacher_question_drafting",
        caller_scope="teacher",
        request_id="req-draft-1",
    )
    ref = InstitutionalRef(lesson_id="lesson-db-01", lesson_version="v1")
    draft_in = QuestionDraftRecordInput(
        institutional_ref=ref,
        question_text="Explain Atomicity and Durability in ACID transactions.",
        expected_answer="Atomicity ensures rollback on failure; Durability persists committed changes via WAL.",
        evidence_refs=[EvidenceRef(chunk_id="c1", document_id="doc1", page=1, score=0.9)],
        adapter_mode="mock",
        idempotency_key="draft-idem-1",
    )
    r1 = await teacher_exec.record_question_draft(draft_in)
    assert r1.publication_status == "draft_suggestion"
    assert r1.released is False
    assert r1.teacher_review_required is True
    assert r1.idempotent_replay is False

    r2 = await teacher_exec.record_question_draft(draft_in)
    assert r2.idempotent_replay is True
    assert r2.record_id == r1.record_id

    mutated = draft_in.model_copy(update={"question_text": "Different essay question?"})
    with pytest.raises(AppError) as conflict_exc:
        await teacher_exec.record_question_draft(mutated)
    assert conflict_exc.value.code == ErrorCode.IDEMPOTENCY_KEY_CONFLICT

    dialogue_exec = AuthorizedToolExecutor(
        lcms_client=mock_lcms,
        policy=ROLE_POLICIES["dialogue_assessment"],
        assessment_mode="home_understanding",
        caller_scope="student",
        request_id="req-dial-1",
    )
    r_dial = await dialogue_exec.record_dialogue_outcome(
        DialogueOutcomeRecordInput(
            institutional_ref=ref.model_copy(update={"student_id": "SV001", "attempt_id": "att-1"}),
            session_id="sess-abc",
            stopping_reason="criteria_covered",
            turn_count=3,
            summary_feedback="Student explained Atomicity and Isolation clearly.",
            adapter_mode="mock",
            idempotency_key="dial-idem-1",
        )
    )
    assert r_dial.publication_status == "draft_suggestion"
    assert r_dial.released is False


# ------------------------------------------------------------------ HttpLcmsToolClient


@pytest.mark.asyncio
async def test_http_lcms_client_sends_headers_sanitizes_and_maps_errors():
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        if request.url.path == "/api/internal/agent/lessons/lesson-db-01":
            return httpx.Response(
                200,
                json={
                    "lesson_id": "lesson-db-01",
                    "lesson_version": "v1",
                    "course_id": "CS301",
                    "title": "Transactions",
                    "approved": True,
                    "student_safe_content": "Safe text.",
                    "teacher_only_notes": "SECRET_HTTP_ANSWER_KEY",
                },
            )
        if request.url.path == "/api/internal/agent/lessons/missing":
            return httpx.Response(404, json={"error": {"code": "LESSON_NOT_FOUND", "message": "Not found"}})
        if request.url.path == "/api/internal/agent/drafts/questions":
            return httpx.Response(409, json={"error": {"code": "IDEMPOTENCY_KEY_CONFLICT", "message": "Conflict"}})
        return httpx.Response(500, text="upstream broken")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://lcms.test",
    ) as async_client:
        http_client = HttpLcmsToolClient(
            base_url="http://lcms.test",
            service_secret="test-lcms-secret-123",
            client=async_client,
        )
        lesson = await http_client.fetch_authorized_lesson(
            lesson_id="lesson-db-01",
            visibility_scope="student_safe",
            request_id="req-http-1",
        )
        assert lesson.source_mode == "http"
        assert lesson.teacher_only_notes is None
        assert seen_headers.get("x-agent-service-key") == "test-lcms-secret-123"
        assert seen_headers.get("x-request-id") == "req-http-1"

        with pytest.raises(AppError) as not_found_exc:
            await http_client.fetch_authorized_lesson(lesson_id="missing")
        assert not_found_exc.value.code == ErrorCode.LESSON_NOT_FOUND

        with pytest.raises(AppError) as conflict_exc:
            await http_client.record_question_draft(
                QuestionDraftRecordInput(
                    institutional_ref=InstitutionalRef(lesson_id="lesson-db-01"),
                    question_text="Q?",
                    expected_answer="A",
                    adapter_mode="mock",
                    idempotency_key="http-idem-1",
                )
            )
        assert conflict_exc.value.code == ErrorCode.IDEMPOTENCY_KEY_CONFLICT
        assert seen_headers.get("idempotency-key") == "http-idem-1"

        with pytest.raises(AppError) as err_502:
            await http_client.fetch_approved_questions_rubric(lesson_id="bad")
        assert err_502.value.code == ErrorCode.LCMS_TOOL_ERROR

    def conn_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(conn_fail),
        base_url="http://lcms.test",
    ) as down_client:
        down_lcms = HttpLcmsToolClient(base_url="http://lcms.test", client=down_client)
        with pytest.raises(AppError) as unavail_exc:
            await down_lcms.fetch_authorized_lesson(lesson_id="lesson-db-01")
        assert unavail_exc.value.code == ErrorCode.LCMS_UNAVAILABLE


# ------------------------------------------------------------------ Versioned Lesson Knowledge + ChromaDB


@pytest.mark.asyncio
async def test_knowledge_versioning_and_answer_key_isolation_via_api(client, monkeypatch, mock_lcms, mem_store):
    stack = Stack(threshold=0.0).install(monkeypatch)
    knowledge = LessonKnowledgeService(
        embedding_service=stack.embedding,
        chroma_repo=stack.repo,
        vector_service=stack.vector,
        similarity_threshold=0.0,
    )
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: mock_lcms)
    monkeypatch.setattr("app.api.agent_h12._get_knowledge_service", lambda: knowledge)
    monkeypatch.setattr("app.api.agent_h12._get_session_store", lambda: mem_store)

    # Index v1 directly with confidential teacher answer-key canary
    r_v1 = await client.post(
        "/api/agent/knowledge/index",
        json={
            "caller_scope": "teacher",
            "lesson_id": "lesson-db-01",
            "knowledge_version": "v1",
            "title": "ACID Basics",
            "student_safe_content": "Atomicity rolls back failed database transactions completely.",
            "teacher_only_notes": "SECRET_ANSWER_KEY_TOKEN_991 requires mentioning write-ahead undo log.",
            "approved": True,
        },
    )
    assert r_v1.status_code == 200, r_v1.text
    assert r_v1.json()["student_safe_chunks"] >= 1
    assert r_v1.json()["teacher_only_chunks"] >= 1

    # Index v2 pulled from Mock LCMS tool
    r_v2 = await client.post(
        "/api/agent/knowledge/index",
        json={
            "caller_scope": "teacher",
            "lesson_id": "lesson-db-01",
            "knowledge_version": "v2",
            "from_lcms": True,
        },
    )
    assert r_v2.status_code == 200, r_v2.text
    assert r_v2.json()["source_mode"] == "mock"

    versions = knowledge.list_versions("lesson-db-01")
    assert [v["knowledge_version"] for v in versions] == ["v1", "v2"]

    # Student queries using exact canary phrase from teacher_only_notes -> MUST NOT leak
    r_student_canary = await client.post(
        "/api/agent/knowledge/retrieve",
        json={
            "agent_role": "dialogue_assessment",
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV001",
                "attempt_id": "att-1",
            },
            "query": "SECRET_ANSWER_KEY_TOKEN_991 write-ahead undo log",
        },
    )
    assert r_student_canary.status_code == 200
    body_s = r_student_canary.json()
    assert body_s["visibility_scope"] == "student_safe"
    assert "SECRET_ANSWER_KEY_TOKEN_991" not in json.dumps(body_s)
    assert all(c["visibility"] == "student_safe" for c in body_s["chunks"])

    # Teacher in grading_support scope CAN retrieve teacher_only answer-key notes
    r_teacher_canary = await client.post(
        "/api/agent/knowledge/retrieve",
        json={
            "agent_role": "grading_support",
            "caller_scope": "teacher",
            "assessment_mode": "home_knowing",
            "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1"},
            "query": "SECRET_ANSWER_KEY_TOKEN_991 write-ahead undo log",
        },
    )
    assert r_teacher_canary.status_code == 200
    assert "SECRET_ANSWER_KEY_TOKEN_991" in r_teacher_canary.json()["context"]

    # Version isolation: querying v1 must not return v2's Serializable content
    r_v1_query = await client.post(
        "/api/agent/knowledge/retrieve",
        json={
            "agent_role": "dialogue_assessment",
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV001",
                "attempt_id": "att-1",
            },
            "query": "Atomicity rollback transactions",
        },
    )
    assert r_v1_query.status_code == 200
    assert all(c["knowledge_version"] == "v1" for c in r_v1_query.json()["chunks"])
    assert "Serializable" not in r_v1_query.json()["context"]


@pytest.mark.asyncio
async def test_knowledge_index_and_retrieve_reject_unapproved_student_and_exam_mode(
    client, monkeypatch, mock_lcms, mem_store
):
    stack = Stack(threshold=0.0).install(monkeypatch)
    knowledge = LessonKnowledgeService(
        embedding_service=stack.embedding,
        chroma_repo=stack.repo,
        vector_service=stack.vector,
    )
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: mock_lcms)
    monkeypatch.setattr("app.api.agent_h12._get_knowledge_service", lambda: knowledge)
    monkeypatch.setattr("app.api.agent_h12._get_session_store", lambda: mem_store)

    r_student = await client.post(
        "/api/agent/knowledge/index",
        json={
            "caller_scope": "student",
            "lesson_id": "lesson-db-01",
            "knowledge_version": "v1",
            "student_safe_content": "Attempted poisoning",
        },
    )
    assert r_student.status_code == 403
    assert r_student.json()["error"]["code"] == ErrorCode.AGENT_SCOPE_FORBIDDEN

    r_unapproved = await client.post(
        "/api/agent/knowledge/index",
        json={
            "caller_scope": "teacher",
            "lesson_id": "lesson-draft-99",
            "knowledge_version": "v1",
            "student_safe_content": "Not approved yet",
            "approved": False,
        },
    )
    assert r_unapproved.status_code == 422
    assert r_unapproved.json()["error"]["code"] == ErrorCode.UNAPPROVED_KNOWLEDGE

    r_exam = await client.post(
        "/api/agent/knowledge/retrieve",
        json={
            "agent_role": "dialogue_assessment",
            "caller_scope": "student",
            "assessment_mode": "classroom_exam",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV001",
                "attempt_id": "att-1",
            },
            "query": "ACID",
        },
    )
    assert r_exam.status_code == 403
    assert r_exam.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN


# ------------------------------------------------------------------ Isolated Session Memory (SQLite & API)


@pytest.mark.asyncio
async def test_sessions_api_creates_isolated_sessions_and_blocks_classroom_exam(
    client, monkeypatch, mock_lcms, mem_store
):
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: mock_lcms)
    monkeypatch.setattr("app.api.agent_h12._get_session_store", lambda: mem_store)

    r_s1 = await client.post(
        "/api/agent/sessions",
        json={
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV001",
                "attempt_id": "att-1",
            },
        },
    )
    assert r_s1.status_code == 200, r_s1.text
    s1_id = r_s1.json()["session"]["session_id"]
    assert r_s1.json()["created"] is True
    assert r_s1.json()["tool_outcomes"][0]["tool_name"] == "fetch_assessment_context"
    assert r_s1.json()["tool_outcomes"][0]["outcome"] == "ok"

    # Resuming same composite key returns existing session
    r_s1_resume = await client.post(
        "/api/agent/sessions",
        json={
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV001",
                "attempt_id": "att-1",
            },
        },
    )
    assert r_s1_resume.status_code == 200
    assert r_s1_resume.json()["created"] is False
    assert r_s1_resume.json()["session"]["session_id"] == s1_id

    # Different student or different attempt gets a distinct isolated session
    r_s2 = await client.post(
        "/api/agent/sessions",
        json={
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV002",
                "attempt_id": "att-1",
            },
        },
    )
    assert r_s2.status_code == 200
    assert r_s2.json()["session"]["session_id"] != s1_id

    # Cross-student read via GET /api/agent/sessions/{session_id} is forbidden
    r_cross = await client.get(f"/api/agent/sessions/{s1_id}", params={"student_id": "SV002"})
    assert r_cross.status_code == 403
    assert r_cross.json()["error"]["code"] == ErrorCode.AGENT_SCOPE_FORBIDDEN

    # Wrong lesson or attempt for owner is also forbidden
    r_wrong_attempt = await client.get(
        f"/api/agent/sessions/{s1_id}",
        params={"student_id": "SV001", "attempt_id": "att-2"},
    )
    assert r_wrong_attempt.status_code == 403

    # Owner reads cleanly and tool audit was persisted in SQLite
    r_own = await client.get(
        f"/api/agent/sessions/{s1_id}",
        params={"student_id": "SV001", "lesson_id": "lesson-db-01", "attempt_id": "att-1"},
    )
    assert r_own.status_code == 200
    assert len(mem_store.list_tool_audit(s1_id)) >= 1

    # Student currently in classroom exam per LCMS is denied even if requesting home_understanding
    r_in_exam = await client.post(
        "/api/agent/sessions",
        json={
            "caller_scope": "student",
            "assessment_mode": "home_understanding",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV_IN_EXAM_01",
                "attempt_id": "att-1",
            },
        },
    )
    assert r_in_exam.status_code == 403
    assert r_in_exam.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN


def test_session_store_cross_isolation_context_window_closure_and_retention(mem_store):
    s1, _ = mem_store.create_or_get_session(
        student_id="SV001",
        lesson_id="lesson-db-01",
        lesson_version="v1",
        attempt_id="att-1",
        assessment_mode="home_understanding",
        max_turns=6,
    )
    s2, _ = mem_store.create_or_get_session(
        student_id="SV002",
        lesson_id="lesson-db-01",
        lesson_version="v1",
        attempt_id="att-1",
        assessment_mode="home_understanding",
        max_turns=6,
    )
    mem_store.append_turn(s2.session_id, student_id="SV002", speaker="student", message_text="SV002_PRIVATE_TEXT")

    for i in range(1, 6):
        mem_store.append_turn(
            s1.session_id,
            student_id="SV001",
            speaker="student",
            message_text=f"SV001 student response {i}",
            evidence_refs=[EvidenceRef(chunk_id=f"chunk-{i}", document_id="doc-1", page=1, score=0.8)],
            window_turns=4,
        )
        mem_store.append_turn(
            s1.session_id,
            student_id="SV001",
            speaker="agent",
            message_text=f"Agent probing follow-up {i}",
            window_turns=4,
        )

    ctx = mem_store.build_model_context(s1.session_id, student_id="SV001", window_turns=4, max_chars=900)
    assert "SV002_PRIVATE_TEXT" not in ctx.rendered_context
    assert "SV002_PRIVATE_TEXT" not in ctx.context_summary
    assert len(ctx.recent_turns) == 4
    assert ctx.policy.total_turns == 10
    assert ctx.policy.included_turns == 4
    assert ctx.policy.summarized_turns == 6
    assert ctx.context_summary
    assert len(ctx.rendered_context) <= 900

    # Cross-student turn append fails
    with pytest.raises(AppError) as cross_append:
        mem_store.append_turn(s1.session_id, student_id="SV002", speaker="student", message_text="Hijack")
    assert cross_append.value.code == ErrorCode.AGENT_SCOPE_FORBIDDEN

    # Closing session prevents any subsequent turns
    closed = mem_store.close_session(s1.session_id, student_id="SV001", stopping_reason="criteria_covered")
    assert closed.status == "closed"
    with pytest.raises(AppError) as post_close:
        mem_store.append_turn(s1.session_id, student_id="SV001", speaker="student", message_text="Continue!")
    assert post_close.value.code == ErrorCode.SESSION_STATE_CONFLICT

    # Max turns limit enforcement on an active session
    s_short, _ = mem_store.create_or_get_session(
        student_id="SV001",
        lesson_id="lesson-db-01",
        lesson_version="v1",
        attempt_id="att-2",
        assessment_mode="home_understanding",
        max_turns=1,
    )
    mem_store.append_turn(s_short.session_id, student_id="SV001", speaker="student", message_text="Turn 1")
    with pytest.raises(AppError) as max_exc:
        mem_store.append_turn(s_short.session_id, student_id="SV001", speaker="student", message_text="Turn 2")
    assert max_exc.value.code == ErrorCode.SESSION_STATE_CONFLICT

    # Retention policy purge deletes expired sessions
    future = datetime.now(timezone.utc) + timedelta(days=31)
    assert mem_store.purge_expired(now_utc=future) == 3
    with pytest.raises(AppError) as gone_exc:
        mem_store.get_session_for_owner(s1.session_id, student_id="SV001")
    assert gone_exc.value.code == ErrorCode.SESSION_NOT_FOUND


# ------------------------------------------------------------------ Config validation


@pytest.mark.parametrize(
    "override,expected_msg",
    [
        ({"agent_lcms_timeout_s": 0}, "AGENT_LCMS_TIMEOUT_S"),
        ({"agent_session_db_path": "  "}, "AGENT_SESSION_DB_PATH"),
        ({"agent_session_retention_days": 0}, "AGENT_SESSION_RETENTION_DAYS"),
        ({"agent_context_window_turns": 0}, "AGENT_CONTEXT_WINDOW_TURNS"),
        ({"agent_session_max_context_chars": 100}, "AGENT_SESSION_MAX_CONTEXT_CHARS"),
    ],
)
def test_h12_milestone2_settings_validation_fails_fast(override, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        Settings(_env_file=None, **override)
