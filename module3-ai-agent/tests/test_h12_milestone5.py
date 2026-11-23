"""H12 AI Agent Milestone 5 tests: HTTP LCMS integration, ASP.NET camelCase bridge,
cross-session isolation, and classroom-mode denial (PI 3.3, PI 4.2, Week 8)."""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio

from app.core.config import Settings, get_settings
from app.core.exceptions import ErrorCode
from app.h12_agent.idempotency import get_idempotency_store
from app.h12_agent.knowledge import LessonKnowledgeService
from app.h12_agent.lcms_simulator import create_lcms_simulator_app
from app.h12_agent.lcms_tools import (
    HttpLcmsToolClient,
    MockLcmsToolClient,
    close_lcms_tool_client,
    get_lcms_tool_client,
)
from app.h12_agent.session_store import SessionMemoryStore
from tests.fakes import RecordingLLM, Stack


@pytest.fixture(autouse=True)
def _clean_idempotency():
    store = get_idempotency_store()
    store.clear()
    yield
    store.clear()


async def _build_bridge(
    monkeypatch,
    tmp_path,
    *,
    camel_case: bool = True,
    service_secret: str = "secret-test-key",
    client_secret: str | None = None,
):
    stack = Stack(threshold=0.0).install(monkeypatch)
    backend_mock = MockLcmsToolClient()
    sim_app = create_lcms_simulator_app(
        mock_client=backend_mock,
        service_secret=service_secret,
        camel_case=camel_case,
    )
    sim_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sim_app),
        base_url="http://lcms.test",
    )
    http_lcms = HttpLcmsToolClient(
        base_url="http://lcms.test",
        service_secret=service_secret if client_secret is None else client_secret,
        client=sim_http,
    )
    store = SessionMemoryStore(str(tmp_path / f"m5_{'camel' if camel_case else 'snake'}.db"))
    knowledge = LessonKnowledgeService(
        embedding_service=stack.embedding,
        chroma_repo=stack.repo,
        vector_service=stack.vector,
        similarity_threshold=0.0,
    )
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: http_lcms)
    monkeypatch.setattr("app.api.agent_h12._get_knowledge_service", lambda: knowledge)
    monkeypatch.setattr("app.api.agent_h12._get_session_store", lambda: store)
    return stack, backend_mock, sim_app, http_lcms, knowledge, store


@pytest_asyncio.fixture()
async def http_bridge(monkeypatch, tmp_path):
    res = await _build_bridge(monkeypatch, tmp_path, camel_case=True, service_secret="secret-test-key")
    _stack, backend_mock, _sim_app, http_lcms, _knowledge, store = res
    yield res
    await http_lcms.close()
    backend_mock.reset()
    store.close()


def _auth_headers(
    *,
    key: str = "secret-test-key",
    request_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    headers = {"X-Agent-Service-Key": key}
    if request_id:
        headers["X-Request-ID"] = request_id
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _start_understanding_payload(
    *,
    student_id: str = "SV001",
    attempt_id: str = "ATT_M5_01",
    lesson_id: str = "lesson-db-01",
    lesson_version: str = "v1",
    channel: str = "text_chat",
    assessment_mode: str = "home_understanding",
) -> dict:
    return {
        "caller_scope": "student",
        "assessment_mode": assessment_mode,
        "channel": channel,
        "institutional_ref": {
            "lesson_id": lesson_id,
            "lesson_version": lesson_version,
            "student_id": student_id,
            "attempt_id": attempt_id,
        },
        "language": "en",
    }


# ------------------------------------------------------------------ 1..3: E2E HTTP Workflows (snake & camelCase)


@pytest.mark.asyncio
@pytest.mark.parametrize("camel_case", [False, True])
async def test_e2e_http_question_drafting_forwards_headers_and_records_lcms_drafts(
    client, monkeypatch, tmp_path, camel_case
):
    _stack, backend_mock, sim_app, http_lcms, _knowledge, store = await _build_bridge(
        monkeypatch, tmp_path, camel_case=camel_case, service_secret="secret-test-key"
    )
    try:
        settings = get_settings()
        monkeypatch.setattr(settings, "agent_model_adapter", "mock")
        monkeypatch.setattr(settings, "agent_auth_required", True)
        monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")

        r = await client.post(
            "/api/agent/questions/draft",
            headers=_auth_headers(request_id="11111111-2222-4333-8444-555555555555", idempotency_key="m5-draft-01"),
            json={
                "caller_scope": "teacher",
                "assessment_mode": "teacher_question_drafting",
                "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v2"},
                "topic_focus": "SQL isolation levels and WAL recovery",
                "number_of_questions": 2,
                "difficulty": "hard",
                "question_format": "essay",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["publication_status"] == "draft_suggestion"
        assert body["released"] is False
        assert body["teacher_review_required"] is True
        assert len(body["drafts"]) == 2
        assert all(d["lcms_receipt"]["source_mode"] == "http" for d in body["drafts"])
        assert all(
            o["source_mode"] == "http"
            for o in body["tool_outcomes"]
            if o["tool_name"] in {"fetch_authorized_lesson", "record_question_draft"}
        )

        # Verify simulator captured auth header and request_id across HTTP hops.
        paths = [entry["path"] for entry in sim_app.state.request_log]
        assert "/api/internal/agent/lessons/lesson-db-01" in paths
        assert "/api/internal/agent/drafts/questions" in paths
        assert all(entry["has_service_key"] is True for entry in sim_app.state.request_log)
        assert all(
            entry["request_id"] == "11111111-2222-4333-8444-555555555555"
            for entry in sim_app.state.request_log
        )
        assert len(backend_mock.list_records("question_draft")) == 2
    finally:
        await http_lcms.close()
        backend_mock.reset()
        store.close()


@pytest.mark.asyncio
async def test_e2e_http_knowing_grading_enforces_no_silent_grade_publication_and_idempotency(
    client, monkeypatch, http_bridge
):
    _stack, backend_mock, sim_app, _http_lcms, _knowledge, _store = http_bridge
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_model_adapter", "mock")
    monkeypatch.setattr(settings, "agent_auth_required", True)
    monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")

    payload = {
        "caller_scope": "lcms_service",
        "assessment_mode": "home_knowing",
        "institutional_ref": {
            "lesson_id": "lesson-db-01",
            "lesson_version": "v1",
            "student_id": "SV001",
            "attempt_id": "ATT_KNOW_01",
        },
        "student_answer": (
            "A transaction is an atomic all-or-nothing unit supporting rollback on error. "
            "Consistency preserves constraints, isolation isolates concurrent reads, and WAL preserves durability."
        ),
        "question_format": "essay",
        "language": "en",
    }

    r1 = await client.post(
        "/api/agent/knowing/grade-suggestion",
        headers=_auth_headers(idempotency_key="m5-know-01"),
        json=payload,
    )
    assert r1.status_code == 200, r1.text
    assert r1.headers["X-Idempotent-Replay"] == "false"
    b1 = r1.json()
    assert b1["status"] == "pending_teacher_review"
    assert b1["teacher_handoff"] is True
    assert b1["teacher_review_required"] is True
    assert b1["publication_status"] == "draft_suggestion"
    assert b1["released"] is False
    assert b1["lcms_receipt"]["source_mode"] == "http"
    assert len(backend_mock.list_records("grading_suggestion")) == 1
    assert any(e["path"] == "/api/internal/agent/suggestions/grading" for e in sim_app.state.request_log)

    # Idempotent replay must not hit LCMS again or duplicate suggestion records.
    calls_before_replay = len(sim_app.state.request_log)
    r2 = await client.post(
        "/api/agent/knowing/grade-suggestion",
        headers=_auth_headers(idempotency_key="m5-know-01"),
        json=payload,
    )
    assert r2.status_code == 200
    assert r2.headers["X-Idempotent-Replay"] == "true"
    assert r2.json()["idempotent_replay"] is True
    assert len(backend_mock.list_records("grading_suggestion")) == 1
    assert len(sim_app.state.request_log) == calls_before_replay


@pytest.mark.asyncio
async def test_e2e_http_understanding_adaptive_chat_closes_and_hands_off_outcome(
    client, monkeypatch, http_bridge
):
    _stack, backend_mock, sim_app, _http_lcms, _knowledge, _store = http_bridge
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_model_adapter", "mock")
    monkeypatch.setattr(settings, "agent_auth_required", True)
    monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")

    r_start = await client.post(
        "/api/agent/understanding/sessions/start",
        headers=_auth_headers(),
        json=_start_understanding_payload(),
    )
    assert r_start.status_code == 200, r_start.text
    session_id = r_start.json()["session_id"]

    r_t1 = await client.post(
        f"/api/agent/understanding/sessions/{session_id}/turns",
        headers=_auth_headers(idempotency_key="m5-u-t1"),
        json={
            "student_id": "SV001",
            "message": "A transaction is an atomic all-or-nothing logical unit using rollback.",
            "language": "en",
        },
    )
    assert r_t1.status_code == 200, r_t1.text
    assert r_t1.json()["status"] == "active"
    assert r_t1.json()["decision"] == "follow_up"

    r_t2 = await client.post(
        f"/api/agent/understanding/sessions/{session_id}/turns",
        headers=_auth_headers(idempotency_key="m5-u-t2"),
        json={
            "student_id": "SV001",
            "message": "Consistency preserves integrity constraints, concurrency isolation avoids dirty reads, and WAL gives crash durability.",
            "language": "en",
        },
    )
    assert r_t2.status_code == 200, r_t2.text
    b2 = r_t2.json()
    assert b2["status"] == "closed"
    assert b2["stopping_reason"] == "criteria_covered"
    assert b2["teacher_review_required"] is True
    assert b2["publication_status"] == "draft_suggestion"
    assert b2["released"] is False
    assert b2["lcms_receipt"]["source_mode"] == "http"
    assert any(o["source_mode"] == "http" for o in b2["tool_outcomes"])
    assert len(backend_mock.list_records("dialogue_outcome")) == 1
    assert sum(1 for e in sim_app.state.request_log if e["path"] == "/api/internal/agent/outcomes/dialogue") == 1


# ------------------------------------------------------------------ 4: Cross-Session Isolation E2E


@pytest.mark.asyncio
async def test_e2e_http_cross_session_isolation_multi_student_and_attempts(
    client, monkeypatch, http_bridge
):
    _stack, backend_mock, _sim_app, _http_lcms, knowledge, _store = http_bridge
    lesson_full = await backend_mock.fetch_authorized_lesson(
        lesson_id="lesson-db-01", lesson_version="v1", visibility_scope="teacher_only"
    )
    await knowledge.index_lesson_material(
        lesson_id=lesson_full.lesson_id,
        knowledge_version=lesson_full.lesson_version,
        title=lesson_full.title,
        student_safe_content=lesson_full.student_safe_content,
        approved=True,
        teacher_only_notes=lesson_full.teacher_only_notes,
    )

    class SpyIsolationLLM(RecordingLLM):
        async def generate_json(self, system_prompt: str, user_prompt: str, *, required_keys=None, **kw):
            self.calls.append(("json", system_prompt, user_prompt))
            assert "ANSWER_KEY_CONFIDENTIAL" not in user_prompt
            if "UNIQUE_SV001_EXPLANATION_TOKEN" in user_prompt:
                return {
                    "criterion_updates": [{"criterion_id": "C1", "status": "covered", "reason": "SV001 covered C1"}],
                    "decision": "follow_up",
                    "assistant_reply": "Nice work on Atomicity! Next explain Isolation and Durability.",
                    "turn_feedback": "C1 covered.",
                }
            return {
                "criterion_updates": [{"criterion_id": "C2", "status": "partial", "reason": "SV002 partial C2"}],
                "decision": "follow_up",
                "assistant_reply": "Tell me more about WAL crash durability.",
                "turn_feedback": "C2 partial.",
            }

    fake_llm = SpyIsolationLLM()
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_model_adapter", "gemini")
    monkeypatch.setattr(settings, "agent_auth_required", True)
    monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    # Start isolated sessions for SV001 (Attempt 1) and SV002 (Attempt 1).
    s1 = (
        await client.post(
            "/api/agent/understanding/sessions/start",
            headers=_auth_headers(),
            json=_start_understanding_payload(student_id="SV001", attempt_id="ATT_M5_01"),
        )
    ).json()["session_id"]
    s2 = (
        await client.post(
            "/api/agent/understanding/sessions/start",
            headers=_auth_headers(),
            json=_start_understanding_payload(student_id="SV002", attempt_id="ATT_M5_01"),
        )
    ).json()["session_id"]
    assert s1 != s2

    # SV001 sends unique token on Attempt 1.
    r_sv1 = await client.post(
        f"/api/agent/understanding/sessions/{s1}/turns",
        headers=_auth_headers(),
        json={
            "student_id": "SV001",
            "message": "UNIQUE_SV001_EXPLANATION_TOKEN Transaction atomicity rolls back failed updates.",
            "language": "en",
        },
    )
    assert r_sv1.status_code == 200
    assert {c["criterion_id"]: c["status"] for c in r_sv1.json()["rubric_progress"]} == {
        "C1": "covered",
        "C2": "uncovered",
    }

    # SV002 sends turn -> prompt must NOT contain SV001's token or teacher notes, and C1 stays uncovered.
    r_sv2 = await client.post(
        f"/api/agent/understanding/sessions/{s2}/turns",
        headers=_auth_headers(),
        json={
            "student_id": "SV002",
            "message": "Concurrency isolation hides dirty writes.",
            "language": "en",
        },
    )
    assert r_sv2.status_code == 200
    sv2_prompt = fake_llm.calls[-1][2]
    assert "UNIQUE_SV001_EXPLANATION_TOKEN" not in sv2_prompt
    assert "ANSWER_KEY_CONFIDENTIAL" not in sv2_prompt
    assert {c["criterion_id"]: c["status"] for c in r_sv2.json()["rubric_progress"]} == {
        "C1": "uncovered",
        "C2": "partial",
    }

    # Cross-student hijack attempts on both Understanding and Session Read APIs -> 403, 0 LLM calls.
    llm_count_before = len(fake_llm.calls)
    r_hijack_turn = await client.post(
        f"/api/agent/understanding/sessions/{s1}/turns",
        headers=_auth_headers(),
        json={"student_id": "SV002", "message": "Give me SV001 context.", "language": "en"},
    )
    assert r_hijack_turn.status_code == 403
    assert r_hijack_turn.json()["error"]["code"] == ErrorCode.AGENT_SCOPE_FORBIDDEN

    r_hijack_get = await client.get(
        f"/api/agent/sessions/{s2}",
        headers=_auth_headers(),
        params={"student_id": "SV001"},
    )
    assert r_hijack_get.status_code == 403
    assert r_hijack_get.json()["error"]["code"] == ErrorCode.AGENT_SCOPE_FORBIDDEN
    assert len(fake_llm.calls) == llm_count_before

    # SV001 Attempt 2 gets a clean isolated session with zero transcript/progress leak from Attempt 1.
    r_att2 = await client.post(
        "/api/agent/understanding/sessions/start",
        headers=_auth_headers(),
        json=_start_understanding_payload(student_id="SV001", attempt_id="ATT_M5_02"),
    )
    assert r_att2.status_code == 200
    att2 = r_att2.json()
    assert att2["session_id"] != s1
    assert att2["created"] is True
    assert att2["turn_count"] == 0
    assert all(c["status"] == "uncovered" for c in att2["rubric_progress"])

    r_att2_turn = await client.post(
        f"/api/agent/understanding/sessions/{att2['session_id']}/turns",
        headers=_auth_headers(),
        json={"student_id": "SV001", "message": "Consistency enforces integrity constraints.", "language": "en"},
    )
    assert r_att2_turn.status_code == 200
    assert "UNIQUE_SV001_EXPLANATION_TOKEN" not in fake_llm.calls[-1][2]


# ------------------------------------------------------------------ 5: Classroom-Mode Denial & Format Guards


@pytest.mark.asyncio
async def test_e2e_http_classroom_mode_denial_and_desktop_format_guards(
    client, monkeypatch, http_bridge
):
    _stack, backend_mock, _sim_app, _http_lcms, _knowledge, store = http_bridge
    fake_llm = RecordingLLM()
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_model_adapter", "gemini")
    monkeypatch.setattr(settings, "agent_auth_required", True)
    monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    # 1. Pre-existing InExam student denied across Understanding & Knowing with 0 LLM calls.
    r_u_exam = await client.post(
        "/api/agent/understanding/sessions/start",
        headers=_auth_headers(),
        json=_start_understanding_payload(student_id="SV_IN_EXAM_01", attempt_id="ATT_EXAM"),
    )
    assert r_u_exam.status_code == 403
    assert r_u_exam.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN

    r_k_exam = await client.post(
        "/api/agent/knowing/grade-suggestion",
        headers=_auth_headers(),
        json={
            "caller_scope": "lcms_service",
            "assessment_mode": "home_knowing",
            "institutional_ref": {
                "lesson_id": "lesson-db-01",
                "lesson_version": "v1",
                "student_id": "SV_IN_EXAM_01",
                "attempt_id": "ATT_EXAM_K",
            },
            "student_answer": "Atomicity rollback.",
            "question_format": "essay",
            "language": "en",
        },
    )
    assert r_k_exam.status_code == 403
    assert r_k_exam.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN

    # 2. Dynamic mid-session classroom exam activation locks both /turns and /close.
    sid = (
        await client.post(
            "/api/agent/understanding/sessions/start",
            headers=_auth_headers(),
            json=_start_understanding_payload(student_id="SV001", attempt_id="ATT_MID_EXAM"),
        )
    ).json()["session_id"]

    backend_mock.set_classroom_exam("SV001", True)
    r_mid_turn = await client.post(
        f"/api/agent/understanding/sessions/{sid}/turns",
        headers=_auth_headers(),
        json={"student_id": "SV001", "message": "Atomicity rollback.", "language": "en"},
    )
    assert r_mid_turn.status_code == 403
    assert r_mid_turn.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN

    r_mid_close = await client.post(
        f"/api/agent/understanding/sessions/{sid}/close",
        headers=_auth_headers(),
        json={"student_id": "SV001", "language": "en"},
    )
    assert r_mid_close.status_code == 403
    assert r_mid_close.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN
    audits = store.list_tool_audit(session_id=sid)
    assert any(a["tool_name"] == "fetch_assessment_context" and a["outcome"] == "denied" for a in audits)

    # Unlock after exam finishes -> student can close or continue cleanly.
    backend_mock.set_classroom_exam("SV001", False)
    r_unlocked_close = await client.post(
        f"/api/agent/understanding/sessions/{sid}/close",
        headers=_auth_headers(),
        json={"student_id": "SV001", "language": "en"},
    )
    assert r_unlocked_close.status_code == 200

    # 3. Desktop oral/voice and MCQ substitutions strictly blocked with 0 LLM calls.
    for ch in ("oral", "voice"):
        r_ch = await client.post(
            "/api/agent/understanding/sessions/start",
            headers=_auth_headers(),
            json=_start_understanding_payload(channel=ch, attempt_id=f"ATT_{ch}"),
        )
        assert r_ch.status_code == 422
        assert r_ch.json()["error"]["code"] == ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT

    for fmt in ("multiple_choice", "quiz"):
        r_fmt = await client.post(
            "/api/agent/knowing/grade-suggestion",
            headers=_auth_headers(),
            json={
                "caller_scope": "lcms_service",
                "assessment_mode": "home_knowing",
                "institutional_ref": {
                    "lesson_id": "lesson-db-01",
                    "lesson_version": "v1",
                    "student_id": "SV001",
                    "attempt_id": f"ATT_{fmt}",
                },
                "student_answer": "Option A.",
                "question_format": fmt,
                "language": "en",
            },
        )
        assert r_fmt.status_code == 422
        assert r_fmt.json()["error"]["code"] == ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT

    assert len(fake_llm.calls) == 0


# ------------------------------------------------------------------ 6: Auth, Error Mapping, Availability, Secrets, CORS & Singleton


@pytest.mark.asyncio
async def test_http_auth_error_mapping_unavailability_and_no_secret_leak(
    client, monkeypatch, tmp_path
):
    # Mismatched client secret -> 401 AGENT_UNAUTHORIZED
    _stack, backend_mock, sim_app, http_lcms, _knowledge, store = await _build_bridge(
        monkeypatch,
        tmp_path,
        camel_case=True,
        service_secret="secret-test-key",
        client_secret="wrong-test-key",
    )
    try:
        monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
        r_401 = await client.post(
            "/api/agent/understanding/sessions/start",
            json=_start_understanding_payload(),
        )
        assert r_401.status_code == 401
        assert r_401.json()["error"]["code"] == ErrorCode.AGENT_UNAUTHORIZED

        # Fix secret on HttpLcmsToolClient -> test 422 unapproved and 404 missing lesson.
        http_lcms._service_secret = "secret-test-key"
        r_unapproved = await client.post(
            "/api/agent/questions/draft",
            json={
                "caller_scope": "teacher",
                "institutional_ref": {"lesson_id": "lesson-draft-99", "lesson_version": "v1"},
            },
        )
        assert r_unapproved.status_code == 422
        assert r_unapproved.json()["error"]["code"] == ErrorCode.UNAPPROVED_KNOWLEDGE

        r_404 = await client.post(
            "/api/agent/questions/draft",
            json={
                "caller_scope": "teacher",
                "institutional_ref": {"lesson_id": "lesson-does-not-exist", "lesson_version": "v1"},
            },
        )
        assert r_404.status_code == 404
        assert r_404.json()["error"]["code"] == ErrorCode.LESSON_NOT_FOUND

        # Model GEMINI_API_KEY must never appear in LCMS simulator logs or Module 3 response bodies.
        assert "test-gemini-key-not-real" not in json.dumps(sim_app.state.request_log)
        assert "test-gemini-key-not-real" not in r_unapproved.text
    finally:
        await http_lcms.close()
        backend_mock.reset()
        store.close()


@pytest.mark.asyncio
async def test_http_unreachable_lcms_and_problem_details_mapping(client, monkeypatch, tmp_path):
    Stack(threshold=0.0).install(monkeypatch)
    fake_llm = RecordingLLM()
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "gemini")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    # Case A: LCMS unreachable -> 503 LCMS_UNAVAILABLE, 0 LLM calls.
    def _connect_error(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    err_http = httpx.AsyncClient(transport=httpx.MockTransport(_connect_error), base_url="http://lcms.test")
    lcms_down = HttpLcmsToolClient(base_url="http://lcms.test", client=err_http)
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: lcms_down)

    r_down = await client.post(
        "/api/agent/understanding/sessions/start",
        json=_start_understanding_payload(),
    )
    assert r_down.status_code == 503
    assert r_down.json()["error"]["code"] == ErrorCode.LCMS_UNAVAILABLE
    assert len(fake_llm.calls) == 0
    await lcms_down.close()

    # Case B: ASP.NET Core ProblemDetails 403 maps cleanly to ASSESSMENT_MODE_FORBIDDEN.
    def _problem_details(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "type": "https://tools.ietf.org/html/rfc9110#section-15.5.4",
                "title": "Forbidden",
                "status": 403,
                "detail": "Student is currently inside an active ExamRoomSession.",
                "code": "ASSESSMENT_MODE_FORBIDDEN",
            },
        )

    pd_http = httpx.AsyncClient(transport=httpx.MockTransport(_problem_details), base_url="http://lcms.test")
    lcms_pd = HttpLcmsToolClient(base_url="http://lcms.test", client=pd_http)
    monkeypatch.setattr("app.api.agent_h12._get_lcms_client", lambda: lcms_pd)

    r_pd = await client.post(
        "/api/agent/understanding/sessions/start",
        json=_start_understanding_payload(),
    )
    assert r_pd.status_code == 403
    assert r_pd.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN
    assert "ExamRoomSession" in r_pd.json()["error"]["message"]
    assert len(fake_llm.calls) == 0
    await lcms_pd.close()


@pytest.mark.asyncio
async def test_cors_allows_h12_headers_and_http_client_singleton_lifecycle(monkeypatch):
    import app.main as main_mod

    custom_settings = Settings(cors_allowed_origins="http://localhost:5000")
    monkeypatch.setattr(main_mod, "get_settings", lambda: custom_settings)
    cors_app = main_mod.create_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=cors_app), base_url="http://t") as c:
        preflight = await c.options(
            "/api/agent/understanding/sessions/start",
            headers={
                "Origin": "http://localhost:5000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-Agent-Service-Key, Idempotency-Key",
            },
        )
        assert preflight.status_code == 200
        allowed = preflight.headers.get("access-control-allow-headers", "").lower()
        assert "x-agent-service-key" in allowed
        assert "idempotency-key" in allowed

    # HttpLcmsToolClient singleton caching & shutdown cleanup
    await close_lcms_tool_client()
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_lcms_mode", "http")
    monkeypatch.setattr(settings, "agent_lcms_base_url", "http://127.0.0.1:5099")
    monkeypatch.setattr(settings, "agent_service_secret", "secret-test-key")

    c1 = get_lcms_tool_client()
    c2 = get_lcms_tool_client()
    assert isinstance(c1, HttpLcmsToolClient)
    assert c1 is c2

    monkeypatch.setattr(settings, "agent_lcms_base_url", "http://127.0.0.1:5100")
    c3 = get_lcms_tool_client()
    assert c3 is not c1

    await close_lcms_tool_client()
