"""H12 AI Agent Milestone 1 tests: framework, replaceable adapter, boundaries, auth & idempotency."""

from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings
from app.h12_agent.idempotency import get_idempotency_store
from tests.fakes import RecordingLLM, Stack, sample_pdf_bytes


@pytest.fixture(autouse=True)
def _clean_idempotency():
    store = get_idempotency_store()
    store.clear()
    yield
    store.clear()


def _base_payload(
    *,
    agent_role: str = "question_drafting",
    caller_scope: str = "teacher",
    assessment_mode: str = "teacher_question_drafting",
    lesson_id: str = "lesson-db-01",
    prompt_input: str = "Draft an essay question checking ACID transaction understanding.",
    student_id: str | None = None,
    attempt_id: str | None = None,
    document_id: str | None = None,
    requested_tools: list[str] | None = None,
    format_hint: str | None = "essay",
    channel: str = "text_chat",
    idempotency_key: str | None = None,
) -> dict:
    ref: dict = {"lesson_id": lesson_id, "lesson_version": "v1"}
    if student_id is not None:
        ref["student_id"] = student_id
    if attempt_id is not None:
        ref["attempt_id"] = attempt_id
    body: dict = {
        "agent_role": agent_role,
        "caller_scope": caller_scope,
        "assessment_mode": assessment_mode,
        "institutional_ref": ref,
        "prompt_input": prompt_input,
        "channel": channel,
    }
    if document_id is not None:
        body["document_id"] = document_id
    if requested_tools is not None:
        body["requested_tools"] = requested_tools
    if format_hint is not None:
        body["format_hint"] = format_hint
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return body


# ------------------------------------------------------------------ capabilities


@pytest.mark.asyncio
async def test_capabilities_reports_langgraph_and_switches_mock_vs_live(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "agent_model_adapter", "mock")
    r_mock = await client.get("/api/agent/capabilities")
    assert r_mock.status_code == 200
    body_mock = r_mock.json()
    assert body_mock["framework"] == "langgraph"
    assert body_mock["framework_version"]
    assert body_mock["adapter_mode"] == "mock"
    assert body_mock["is_mock"] is True
    assert "classroom_exam" in body_mock["forbidden_modes"]
    assert set(body_mock["forbidden_formats"]) >= {"multiple_choice", "true_false", "quiz"}
    assert set(body_mock["forbidden_channels"]) >= {"voice", "oral"}
    roles = {r["role"]: r for r in body_mock["roles"]}
    assert set(roles) == {"question_drafting", "grading_support", "dialogue_assessment"}
    assert all(r["auto_release_allowed"] is False for r in roles.values())
    assert roles["dialogue_assessment"]["visibility_scope"] == "student_safe"

    monkeypatch.setattr(s, "agent_model_adapter", "gemini")
    r_live = await client.get("/api/agent/capabilities")
    assert r_live.status_code == 200
    body_live = r_live.json()
    assert body_live["adapter_mode"] == "live"
    assert body_live["is_mock"] is False


# ------------------------------------------------------------------ mock vs live execution


@pytest.mark.asyncio
async def test_execute_skeleton_mock_mode_labels_response_and_runs_5_langgraph_steps(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
    r = await client.post("/api/agent/execute-skeleton", json=_base_payload())
    assert r.status_code == 200, r.text
    assert r.headers["X-AI-Adapter-Mode"] == "mock"
    assert r.headers["X-AI-Framework"] == "langgraph"
    assert r.headers["X-Idempotent-Replay"] == "false"

    body = r.json()
    assert body["publication_status"] == "draft_suggestion"
    assert body["released"] is False
    assert body["teacher_review_required"] is True
    assert body["idempotent_replay"] is False
    assert body["draft_output"]["summary"].startswith("[MOCK]")

    trace = body["trace"]
    assert trace["framework"] == "langgraph"
    assert trace["adapter_mode"] == "mock"
    assert trace["is_mock"] is True
    assert trace["provider"] == "deterministic_mock"
    assert [s["step"] for s in trace["observable_steps"]] == [
        "validate_mode_and_scope",
        "select_allowed_tools",
        "retrieve_grounded_evidence",
        "invoke_model_adapter",
        "validate_draft_boundary",
    ]
    assert trace["observable_steps"][3]["outcome"] == "mocked"


@pytest.mark.asyncio
async def test_execute_skeleton_live_mode_uses_injected_llm_and_strips_forbidden_fields(client, monkeypatch):
    class StructuredFakeLLM(RecordingLLM):
        async def generate_json(self, system_prompt: str, user_prompt: str, *, required_keys=None, **kw):
            self.calls.append(("json", system_prompt, user_prompt))
            return {
                "summary": "Explain ACID atomicity based on section 2.",
                "proposed_action": "handoff_question_draft_to_teacher",
                "criterion_links": ["C1"],
                # Attempted model forgery must be stripped by validate_draft_boundary
                "released": True,
                "final_grade": 10.0,
            }

    fake_llm = StructuredFakeLLM()
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "gemini")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    r = await client.post("/api/agent/execute-skeleton", json=_base_payload())
    assert r.status_code == 200, r.text
    assert r.headers["X-AI-Adapter-Mode"] == "live"
    assert len(fake_llm.calls) == 1

    body = r.json()
    assert body["trace"]["adapter_mode"] == "live"
    assert body["trace"]["is_mock"] is False
    assert body["trace"]["provider"] == "google_gemini"
    assert body["publication_status"] == "draft_suggestion"
    assert body["released"] is False
    assert "released" not in body["draft_output"]["payload"]
    assert "final_grade" not in body["draft_output"]["payload"]


# ------------------------------------------------------------------ classroom examination denial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_role,caller_scope",
    [
        ("dialogue_assessment", "student"),
        ("grading_support", "teacher"),
        ("question_drafting", "lcms_service"),
    ],
)
async def test_classroom_exam_mode_is_strictly_denied_without_calling_llm(
    client, monkeypatch, agent_role, caller_scope
):
    fake_llm = RecordingLLM()
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "gemini")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    payload = _base_payload(
        agent_role=agent_role,
        caller_scope=caller_scope,
        assessment_mode="classroom_exam",
        student_id="DTC235210055",
        attempt_id="att-1",
    )
    r = await client.post("/api/agent/execute-skeleton", json=payload)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "ASSESSMENT_MODE_FORBIDDEN"
    assert fake_llm.calls == []


# ------------------------------------------------------------------ format & channel substitution denial


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_format", ["multiple_choice", "mcq", "true_false", "quiz"])
async def test_knowing_and_drafting_reject_multiple_choice_substitution(client, monkeypatch, bad_format):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
    r = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            agent_role="grading_support",
            caller_scope="teacher",
            assessment_mode="home_knowing",
            format_hint=bad_format,
        ),
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNSUPPORTED_ASSESSMENT_FORMAT"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_channel", ["voice", "oral", "audio", "speech"])
async def test_understanding_dialogue_rejects_voice_or_oral_channel(client, monkeypatch, bad_channel):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
    r = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            agent_role="dialogue_assessment",
            caller_scope="student",
            assessment_mode="home_understanding",
            student_id="DTC235210055",
            attempt_id="att-1",
            format_hint="adaptive_text_chat",
            channel=bad_channel,
        ),
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNSUPPORTED_ASSESSMENT_FORMAT"


# ------------------------------------------------------------------ role, scope & visibility boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("role,mode", [
    ("question_drafting", "teacher_question_drafting"),
    ("grading_support", "home_knowing"),
])
async def test_student_scope_cannot_invoke_teacher_agent_roles(client, monkeypatch, role, mode):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
    r = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(agent_role=role, caller_scope="student", assessment_mode=mode),
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "AGENT_SCOPE_FORBIDDEN"


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden_tool", ["record_grading_suggestion", "record_question_draft"])
async def test_dialogue_assessment_rejects_out_of_scope_tools(client, monkeypatch, forbidden_tool):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")
    r = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            agent_role="dialogue_assessment",
            caller_scope="student",
            assessment_mode="home_understanding",
            student_id="DTC235210055",
            attempt_id="att-1",
            format_hint="adaptive_text_chat",
            requested_tools=[forbidden_tool],
        ),
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "AGENT_SCOPE_FORBIDDEN"


@pytest.mark.asyncio
async def test_dialogue_assessment_enforces_student_safe_visibility_and_session_ids(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")

    missing_ids = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            agent_role="dialogue_assessment",
            caller_scope="student",
            assessment_mode="home_understanding",
            format_hint="adaptive_text_chat",
        ),
    )
    assert missing_ids.status_code == 422
    assert missing_ids.json()["error"]["code"] == "VALIDATION_ERROR"

    ok_resp = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            agent_role="dialogue_assessment",
            caller_scope="student",
            assessment_mode="home_understanding",
            student_id="DTC235210055",
            attempt_id="att-2",
            format_hint="adaptive_text_chat",
        ),
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["trace"]["visibility_scope"] == "student_safe"


# ------------------------------------------------------------------ service authentication


@pytest.mark.asyncio
async def test_service_auth_required_enforces_x_agent_service_key(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "agent_model_adapter", "mock")
    monkeypatch.setattr(s, "agent_auth_required", True)
    monkeypatch.setattr(s, "agent_service_secret", "test-secret-h12")

    r_missing = await client.post("/api/agent/execute-skeleton", json=_base_payload())
    assert r_missing.status_code == 401
    assert r_missing.json()["error"]["code"] == "AGENT_UNAUTHORIZED"

    r_wrong = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(),
        headers={"X-Agent-Service-Key": "wrong-key"},
    )
    assert r_wrong.status_code == 401
    assert r_wrong.json()["error"]["code"] == "AGENT_UNAUTHORIZED"

    r_ok = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(),
        headers={"X-Agent-Service-Key": "test-secret-h12"},
    )
    assert r_ok.status_code == 200


# ------------------------------------------------------------------ idempotency


@pytest.mark.asyncio
async def test_idempotency_replay_avoids_duplicate_llm_call_and_detects_conflict(client, monkeypatch):
    class CountingLLM(RecordingLLM):
        async def generate_json(self, system_prompt: str, user_prompt: str, *, required_keys=None, **kw):
            self.calls.append(("json", system_prompt, user_prompt))
            return {
                "summary": "Drafted essay prompt once.",
                "proposed_action": "handoff_question_draft_to_teacher",
            }

    fake_llm = CountingLLM()
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "gemini")
    monkeypatch.setattr("app.api.agent_h12._get_llm_service", lambda: fake_llm)

    payload = _base_payload(prompt_input="Generate 1 fixed essay question about 3NF.")
    headers = {"Idempotency-Key": "req-idem-001"}

    first = await client.post("/api/agent/execute-skeleton", json=payload, headers=headers)
    assert first.status_code == 200
    assert first.json()["idempotent_replay"] is False
    assert first.headers["X-Idempotent-Replay"] == "false"
    assert len(fake_llm.calls) == 1

    second = await client.post("/api/agent/execute-skeleton", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert second.headers["X-Idempotent-Replay"] == "true"
    assert len(fake_llm.calls) == 1

    mutated = _base_payload(prompt_input="Different question prompt with same key.")
    conflict = await client.post("/api/agent/execute-skeleton", json=mutated, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"


# ------------------------------------------------------------------ grounded retrieval inside graph


@pytest.mark.asyncio
async def test_skeleton_retrieves_chromadb_evidence_when_document_id_supplied(client, monkeypatch):
    Stack(threshold=0.0).install(monkeypatch)
    monkeypatch.setattr(get_settings(), "agent_model_adapter", "mock")

    up = await client.post(
        "/api/documents/upload",
        files={
            "file": (
                "acid_lesson.pdf",
                sample_pdf_bytes(["Database transactions satisfy Atomicity, Consistency, Isolation and Durability."]),
                "application/pdf",
            )
        },
    )
    assert up.status_code == 200
    doc_id = up.json()["document_id"]

    r = await client.post(
        "/api/agent/execute-skeleton",
        json=_base_payload(
            document_id=doc_id,
            prompt_input="Explain Atomicity and Durability in database transactions.",
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["evidence_refs"]) >= 1
    assert all(ref["document_id"] == doc_id and ref["page"] == 1 for ref in body["evidence_refs"])
    retrieve_step = next(s for s in body["trace"]["observable_steps"] if s["step"] == "retrieve_grounded_evidence")
    assert retrieve_step["outcome"] == "ok"


# ------------------------------------------------------------------ config validation & secret safety


@pytest.mark.parametrize(
    "override,expected_msg",
    [
        ({"agent_model_adapter": "unknown"}, "AGENT_MODEL_ADAPTER"),
        ({"agent_lcms_mode": "ftp"}, "AGENT_LCMS_MODE"),
        ({"agent_lcms_base_url": "not-a-url"}, "AGENT_LCMS_BASE_URL"),
        ({"agent_max_dialogue_turns": 0}, "AGENT_MAX_DIALOGUE_TURNS"),
        ({"agent_idempotency_ttl_s": 0}, "AGENT_IDEMPOTENCY_TTL_S"),
        ({"agent_auth_required": True, "agent_service_secret": ""}, "AGENT_SERVICE_SECRET"),
    ],
)
def test_h12_settings_validation_fails_fast(override, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        Settings(_env_file=None, **override)


def test_h12_service_secret_never_leaks_in_repr_or_error():
    secret = "test-secret-canary-do-not-leak-987654"
    s = Settings(_env_file=None, agent_auth_required=True, agent_service_secret=secret)
    assert secret not in repr(s)
    assert secret not in str(s)
    with pytest.raises(ValueError) as exc:
        Settings(_env_file=None, agent_auth_required=True, agent_service_secret=secret, agent_model_adapter="bad")
    assert secret not in str(exc.value)
