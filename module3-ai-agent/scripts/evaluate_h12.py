"""H12 6-Criteria Benchmark Evaluator (PI 4.1) & Markdown/JSON Report Generator (Milestone 6).

Evaluates all six PI 4.1 dimensions over HTTP (FastAPI app -> HttpLcmsToolClient -> LCMS Simulator):
1. question_relevance
2. lesson_grounding
3. rubric_coverage
4. scoring_agreement
5. understanding_followup_quality (+ PI 3.4 adversarial dialogue security)
6. failure_handling (cross-session isolation, classroom-mode denial, voice/oral denial,
   malformed JSON recovery, LCMS unavailability, auth, idempotency)

Usage:
    python scripts/evaluate_h12.py [--mode mock|gemini] [--dataset PATH]
                                   [--output-md PATH] [--output-json PATH] [--validate-only]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import agent_h12  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import ErrorCode  # noqa: E402
from app.core.rate_limit import limiter  # noqa: E402
from app.h12_agent.idempotency import get_idempotency_store  # noqa: E402
from app.h12_agent.knowledge import LessonKnowledgeService  # noqa: E402
from app.h12_agent.lcms_simulator import create_lcms_simulator_app  # noqa: E402
from app.h12_agent.lcms_tools import (  # noqa: E402
    HttpLcmsToolClient,
    LessonMaterialRecord,
    LessonQuestionsRubricBundle,
    MockLcmsToolClient,
)
from app.h12_agent.session_store import SessionMemoryStore  # noqa: E402
from app.main import app as module3_app  # noqa: E402
from app.repositories.chroma_repository import ChromaRepository  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.vector_service import VectorService  # noqa: E402

DEFAULT_DATASET = ROOT / "tests" / "golden" / "h12_eval_dataset.json"
REQUIRED_CRITERIA = (
    "question_relevance",
    "lesson_grounding",
    "rubric_coverage",
    "scoring_agreement",
    "understanding_followup_quality",
    "failure_handling",
)
_WORD_RE = re.compile(r"[a-z0-9_]+")
_EMBED_DIM = 512


def _lexical_vector(text: str) -> list[float]:
    vec = [0.0] * _EMBED_DIM
    for word in _WORD_RE.findall((text or "").lower()):
        if len(word) < 3:
            continue
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _OfflineLexicalEmbedding:
    """Self-contained deterministic bag-of-words embedder for offline PI 4.1 verification."""

    model_name = "lexical-eval-embedding"

    async def embed_text(self, text: str) -> list[float]:
        return _lexical_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_lexical_vector(t) for t in texts]

    async def close(self) -> None:
        return None


class _BadJsonLLM(LLMService):
    """Fake LLM returning malformed Understanding JSON to test fail-closed recovery."""

    model_name = "bad-json-eval-llm"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system_prompt: str, user_prompt: str, **_: Any) -> str:
        self.calls += 1
        return "not json"

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        required_keys: Optional[list[str]] = None,
        **_: Any,
    ) -> Any:
        self.calls += 1
        return {"criterion_updates": [], "decision": "invalid_decision", "assistant_reply": ""}

    async def check_health(self) -> dict[str, Any]:
        return {"status": "ok"}


def load_h12_dataset(path: Path | str = DEFAULT_DATASET) -> dict[str, Any]:
    """Load and structurally validate the multi-lesson H12 6-criteria golden dataset."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("H12 evaluation dataset root must be a JSON object.")

    meta = data.get("metadata")
    if not isinstance(meta, dict) or not isinstance(meta.get("acceptance_thresholds"), dict):
        raise ValueError("Dataset 'metadata.acceptance_thresholds' object is required.")

    for key in REQUIRED_CRITERIA:
        cases = data.get(key)
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Dataset must define a non-empty list for '{key}'.")
    return data


def _register_extra_lessons(mock_lcms: MockLcmsToolClient, dataset: dict[str, Any]) -> list[tuple[str, str]]:
    lessons: list[tuple[str, str]] = [
        ("lesson-db-01", "v1"),
        ("lesson-db-01", "v2"),
        ("lesson-se-02", "v1"),
    ]
    for item in dataset.get("extra_lessons", []):
        lesson = LessonMaterialRecord.model_validate(item["lesson"])
        bundle = LessonQuestionsRubricBundle.model_validate(item["bundle"])
        mock_lcms.register_lesson(lesson, bundle)
        lessons.append((lesson.lesson_id, lesson.lesson_version))
    return lessons


async def _preindex_all_approved(
    mock_lcms: MockLcmsToolClient,
    lesson_keys: list[tuple[str, str]],
    *services: LessonKnowledgeService,
) -> None:
    for lid, ver in lesson_keys:
        full = await mock_lcms.fetch_authorized_lesson(
            lesson_id=lid, lesson_version=ver, visibility_scope="teacher_only"
        )
        for svc in services:
            await svc.index_lesson_material(
                lesson_id=full.lesson_id,
                knowledge_version=full.lesson_version,
                title=full.title,
                student_safe_content=full.student_safe_content,
                approved=full.approved,
                teacher_only_notes=full.teacher_only_notes,
            )


def _auth_headers(
    *,
    secret: str,
    idempotency_key: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict[str, str]:
    headers: dict[str, str] = {"X-Agent-Service-Key": secret}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if request_id:
        headers["X-Request-ID"] = request_id
    return headers


async def run_h12_evaluation(
    *,
    dataset_path: Path | str = DEFAULT_DATASET,
    mode: Literal["mock", "gemini"] = "mock",
) -> dict[str, Any]:
    """Execute the complete H12 6-criteria evaluation over HTTP + Simulator and return metrics."""
    started = time.perf_counter()
    dataset = load_h12_dataset(dataset_path)
    thresholds = dataset["metadata"]["acceptance_thresholds"]
    secret = "secret-test-key"

    mock_lcms = MockLcmsToolClient()
    lesson_keys = _register_extra_lessons(mock_lcms, dataset)
    sim_app = create_lcms_simulator_app(mock_client=mock_lcms, service_secret=secret, camel_case=True)
    sim_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=sim_app), base_url="http://lcms.test")
    http_lcms = HttpLcmsToolClient(base_url="http://lcms.test", service_secret=secret, client=sim_http)

    session_store = SessionMemoryStore(":memory:")
    chroma_client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))

    if mode == "mock":
        embedder: Any = _OfflineLexicalEmbedding()
    else:
        from app.services.embedding_service import get_embedding_service

        embedder = get_embedding_service()

    repo_open = ChromaRepository(client=chroma_client, collection_name=f"h12_eval_open_{uuid.uuid4().hex[:8]}")
    repo_strict = ChromaRepository(client=chroma_client, collection_name=f"h12_eval_strict_{uuid.uuid4().hex[:8]}")
    vec_open = VectorService(embedding_service=embedder, chroma_repo=repo_open)
    vec_strict = VectorService(embedding_service=embedder, chroma_repo=repo_strict)
    know_open = LessonKnowledgeService(
        embedding_service=embedder, chroma_repo=repo_open, vector_service=vec_open, similarity_threshold=0.0
    )
    know_strict = LessonKnowledgeService(
        embedding_service=embedder, chroma_repo=repo_strict, vector_service=vec_strict, similarity_threshold=0.95
    )

    settings = get_settings()
    orig_adapter = settings.agent_model_adapter
    orig_auth_req = settings.agent_auth_required
    orig_secret = settings.agent_service_secret
    orig_lcms_mode = settings.agent_lcms_mode
    orig_max_retries = settings.llm_max_retries
    orig_retry_backoff = settings.llm_retry_backoff_s
    orig_get_lcms = agent_h12._get_lcms_client
    orig_get_know = agent_h12._get_knowledge_service
    orig_get_store = agent_h12._get_session_store
    orig_get_llm = agent_h12._get_llm_service

    api_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=module3_app), base_url="http://testserver")
    idem_store = get_idempotency_store()
    idem_store.clear()
    limiter.reset()

    zero_silent_grade_publication = True
    zero_answer_key_leakage = True

    async def _pace() -> None:
        if mode == "gemini":
            await asyncio.sleep(1.8)

    try:
        settings.agent_model_adapter = "mock" if mode == "mock" else "gemini"
        if mode == "gemini":
            from app.services import llm_service as _llm_mod

            settings.llm_max_retries = 4
            settings.llm_retry_backoff_s = 3.5
            _llm_mod._llm_service = None
        settings.agent_auth_required = True
        settings.agent_service_secret = secret
        settings.agent_lcms_mode = "http"
        agent_h12._get_lcms_client = lambda: http_lcms
        agent_h12._get_knowledge_service = lambda: know_open
        agent_h12._get_session_store = lambda: session_store

        await _preindex_all_approved(mock_lcms, lesson_keys, know_open, know_strict)

        # ---- 1. Question Relevance ----
        qr_details: list[dict[str, Any]] = []
        for case in dataset["question_relevance"]:
            if case.get("expected_status_code") == 200:
                await _pace()
            r = await api_client.post(
                "/api/agent/questions/draft",
                headers=_auth_headers(secret=secret),
                json={
                    "caller_scope": "teacher",
                    "assessment_mode": "teacher_question_drafting",
                    "institutional_ref": {"lesson_id": case["lesson_id"], "lesson_version": case["lesson_version"]},
                    "topic_focus": case["topic_focus"],
                    "number_of_questions": case["number_of_questions"],
                    "difficulty": case["difficulty"],
                    "question_format": case["question_format"],
                    "language": "en",
                },
            )
            ok = r.status_code == case["expected_status_code"]
            if ok and r.status_code == 200:
                b = r.json()
                drafts = b.get("drafts", [])
                joined = " ".join(
                    f"{d['question']} {d['expected_answer']} {' '.join(d.get('key_points', []))}"
                    for d in drafts
                ).lower()
                kw_hits = all(kw.lower() in joined for kw in case.get("expected_keywords", []))
                format_ok = all(d["format"] == "essay" and not d["released"] and d["teacher_review_required"] for d in drafts)
                ev_ok = all(len(d.get("evidence_refs", [])) >= 1 for d in drafts)
                ok = bool(drafts) and kw_hits and format_ok and ev_ok and not b.get("released", True)
                if "ANSWER_KEY_CONFIDENTIAL" in joined:
                    zero_answer_key_leakage = False
                if b.get("released") is not False or b.get("publication_status") != "draft_suggestion":
                    zero_silent_grade_publication = False
            elif ok and case.get("expected_error_code"):
                ok = r.json().get("error", {}).get("code") == case["expected_error_code"]
            qr_details.append({"id": case["id"], "passed": ok, "http_status": r.status_code})

        # ---- 2. Lesson Grounding ----
        lg_details: list[dict[str, Any]] = []
        for case in dataset["lesson_grounding"]:
            limiter.reset()
            use_strict = float(case.get("similarity_threshold", 0.0)) > 0.5
            agent_h12._get_knowledge_service = (lambda: know_strict) if use_strict else (lambda: know_open)
            try:
                if case.get("expected_error_code") == ErrorCode.UNAPPROVED_KNOWLEDGE:
                    r = await api_client.post(
                        "/api/agent/knowledge/index",
                        headers=_auth_headers(secret=secret),
                        json={
                            "caller_scope": "teacher",
                            "lesson_id": case["lesson_id"],
                            "knowledge_version": case["lesson_version"],
                            "from_lcms": True,
                        },
                    )
                else:
                    r = await api_client.post(
                        "/api/agent/knowledge/retrieve",
                        headers=_auth_headers(secret=secret),
                        json={
                            "agent_role": "dialogue_assessment",
                            "caller_scope": "student",
                            "assessment_mode": "home_understanding",
                            "institutional_ref": {
                                "lesson_id": case["lesson_id"],
                                "lesson_version": case["lesson_version"],
                                "student_id": "SV_EVAL_LG",
                                "attempt_id": "ATT_EVAL_LG",
                            },
                            "query": case["query"],
                        },
                    )
            finally:
                agent_h12._get_knowledge_service = lambda: know_open

            ok = r.status_code == case["expected_status_code"]
            if ok and r.status_code == 200:
                b = r.json()
                ctx = b.get("context", "")
                marker = case.get("forbidden_marker", "ANSWER_KEY_CONFIDENTIAL")
                leaked = marker in ctx
                if leaked:
                    zero_answer_key_leakage = False
                refs = b.get("evidence_refs", [])
                if case.get("expect_insufficient_context"):
                    ok = b.get("insufficient_context") is True and len(refs) == 0 and not leaked
                else:
                    top_ver_ok = b.get("knowledge_version") == case["lesson_version"]
                    chunks = b.get("chunks", [])
                    chunk_ver_ok = all(
                        ch.get("knowledge_version") == case["lesson_version"]
                        and ch.get("visibility") == case["visibility_scope"]
                        for ch in chunks
                    )
                    ok = bool(refs) and not b.get("insufficient_context") and not leaked and top_ver_ok and chunk_ver_ok
            elif ok and case.get("expected_error_code"):
                ok = r.json().get("error", {}).get("code") == case["expected_error_code"]
            lg_details.append({"id": case["id"], "passed": ok, "http_status": r.status_code})

        # ---- 3. Rubric Coverage ----
        rc_details: list[dict[str, Any]] = []
        for idx, case in enumerate(dataset["rubric_coverage"], start=1):
            limiter.reset()
            if case.get("kind") == "understanding_progress":
                r = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {
                            "lesson_id": case["lesson_id"],
                            "lesson_version": case["lesson_version"],
                            "student_id": "SV_EVAL_RC",
                            "attempt_id": f"ATT_EVAL_RC_{idx}",
                        },
                        "language": "en",
                    },
                )
                ok = r.status_code == 200
                if ok:
                    b = r.json()
                    got_ids = [c["criterion_id"] for c in b["rubric_progress"]]
                    ok = got_ids == case["expected_criterion_ids"]
            else:
                await _pace()
                r = await api_client.post(
                    "/api/agent/knowing/grade-suggestion",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "lcms_service",
                        "assessment_mode": "home_knowing",
                        "institutional_ref": {
                            "lesson_id": case["lesson_id"],
                            "lesson_version": case["lesson_version"],
                            "student_id": "SV_EVAL_RC",
                            "attempt_id": f"ATT_EVAL_RC_{idx}",
                        },
                        "student_answer": case["student_answer"],
                        "question_format": "essay",
                        "language": "en",
                    },
                )
                ok = r.status_code == 200
                if ok:
                    b = r.json()
                    got_ids = [c["criterion_id"] for c in b["criteria"]]
                    sum_ok = abs(sum(c["suggested_points"] for c in b["criteria"]) - b["suggested_total"]) < 1e-6
                    reasons_ok = all(bool(str(c.get("reason", "")).strip()) for c in b["criteria"])
                    no_release = b["released"] is False and b["teacher_handoff"] is True
                    if not no_release:
                        zero_silent_grade_publication = False
                    ok = got_ids == case["expected_criterion_ids"] and sum_ok and reasons_ok and no_release
            rc_details.append({"id": case["id"], "passed": ok, "http_status": r.status_code})

        # ---- 4. Scoring Agreement ----
        sa_details: list[dict[str, Any]] = []
        abs_errors: list[float] = []
        status_matches = 0
        for idx, case in enumerate(dataset["scoring_agreement"], start=1):
            limiter.reset()
            if not case.get("expected_warnings") and not case.get("expected_flags"):
                await _pace()
            r = await api_client.post(
                "/api/agent/knowing/grade-suggestion",
                headers=_auth_headers(secret=secret),
                json={
                    "caller_scope": "lcms_service",
                    "assessment_mode": "home_knowing",
                    "institutional_ref": {
                        "lesson_id": case["lesson_id"],
                        "lesson_version": case["lesson_version"],
                        "student_id": f"SV_EVAL_SA_{idx}",
                        "attempt_id": f"ATT_EVAL_SA_{idx}",
                    },
                    "student_answer": case["student_answer"],
                    "question_format": "essay",
                    "language": "en",
                },
            )
            ok = r.status_code == 200
            err = 999.0
            status_ok = False
            if ok:
                b = r.json()
                err = abs(float(b["suggested_total"]) - float(case["expected_score"]))
                abs_errors.append(err)
                status_ok = b["status"] == case["expected_status"]
                if status_ok:
                    status_matches += 1
                unmet_ok = b.get("unmet_required", []) == case.get("expected_unmet_required", [])
                flags_ok = all(f in b.get("flags", []) for f in case.get("expected_flags", []))
                warn_ok = all(w in b.get("warnings", []) for w in case.get("expected_warnings", []))
                boundary_ok = (
                    b.get("released") is False
                    and b.get("teacher_review_required") is True
                    and b.get("publication_status") == "draft_suggestion"
                )
                if not boundary_ok:
                    zero_silent_grade_publication = False
                ok = err <= float(thresholds["scoring_agreement_mae_max"]) and status_ok and unmet_ok and flags_ok and warn_ok and boundary_ok
            sa_details.append({"id": case["id"], "passed": ok, "abs_error": round(err, 3), "status_ok": status_ok})

        scoring_mae = round(sum(abs_errors) / max(1, len(abs_errors)), 3)
        status_agreement = round(status_matches / max(1, len(dataset["scoring_agreement"])), 4)

        # ---- 5. Understanding Follow-up Quality & PI 3.4 Dialogue Security ----
        uf_details: list[dict[str, Any]] = []
        sec_checks: list[bool] = []
        for idx, case in enumerate(dataset["understanding_followup_quality"], start=1):
            limiter.reset()
            ctype = case["type"]
            r_s = await api_client.post(
                "/api/agent/understanding/sessions/start",
                headers=_auth_headers(secret=secret),
                json={
                    "caller_scope": "student",
                    "assessment_mode": "home_understanding",
                    "channel": "text_chat",
                    "institutional_ref": {
                        "lesson_id": case["lesson_id"],
                        "lesson_version": case["lesson_version"],
                        "student_id": f"SV_EVAL_UF_{idx}",
                        "attempt_id": f"ATT_EVAL_UF_{idx}",
                    },
                    "max_turns": case.get("max_turns"),
                    "language": "en",
                },
            )
            ok = r_s.status_code == 200
            sid = r_s.json().get("session_id", "") if ok else ""

            if ok and ctype == "adaptive_convergence":
                for turn_spec in case["turns"]:
                    await _pace()
                    rt = await api_client.post(
                        f"/api/agent/understanding/sessions/{sid}/turns",
                        headers=_auth_headers(secret=secret),
                        json={"student_id": f"SV_EVAL_UF_{idx}", "message": turn_spec["message"], "language": "en"},
                    )
                    if rt.status_code != 200:
                        ok = False
                        break
                    tb = rt.json()
                    if "ANSWER_KEY_CONFIDENTIAL" in tb["assistant_reply"]:
                        zero_answer_key_leakage = False
                        ok = False
                    if tb.get("released") is not False:
                        zero_silent_grade_publication = False
                    prog = {c["criterion_id"]: c["status"] for c in tb["rubric_progress"]}
                    ok = (
                        ok
                        and tb["decision"] == turn_spec["expected_decision"]
                        and tb["status"] == turn_spec["expected_status"]
                        and all(prog.get(cid) == "covered" for cid in turn_spec["expected_covered"])
                    )
                    if "expected_stopping_reason" in turn_spec:
                        ok = ok and tb.get("stopping_reason") == turn_spec["expected_stopping_reason"] and tb.get("lcms_receipt") is not None

            elif ok and ctype == "max_turns_reached":
                last_b: dict[str, Any] = {}
                for msg in case["turns"]:
                    await _pace()
                    rt = await api_client.post(
                        f"/api/agent/understanding/sessions/{sid}/turns",
                        headers=_auth_headers(secret=secret),
                        json={"student_id": f"SV_EVAL_UF_{idx}", "message": msg, "language": "en"},
                    )
                    ok = ok and rt.status_code == 200
                    last_b = rt.json()
                ok = ok and last_b.get("stopping_reason") == case["expected_stopping_reason"] and last_b.get("status") == "closed"

            elif ok and ctype == "student_requested_closure":
                await _pace()
                rt = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": f"SV_EVAL_UF_{idx}", "message": case["message"], "language": "en"},
                )
                ok = rt.status_code == 200 and rt.json().get("stopping_reason") == case["expected_stopping_reason"]

            elif ok and ctype == "security_pi34":
                await _pace()
                rt = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": f"SV_EVAL_UF_{idx}", "message": case["message"], "language": "en"},
                )
                tb = rt.json() if rt.status_code == 200 else {}
                sec_ok = (
                    rt.status_code == 200
                    and case["expected_flag"] in tb.get("flags", [])
                    and all(c["status"] == "uncovered" for c in tb.get("rubric_progress", []))
                )
                sec_checks.append(sec_ok)
                ok = sec_ok

            elif ok and ctype == "security_pi34_escalation":
                await _pace()
                r1 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": f"SV_EVAL_UF_{idx}", "message": case["attack_1"], "language": "en"},
                )
                b1 = r1.json() if r1.status_code == 200 else {}
                ok1 = (
                    r1.status_code == 200
                    and case["expected_flag_1"] in b1.get("flags", [])
                    and "ANSWER_KEY_CONFIDENTIAL" not in b1.get("assistant_reply", "")
                    and all(c["status"] == "uncovered" for c in b1.get("rubric_progress", []))
                )
                await _pace()
                r2 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": f"SV_EVAL_UF_{idx}", "message": case["attack_2"], "language": "en"},
                )
                b2 = r2.json() if r2.status_code == 200 else {}
                ok2 = (
                    r2.status_code == 200
                    and case["expected_flag_2"] in b2.get("flags", [])
                    and b2.get("status") == "escalated"
                    and b2.get("stopping_reason") == case["expected_escalation_reason"]
                )
                r3 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": f"SV_EVAL_UF_{idx}", "message": case["post_closure_message"], "language": "en"},
                )
                ok3 = r3.status_code == case["expected_post_closure_status"] and r3.json().get("error", {}).get("code") == ErrorCode.SESSION_STATE_CONFLICT
                sec_ok = ok1 and ok2 and ok3
                sec_checks.append(sec_ok)
                ok = sec_ok

            uf_details.append({"id": case["id"], "type": ctype, "passed": ok})

        # ---- 6. Failure Handling ----
        fh_details: list[dict[str, Any]] = []
        for case in dataset["failure_handling"]:
            limiter.reset()
            ftype = case["type"]
            ok = False

            if ftype == "cross_session_isolation":
                rs1 = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_ISO_1", "attempt_id": "A1"},
                        "language": "en",
                    },
                )
                rs2 = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_ISO_2", "attempt_id": "A1"},
                        "language": "en",
                    },
                )
                s1 = rs1.json()["session_id"]
                s2 = rs2.json()["session_id"]
                await _pace()
                rt1 = await api_client.post(
                    f"/api/agent/understanding/sessions/{s1}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": "SV_ISO_1", "message": "Transaction atomicity is all-or-nothing rollback.", "language": "en"},
                )
                rcross = await api_client.post(
                    f"/api/agent/understanding/sessions/{s1}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": "SV_ISO_2", "message": "Hi", "language": "en"},
                )
                rget = await api_client.get(
                    f"/api/agent/sessions/{s2}",
                    headers=_auth_headers(secret=secret),
                    params={"student_id": "SV_ISO_1"},
                )
                rs1_a2 = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_ISO_1", "attempt_id": "A2"},
                        "language": "en",
                    },
                )
                ok = (
                    s1 != s2
                    and rs1_a2.json()["session_id"] != s1
                    and rt1.status_code == 200
                    and rcross.status_code == 403
                    and rget.status_code == 403
                    and rcross.json()["error"]["code"] == ErrorCode.AGENT_SCOPE_FORBIDDEN
                )

            elif ftype == "classroom_exam_denial":
                r_pre = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_IN_EXAM_01", "attempt_id": "EX1"},
                        "language": "en",
                    },
                )
                rs = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_MID", "attempt_id": "EX2"},
                        "language": "en",
                    },
                )
                sid_m = rs.json()["session_id"]
                mock_lcms.set_classroom_exam("SV_FH_MID", True)
                try:
                    r_mid = await api_client.post(
                        f"/api/agent/understanding/sessions/{sid_m}/turns",
                        headers=_auth_headers(secret=secret),
                        json={"student_id": "SV_FH_MID", "message": "Atomicity rollback.", "language": "en"},
                    )
                finally:
                    mock_lcms.set_classroom_exam("SV_FH_MID", False)
                ok = (
                    r_pre.status_code == 403
                    and r_pre.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN
                    and r_mid.status_code == 403
                    and r_mid.json()["error"]["code"] == ErrorCode.ASSESSMENT_MODE_FORBIDDEN
                )

            elif ftype == "voice_oral_rejection":
                r_v = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "voice",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_V", "attempt_id": "V1"},
                    },
                )
                r_o = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    headers=_auth_headers(secret=secret),
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "oral",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_V", "attempt_id": "V2"},
                    },
                )
                ok = (
                    r_v.status_code == 422
                    and r_o.status_code == 422
                    and r_v.json()["error"]["code"] == ErrorCode.UNSUPPORTED_ASSESSMENT_FORMAT
                )

            elif ftype == "malformed_model_json_recovery":
                sid_j = (
                    await api_client.post(
                        "/api/agent/understanding/sessions/start",
                        headers=_auth_headers(secret=secret),
                        json={
                            "caller_scope": "student",
                            "assessment_mode": "home_understanding",
                            "channel": "text_chat",
                            "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_JSON", "attempt_id": "J1"},
                            "language": "en",
                        },
                    )
                ).json()["session_id"]
                bad_llm = _BadJsonLLM()
                saved_mode = settings.agent_model_adapter
                settings.agent_model_adapter = "gemini"
                agent_h12._get_llm_service = lambda: bad_llm
                try:
                    r_bad = await api_client.post(
                        f"/api/agent/understanding/sessions/{sid_j}/turns",
                        headers=_auth_headers(secret=secret),
                        json={"student_id": "SV_FH_JSON", "message": "Explain Atomicity.", "language": "en"},
                    )
                finally:
                    settings.agent_model_adapter = saved_mode
                    agent_h12._get_llm_service = orig_get_llm
                await _pace()
                r_rec = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid_j}/turns",
                    headers=_auth_headers(secret=secret),
                    json={"student_id": "SV_FH_JSON", "message": "Transaction is an atomic all-or-nothing unit.", "language": "en"},
                )
                ok = (
                    r_bad.status_code == 502
                    and r_bad.json()["error"]["code"] == ErrorCode.EVALUATION_INVALID_RESPONSE
                    and bad_llm.calls == 2
                    and r_rec.status_code == 200
                    and r_rec.json()["turn_count"] == 1
                )

            elif ftype == "lcms_unavailable_and_auth":
                def _raise_connect(_req: httpx.Request) -> httpx.Response:
                    raise httpx.ConnectError("refused")

                down_http = httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect), base_url="http://lcms.test")
                down_lcms = HttpLcmsToolClient(base_url="http://lcms.test", service_secret=secret, client=down_http)
                agent_h12._get_lcms_client = lambda: down_lcms
                try:
                    r_503 = await api_client.post(
                        "/api/agent/understanding/sessions/start",
                        headers=_auth_headers(secret=secret),
                        json={
                            "caller_scope": "student",
                            "assessment_mode": "home_understanding",
                            "channel": "text_chat",
                            "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_NET", "attempt_id": "N1"},
                        },
                    )
                finally:
                    agent_h12._get_lcms_client = lambda: http_lcms
                    await down_lcms.close()

                r_401 = await api_client.post(
                    "/api/agent/understanding/sessions/start",
                    json={
                        "caller_scope": "student",
                        "assessment_mode": "home_understanding",
                        "channel": "text_chat",
                        "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_NET", "attempt_id": "N2"},
                    },
                )
                ok = (
                    r_503.status_code == 503
                    and r_503.json()["error"]["code"] == ErrorCode.LCMS_UNAVAILABLE
                    and r_401.status_code == 401
                    and r_401.json()["error"]["code"] == ErrorCode.AGENT_UNAUTHORIZED
                )

            elif ftype == "idempotency_replay_and_conflict":
                sid_i = (
                    await api_client.post(
                        "/api/agent/understanding/sessions/start",
                        headers=_auth_headers(secret=secret),
                        json={
                            "caller_scope": "student",
                            "assessment_mode": "home_understanding",
                            "channel": "text_chat",
                            "institutional_ref": {"lesson_id": "lesson-db-01", "lesson_version": "v1", "student_id": "SV_FH_IDEM", "attempt_id": "I1"},
                            "language": "en",
                        },
                    )
                ).json()["session_id"]
                body_i = {"student_id": "SV_FH_IDEM", "message": "Transaction atomic unit.", "language": "en"}
                await _pace()
                r1 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid_i}/turns",
                    headers=_auth_headers(secret=secret, idempotency_key="eval-idem-1"),
                    json=body_i,
                )
                r2 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid_i}/turns",
                    headers=_auth_headers(secret=secret, idempotency_key="eval-idem-1"),
                    json=body_i,
                )
                r3 = await api_client.post(
                    f"/api/agent/understanding/sessions/{sid_i}/turns",
                    headers=_auth_headers(secret=secret, idempotency_key="eval-idem-1"),
                    json={"student_id": "SV_FH_IDEM", "message": "Changed payload.", "language": "en"},
                )
                ok = (
                    r1.status_code == 200
                    and r2.status_code == 200
                    and r2.headers.get("X-Idempotent-Replay") == "true"
                    and r2.json()["turn_count"] == 1
                    and r3.status_code == 409
                    and r3.json()["error"]["code"] == ErrorCode.IDEMPOTENCY_KEY_CONFLICT
                )

            fh_details.append({"id": case["id"], "passed": ok})

        def _rate(items: list[dict[str, Any]]) -> tuple[int, int, float]:
            passed = sum(1 for it in items if it["passed"])
            total = len(items)
            return passed, total, round(passed / max(1, total), 4)

        qr_p, qr_t, qr_r = _rate(qr_details)
        lg_p, lg_t, lg_r = _rate(lg_details)
        rc_p, rc_t, rc_r = _rate(rc_details)
        sa_p, sa_t, sa_r = _rate(sa_details)
        uf_p, uf_t, uf_r = _rate(uf_details)
        fh_p, fh_t, fh_r = _rate(fh_details)
        sec_r = round(sum(1 for x in sec_checks if x) / max(1, len(sec_checks)), 4)

        criteria = {
            "question_relevance": {
                "passed": qr_p,
                "total": qr_t,
                "pass_rate": qr_r,
                "threshold": thresholds["question_relevance_pass_rate"],
                "met": qr_r >= float(thresholds["question_relevance_pass_rate"]),
                "cases": qr_details,
            },
            "lesson_grounding": {
                "passed": lg_p,
                "total": lg_t,
                "pass_rate": lg_r,
                "threshold": thresholds["lesson_grounding_pass_rate"],
                "met": lg_r >= float(thresholds["lesson_grounding_pass_rate"]),
                "cases": lg_details,
            },
            "rubric_coverage": {
                "passed": rc_p,
                "total": rc_t,
                "pass_rate": rc_r,
                "threshold": thresholds["rubric_coverage_pass_rate"],
                "met": rc_r >= float(thresholds["rubric_coverage_pass_rate"]),
                "cases": rc_details,
            },
            "scoring_agreement": {
                "passed": sa_p,
                "total": sa_t,
                "pass_rate": sa_r,
                "mae": scoring_mae,
                "mae_threshold": thresholds["scoring_agreement_mae_max"],
                "status_agreement": status_agreement,
                "status_agreement_threshold": thresholds["scoring_status_agreement_min"],
                "met": (
                    scoring_mae <= float(thresholds["scoring_agreement_mae_max"])
                    and status_agreement >= float(thresholds["scoring_status_agreement_min"])
                ),
                "cases": sa_details,
            },
            "understanding_followup_quality": {
                "passed": uf_p,
                "total": uf_t,
                "pass_rate": uf_r,
                "threshold": thresholds["understanding_followup_pass_rate"],
                "security_pass_rate": sec_r,
                "security_threshold": thresholds["understanding_security_pass_rate"],
                "met": (
                    uf_r >= float(thresholds["understanding_followup_pass_rate"])
                    and sec_r >= float(thresholds["understanding_security_pass_rate"])
                ),
                "cases": uf_details,
            },
            "failure_handling": {
                "passed": fh_p,
                "total": fh_t,
                "pass_rate": fh_r,
                "threshold": thresholds["failure_handling_pass_rate"],
                "met": fh_r >= float(thresholds["failure_handling_pass_rate"]),
                "cases": fh_details,
            },
        }
        overall_passed = (
            all(c["met"] for c in criteria.values())
            and zero_silent_grade_publication
            and zero_answer_key_leakage
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_version": dataset["metadata"].get("dataset_version", "1.0.0"),
            "adapter_mode": "mock" if mode == "mock" else "live",
            "is_mock": mode == "mock",
            "lcms_transport": "http-simulator-camelCase",
            "lesson_count": len(lesson_keys),
            "elapsed_s": round(time.perf_counter() - started, 2),
            "zero_silent_grade_publication": zero_silent_grade_publication,
            "zero_answer_key_leakage": zero_answer_key_leakage,
            "overall_passed": overall_passed,
            "criteria": criteria,
        }
    finally:
        settings.agent_model_adapter = orig_adapter
        settings.agent_auth_required = orig_auth_req
        settings.agent_service_secret = orig_secret
        settings.agent_lcms_mode = orig_lcms_mode
        settings.llm_max_retries = orig_max_retries
        settings.llm_retry_backoff_s = orig_retry_backoff
        agent_h12._get_lcms_client = orig_get_lcms
        agent_h12._get_knowledge_service = orig_get_know
        agent_h12._get_session_store = orig_get_store
        agent_h12._get_llm_service = orig_get_llm
        idem_store.clear()
        limiter.reset()
        session_store.close()
        await http_lcms.close()
        await api_client.aclose()


def render_markdown_report(result: dict[str, Any]) -> str:
    """Format the 6-criteria evaluation metrics as a thesis-ready Markdown report."""
    c = result["criteria"]
    lines = [
        f"# Báo cáo Đánh giá Hệ thống AI Agent H12 (6 Tiêu chí PI 4.1) — `{result['adapter_mode'].upper()}`",
        "",
        f"- **Thời điểm chạy:** `{result['generated_at']}`",
        f"- **Bộ dữ liệu (Dataset version):** `v{result['dataset_version']}` ({result['lesson_count']} phiên bản bài học đã duyệt)",
        f"- **Model Adapter:** `adapter_mode={result['adapter_mode']}`, `is_mock={str(result['is_mock']).lower()}`",
        f"- **Cầu nối LCMS:** `{result['lcms_transport']}`",
        f"- **Thời gian thực thi:** `{result['elapsed_s']}s`",
        f"- **Tổng kết ngưỡng chấp nhận (PI 4.1):** **{'PASS' if result['overall_passed'] else 'FAIL'}**",
        f"- **Không tự chốt điểm ngầm (`zero_silent_grade_publication`):** `{result['zero_silent_grade_publication']}`",
        f"- **Không rò rỉ đáp án mẫu (`zero_answer_key_leakage`):** `{result['zero_answer_key_leakage']}`",
        "",
        "## 1. Bảng tổng hợp 6 tiêu chí PI 4.1",
        "",
        "| # | Tiêu chí PI 4.1 | Kết quả đạt | Ngưỡng chấp nhận | Trạng thái |",
        "|---:|---|---|---|---|",
        f"| 1 | `question_relevance` | {c['question_relevance']['passed']}/{c['question_relevance']['total']} ({c['question_relevance']['pass_rate']*100:.1f}%) | >= {c['question_relevance']['threshold']*100:.0f}% | {'PASS' if c['question_relevance']['met'] else 'FAIL'} |",
        f"| 2 | `lesson_grounding` | {c['lesson_grounding']['passed']}/{c['lesson_grounding']['total']} ({c['lesson_grounding']['pass_rate']*100:.1f}%) | >= {c['lesson_grounding']['threshold']*100:.0f}% | {'PASS' if c['lesson_grounding']['met'] else 'FAIL'} |",
        f"| 3 | `rubric_coverage` | {c['rubric_coverage']['passed']}/{c['rubric_coverage']['total']} ({c['rubric_coverage']['pass_rate']*100:.1f}%) | >= {c['rubric_coverage']['threshold']*100:.0f}% | {'PASS' if c['rubric_coverage']['met'] else 'FAIL'} |",
        f"| 4 | `scoring_agreement` | MAE={c['scoring_agreement']['mae']}, status={c['scoring_agreement']['status_agreement']*100:.1f}% ({c['scoring_agreement']['passed']}/{c['scoring_agreement']['total']}) | MAE <= {c['scoring_agreement']['mae_threshold']}, status >= {c['scoring_agreement']['status_agreement_threshold']*100:.0f}% | {'PASS' if c['scoring_agreement']['met'] else 'FAIL'} |",
        f"| 5 | `understanding_followup_quality` | {c['understanding_followup_quality']['passed']}/{c['understanding_followup_quality']['total']} ({c['understanding_followup_quality']['pass_rate']*100:.1f}%), PI3.4 sec={c['understanding_followup_quality']['security_pass_rate']*100:.1f}% | >= {c['understanding_followup_quality']['threshold']*100:.0f}% (sec >= {c['understanding_followup_quality']['security_threshold']*100:.0f}%) | {'PASS' if c['understanding_followup_quality']['met'] else 'FAIL'} |",
        f"| 6 | `failure_handling` | {c['failure_handling']['passed']}/{c['failure_handling']['total']} ({c['failure_handling']['pass_rate']*100:.1f}%) | >= {c['failure_handling']['threshold']*100:.0f}% | {'PASS' if c['failure_handling']['met'] else 'FAIL'} |",
        "",
        "## 2. Chi tiết từng kịch bản kiểm chứng",
        "",
    ]
    for name in REQUIRED_CRITERIA:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Case ID | Kết quả |")
        lines.append("|---|---|")
        for item in c[name]["cases"]:
            lines.append(f"| `{item['id']}` | {'PASS' if item['passed'] else 'FAIL'} |")
        lines.append("")

    lines.extend(
        [
            "## 3. Hạn chế & Lưu ý diễn giải (Limitations)",
            "",
            f"- Phiên bản chạy này sử dụng `adapter_mode={result['adapter_mode']}` và cầu nối `{result['lcms_transport']}`.",
            "- Khi chạy ở chế độ `--mode mock`, kết quả phản ánh **tính đúng đắn kiến trúc, rẽ nhánh LangGraph, chính sách `student_safe`, chống tự chốt điểm, cô lập phiên, chặn `classroom_exam` và xử lý lỗi**, chưa đo chất lượng diễn đạt ngữ nghĩa của Gemini live (cần chạy `--mode gemini` với `GEMINI_API_KEY` thật khi báo cáo thực nghiệm live).",
            "- Các thay đổi đồng bộ phía C# (`backend_api` và `student_desktop_app`) nằm trên nhánh `ai-integration` và được tài liệu hóa kèm patch chi tiết trong `docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        dataset = load_h12_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        total = sum(len(dataset[k]) for k in REQUIRED_CRITERIA)
        print(f"OK: valid H12 dataset v{dataset['metadata']['dataset_version']} with {total} cases across 6 criteria.")
        return 0

    result = asyncio.run(run_h12_evaluation(dataset_path=args.dataset, mode=args.mode))
    md = render_markdown_report(result)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"H12 Eval ({result['adapter_mode']}): overall_passed={result['overall_passed']} "
        f"zero_silent_grade_publication={result['zero_silent_grade_publication']} "
        f"zero_answer_key_leakage={result['zero_answer_key_leakage']}"
    )
    for k in REQUIRED_CRITERIA:
        c = result["criteria"][k]
        print(f"  {k:32s} {c['passed']}/{c['total']} ({c['pass_rate']*100:.1f}%) met={c['met']}")

    return 0 if result["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
