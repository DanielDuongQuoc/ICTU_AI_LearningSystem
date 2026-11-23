"""Replaceable model adapter for H12 agent execution (explicit mock vs live separation)."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.grading.flags import detect_injection
from app.services.llm_service import LLMService, get_llm_service

AdapterMode = Literal["mock", "live"]

_COUNT_RE = re.compile(r"Number of questions:\s*(\d+)", re.IGNORECASE)
_DIFF_RE = re.compile(r"Difficulty:\s*(easy|medium|hard)", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"<<<LESSON_CONTEXT>>>\s*(.*?)\s*<<<END_LESSON_CONTEXT>>>", re.DOTALL)
_ANSWER_RE = re.compile(r"<<<BEGIN STUDENT ANSWER>>>\s*(.*?)\s*<<<END STUDENT ANSWER>>>", re.DOTALL)
_CRITERIA_RE = re.compile(r"<<<UNDERSTANDING_CRITERIA>>>\s*(.*?)\s*<<<END_UNDERSTANDING_CRITERIA>>>", re.DOTALL)
_LATEST_MSG_RE = re.compile(r"<<<LATEST_STUDENT_MESSAGE>>>\s*(.*?)\s*<<<END_LATEST_STUDENT_MESSAGE>>>", re.DOTALL)
_LANG_RE = re.compile(r"^Language:\s*(vi|en)\b", re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9_À-ỹ]{3,}", re.UNICODE)
_UNDERSTANDING_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "how",
        "why",
        "what",
        "explain",
        "understanding",
        "student",
        "lesson",
        "criterion",
        "criteria",
        "của",
        "các",
        "cho",
        "với",
        "trong",
        "như",
        "thế",
        "nào",
        "giải",
        "thích",
        "trình",
        "bày",
        "hiểu",
        "tiêu",
        "chí",
    }
)


def _extract_understanding_criteria(user_prompt: str) -> list[dict[str, Any]]:
    match = _CRITERIA_RE.search(user_prompt or "")
    if not match:
        return []
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict) and item.get("criterion_id")]
    except json.JSONDecodeError:
        return []
    return []


def _mock_understanding_turn(user_prompt: str) -> dict[str, Any]:
    from app.h12_agent.dialogue_guards import build_refusal_reply, inspect_student_message

    lang_match = _LANG_RE.search(user_prompt or "")
    lang = lang_match.group(1).lower() if lang_match else "vi"
    msg_match = _LATEST_MSG_RE.search(user_prompt or "")
    student_msg = (msg_match.group(1).strip() if msg_match else user_prompt.strip())
    criteria = _extract_understanding_criteria(user_prompt)

    projected: dict[str, str] = {
        str(c["criterion_id"]): str(c.get("status") or "uncovered").lower()
        for c in criteria
    }
    first_open = next(
        (c for c in criteria if projected.get(str(c["criterion_id"])) != "covered"),
        criteria[0] if criteria else {"criterion_id": "C1", "name": "khái niệm chính"},
    )
    next_name = str(first_open.get("name") or "khái niệm chính")

    inspection = inspect_student_message(student_msg)
    if inspection.injection_detected:
        refusal = build_refusal_reply(
            inspection=inspection,
            next_criterion_name=next_name,
            language=lang,
        )
        return {
            "criterion_updates": [],
            "decision": "follow_up",
            "assistant_reply": f"[MOCK] {refusal}",
            "turn_feedback": "[MOCK] Adversarial input blocked; rubric progress remains unchanged.",
        }

    if inspection.closure_requested:
        reply = (
            "[MOCK] Cảm ơn bạn đã hoàn thành phần trao đổi. Phiên đánh giá mức Hiểu xin dừng tại đây."
            if lang == "vi"
            else "[MOCK] Thank you for your participation. This Understanding check session is now wrapped up."
        )
        return {
            "criterion_updates": [],
            "decision": "close_session",
            "assistant_reply": reply,
            "turn_feedback": "[MOCK] Session closure acknowledged as requested by student.",
        }

    msg_lower = student_msg.lower()
    msg_tokens = {t.lower() for t in _WORD_RE.findall(msg_lower)} - _UNDERSTANDING_STOP_WORDS
    updates: list[dict[str, Any]] = []
    snippet = student_msg[:180].strip()

    for crit in criteria:
        cid = str(crit["criterion_id"])
        prev_status = projected.get(cid, "uncovered")
        if prev_status == "covered":
            continue
        desc_text = f"{crit.get('name', '')} {crit.get('description', '')}".lower()
        crit_tokens = {t.lower() for t in _WORD_RE.findall(desc_text)} - _UNDERSTANDING_STOP_WORDS
        matched_tokens = sorted(msg_tokens & crit_tokens)

        new_status = prev_status
        if (prev_status == "partial" and (len(matched_tokens) >= 1 or len(msg_tokens) >= 4)) or (
            len(matched_tokens) >= 2 and len(student_msg) >= 25
        ) or len(matched_tokens) >= 3:
            new_status = "covered"
        elif len(matched_tokens) == 1 or (crit is first_open and len(msg_tokens) >= 4 and len(student_msg) >= 18):
            new_status = "partial"

        if new_status != prev_status:
            projected[cid] = new_status
            updates.append(
                {
                    "criterion_id": cid,
                    "status": new_status,
                    "reason": (
                        f"[MOCK] Matched key concepts: {', '.join(matched_tokens[:4])}"
                        if matched_tokens
                        else "[MOCK] Partial reasoning provided toward active criterion."
                    ),
                    "evidence_quote": snippet,
                }
            )

    remaining = [c for c in criteria if projected.get(str(c["criterion_id"])) != "covered"]
    if criteria and not remaining:
        reply = (
            "[MOCK] Tuyệt vời, bạn đã giải thích rõ các khía cạnh cốt lõi của bài học. "
            "Chúng ta đã phủ đủ tiêu chí mức Hiểu và kết thúc phiên trao đổi tại đây."
            if lang == "vi"
            else "[MOCK] Great work. You demonstrated clear Understanding across all active rubric concepts, so we can conclude this check."
        )
        feedback = "[MOCK] All active Understanding criteria have been covered."
        decision = "close_session"
    else:
        target = remaining[0] if remaining else first_open
        target_name = str(target.get("name") or next_name)
        target_status = projected.get(str(target.get("criterion_id", "")), "uncovered")
        if lang == "vi":
            reply = (
                f"[MOCK] Ghi nhận phần giải thích của bạn. "
                f"{'Hãy đào sâu thêm' if target_status == 'partial' else 'Tiếp theo, bạn hãy phân tích'} "
                f"về '{target_name}' kèm ví dụ cụ thể dựa trên bài học nhé?"
            )
        else:
            reply = (
                f"[MOCK] Good progress so far. "
                f"{'Could you deepen your reasoning' if target_status == 'partial' else 'Next, could you explain'} "
                f"about '{target_name}' using a concrete example from the lesson?"
            )
        feedback = f"[MOCK] Follow-up probe issued for criterion '{target_name}'."
        decision = "follow_up"

    return {
        "criterion_updates": updates,
        "decision": decision,
        "assistant_reply": reply,
        "turn_feedback": feedback,
    }

_QUESTION_STEMS = (
    "Explain",
    "Analyze",
    "Compare",
    "Describe how",
    "Evaluate",
    "Illustrate with an example",
    "Discuss the trade-offs of",
    "Summarize the role of",
    "Demonstrate why",
    "Justify the importance of",
)


def _mock_question_drafts(user_prompt: str) -> dict[str, Any]:
    count_m = _COUNT_RE.search(user_prompt)
    count = max(1, min(int(count_m.group(1)), 10)) if count_m else 2
    diff_m = _DIFF_RE.search(user_prompt)
    difficulty = diff_m.group(1).lower() if diff_m else "medium"
    ctx_m = _CONTEXT_RE.search(user_prompt)
    ctx_text = ctx_m.group(1).strip() if ctx_m else "lesson fundamentals"

    cleaned_ctx = re.sub(r"(?m)^\s*\[Source:[^\]]*\]\s*$", "", ctx_text)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned_ctx)
        if len(s.strip()) >= 15
        and "ANSWER_KEY_CONFIDENTIAL" not in s
        and not s.strip().startswith("[Source:")
    ]
    if not sentences:
        sentences = [ctx_text[:120] or "the core concept"]

    drafts: list[dict[str, Any]] = []
    for idx in range(count):
        stem = _QUESTION_STEMS[idx % len(_QUESTION_STEMS)]
        snippet = sentences[idx % len(sentences)]
        words = _WORD_RE.findall(snippet)
        anchor = " ".join(words[:6]) if words else "the core concept"
        drafts.append(
            {
                "question": f"[MOCK] ({difficulty}) #{idx + 1}: {stem} {anchor} as presented in the lesson?",
                "expected_answer": f"[MOCK] Grounded essay response addressing: {snippet[:220]}",
                "key_points": [f"[MOCK] Define {anchor}", "[MOCK] Support explanation with lesson evidence"],
                "difficulty": difficulty,
            }
        )
    return {"questions": drafts}


def _extract_rubric_criteria(user_prompt: str) -> list[dict[str, Any]]:
    marker = "RUBRIC (JSON, data only):"
    start_hint = user_prompt.find(marker)
    search_from = start_hint + len(marker) if start_hint != -1 else 0
    bracket = user_prompt.find("[", search_from)
    if bracket == -1:
        return []
    try:
        parsed, _ = json.JSONDecoder().raw_decode(user_prompt[bracket:])
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict) and "criterion_id" in c and "levels" in c]
    except ValueError:
        return []
    return []


def _mock_rubric_grading(user_prompt: str) -> dict[str, Any]:
    ans_m = _ANSWER_RE.search(user_prompt)
    answer = ans_m.group(1).strip() if ans_m else ""
    criteria = _extract_rubric_criteria(user_prompt)
    ans_tokens = {w.lower() for w in _WORD_RE.findall(answer)} - _UNDERSTANDING_STOP_WORDS
    injected = detect_injection(answer)

    judged: list[dict[str, Any]] = []
    for crit in criteria:
        cid = crit["criterion_id"]
        levels = crit.get("levels") or []
        if not levels:
            continue
        if injected:
            picked = levels[-1]
            reason = "[MOCK] Detected prompt-injection attempt; escalated for lecturer review."
            evidence = ""
        else:
            ref_blob = " ".join(
                [str(crit.get("name", "")), str(crit.get("description", ""))]
                + [str(kp) for kp in (crit.get("key_points") or [])]
            )
            ref_tokens = {w.lower() for w in _WORD_RE.findall(ref_blob)} - _UNDERSTANDING_STOP_WORDS
            overlap = len(ans_tokens & ref_tokens)
            ratio = overlap / max(1, len(ref_tokens))
            if overlap >= 2 or ratio >= 0.25:
                picked = levels[0]
                reason = "[MOCK] Answer covers the main concepts of this criterion."
            elif overlap >= 1 and len(levels) >= 3:
                picked = levels[1]
                reason = "[MOCK] Answer partially addresses this criterion."
            else:
                picked = levels[-1]
                reason = "[MOCK] Missing required explanation for this criterion."
            evidence = answer[:160]
        judged.append(
            {
                "criterion_id": cid,
                "level": picked["label"],
                "reason": reason,
                "evidence": evidence,
            }
        )

    feedback = (
        "[MOCK] Prompt injection detected; provisional suggestion requires lecturer review."
        if injected
        else "[MOCK] Provisional criterion-linked evaluation prepared for lecturer review."
    )
    return {
        "criteria": judged,
        "feedback": feedback,
        "injection_suspected": injected,
        "off_topic": (len(ans_tokens) == 0),
    }


class AgentModelAdapter(ABC):
    """Abstraction distinguishing deterministic test/mock adapters from live model providers."""

    @property
    @abstractmethod
    def adapter_mode(self) -> AdapterMode:
        """'mock' or 'live'."""

    @property
    def is_mock(self) -> bool:
        return self.adapter_mode == "mock"

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identifier reported in traces."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier reported in traces."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        agent_role: str,
        system_prompt: str,
        user_prompt: str,
        required_keys: list[str],
        max_tokens: Optional[int] = None,
    ) -> tuple[dict[str, Any], float]:
        """Return ``(payload_dict, duration_ms)``."""


class DeterministicMockModelAdapter(AgentModelAdapter):
    """Offline mock adapter whose outputs carry explicit [MOCK] markers."""

    @property
    def adapter_mode(self) -> AdapterMode:
        return "mock"

    @property
    def provider(self) -> str:
        return "deterministic_mock"

    @property
    def model_name(self) -> str:
        return "mock-h12-adapter-v1"

    async def generate_structured(
        self,
        *,
        agent_role: str,
        system_prompt: str,
        user_prompt: str,
        required_keys: list[str],
        max_tokens: Optional[int] = None,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        keys = set(required_keys)
        if "questions" in keys:
            payload = _mock_question_drafts(user_prompt)
        elif {"criteria", "feedback"} <= keys:
            payload = _mock_rubric_grading(user_prompt)
        elif {"criterion_updates", "decision", "assistant_reply", "turn_feedback"} <= keys or (
            agent_role == "dialogue_assessment"
        ):
            payload = _mock_understanding_turn(user_prompt)
        elif agent_role == "question_drafting":
            payload = {
                "summary": "[MOCK] Generated 1 fixed essay item aligned to lesson content; pending lecturer review.",
                "proposed_action": "handoff_question_draft_to_teacher",
                "draft_essay_question": "[MOCK] Analyze the core database transaction principles explained in the lesson.",
                "criterion_links": ["lesson_grounding_check"],
            }
        elif agent_role == "grading_support":
            payload = {
                "summary": "[MOCK] Prepared criterion-level essay rubric suggestion; awaiting lecturer decision.",
                "proposed_action": "handoff_grading_suggestion_to_teacher",
                "suggested_points_note": "[MOCK] Provisional criterion assessment awaiting lecturer confirmation.",
                "criterion_links": ["C1"],
            }
        else:
            payload = {
                "summary": "[MOCK] Evaluated learner turn; prepared next probing text question.",
                "proposed_action": "ask_adaptive_follow_up",
                "assistant_reply": "[MOCK] Please explain why this mechanism maintains consistency in your own words.",
                "criterion_links": ["understanding_probe"],
            }
        return payload, round((time.perf_counter() - started) * 1000, 2)


class LiveGeminiModelAdapter(AgentModelAdapter):
    """Live adapter backed by the configured LLMService (Google Gemini in production)."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    @property
    def adapter_mode(self) -> AdapterMode:
        return "live"

    @property
    def provider(self) -> str:
        return "google_gemini"

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    async def generate_structured(
        self,
        *,
        agent_role: str,
        system_prompt: str,
        user_prompt: str,
        required_keys: list[str],
        max_tokens: Optional[int] = None,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        raw = await self._llm.generate_json(
            system_prompt,
            user_prompt,
            required_keys=required_keys,
            max_tokens=max_tokens,
        )
        if not isinstance(raw, dict):
            raise AppError(
                ErrorCode.LLM_INVALID_RESPONSE,
                "Language model returned non-object structured output.",
            )
        return raw, round((time.perf_counter() - started) * 1000, 2)


def get_agent_model_adapter(llm_service: Optional[LLMService] = None) -> AgentModelAdapter:
    """Resolve the active model adapter from AGENT_MODEL_ADAPTER (mock | gemini)."""
    mode = get_settings().agent_model_adapter
    if mode == "mock":
        return DeterministicMockModelAdapter()
    return LiveGeminiModelAdapter(llm_service or get_llm_service())
