"""Security guards and output leak prevention for H12 Understanding dialogue (Milestone 4, PI 3.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.core.exceptions import AppError, ErrorCode
from app.core.output_guard import SHINGLE_WORDS, clamp_text, enforce_no_prompt_leak
from app.grading.flags import detect_injection

MAX_DIALOGUE_REPLY_CHARS = 1800

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_ANSWER_REVEAL_PATTERNS = (
    re.compile(
        r"(?:reveal|show|give|tell|provide|leak|print|dump|share)\b.{0,45}"
        r"\b(?:answer\s*key|correct\s*answer|expected\s*answer|model\s*answer|official\s*answer|solution|teacher\s*notes?)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bjust\s+give\s+me\s+(?:the\s+)?answer\b", re.IGNORECASE),
    re.compile(
        r"(?:cho\s+(?:tôi|mình|em)\s+(?:xem|biết)|tiết\s+lộ|gửi\s+luôn|nói\s+luôn).{0,40}"
        r"(?:đáp\s*án|lời\s*giải|bài\s*mẫu|barem|ghi\s*chú\s*giáo\s*viên)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bđáp\s*án\s*(?:đúng|chính\s*thức)\s*là\s*gì\b", re.IGNORECASE),
)

_REDIRECT_PATTERNS = (
    re.compile(
        r"(?:ignore|forget|disregard|bypass|override).{0,35}"
        r"(?:previous|prior|above|system|rubric|lesson|rules?|instructions?)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|switch\s+(?:role|topic)\s+to|"
        r"talk\s+about\s+something\s+else|write\s+(?:me\s+)?a\s+(?:poem|joke|story|code))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bỏ\s+qua|quên\s+đi|vượt\s+qua).{0,35}(?:hướng\s*dẫn|quy\s*tắc|bài\s*học|rubric|chỉ\s*thị)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?:đóng\s+vai|chuyển\s+sang\s+chủ\s+đề\s+khác)", re.IGNORECASE),
)

_SCORE_ALTER_PATTERNS = (
    re.compile(
        r"(?:give|set|mark|assign|grant|award|override|alter|change|boost|inflate|update).{0,40}"
        r"(?:full\s*points?|max(?:imum)?\s*score|10\s*/\s*10|100\s*%|all\s+criteri(?:on|a)|my\s+score|pass\s+grade)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(?:mark|set)\s+(?:all|every)\s+criteri(?:on|a)\s+(?:as\s+)?met\b", re.IGNORECASE),
    re.compile(
        r"(?:chấm|cho|sửa|tăng|đặt|cập\s*nhật).{0,40}"
        r"(?:tối\s*đa|10\s*điểm|đạt\s*hết|toàn\s*bộ\s*tiêu\s*chí|điểm\s*tuyệt\s*đối)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\boverride\s+(?:the\s+)?(?:score|grade|rubric)\b", re.IGNORECASE),
)

_CLOSURE_PATTERNS = (
    re.compile(r"^\s*(?:stop|end|quit|exit|finish|done|bye|kết\s*thúc|dừng\s*lại|tạm\s*dừng|xong\s*rồi)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(
        r"\b(?:i\s+(?:want|would\s+like)\s+to\s+(?:stop|end|finish|close)|"
        r"please\s+(?:end|close|stop)\s+(?:this|the)\s+(?:session|dialogue|chat|assessment)|"
        r"(?:end|stop)\s+(?:this|the)\s+session\s+now)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:tôi|em|mình)\s+muốn\s+(?:dừng|kết\s*thúc)|"
        r"kết\s*thúc\s+(?:phiên|buổi\s+hội\s+thoại)|"
        r"dừng\s+(?:phiên|hội\s+thoại)\s+(?:tại\s+đây|này))",
        re.IGNORECASE,
    ),
)

_ANSWER_LEAK_REGEXES = (
    re.compile(r"\b(?:official|confidential|model)\s+answer\s+key\b\s*(?::|\bis\b)", re.IGNORECASE),
    re.compile(r"\bđáp\s*án\s*chính\s*thức\b\s*(?::|\blà\b)", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class SecurityInspection:
    """Result of inspecting a student Understanding turn message."""

    flags: list[str]
    injection_detected: bool
    redirect_attempt: bool
    answer_reveal_attempt: bool
    score_alter_attempt: bool
    closure_requested: bool
    block_rubric_progress: bool


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    cleaned = _SPACE_RE.sub(" ", text or "").strip()
    if not cleaned:
        return False
    return any(p.search(cleaned) for p in patterns)


def detect_answer_reveal_attempt(text: str) -> bool:
    """Return True if the message asks to reveal teacher answer keys or model answers."""
    return _matches_any(text, _ANSWER_REVEAL_PATTERNS)


def detect_redirect_attempt(text: str) -> bool:
    """Return True if the message attempts to hijack role, ignore instructions, or redirect topic."""
    return _matches_any(text, _REDIRECT_PATTERNS)


def detect_score_alter_attempt(text: str) -> bool:
    """Return True if the message tries to override scores or force rubric criteria to pass."""
    return _matches_any(text, _SCORE_ALTER_PATTERNS)


def detect_student_closure_request(text: str) -> bool:
    """Return True if the student explicitly asks to conclude the dialogue session."""
    return _matches_any(text, _CLOSURE_PATTERNS)


def inspect_student_message(text: str) -> SecurityInspection:
    """Classify security threats and explicit closure requests on a student message."""
    base_injection = bool(detect_injection(text))
    redirect_attempt = detect_redirect_attempt(text)
    answer_reveal_attempt = detect_answer_reveal_attempt(text)
    score_alter_attempt = detect_score_alter_attempt(text)
    closure_requested = detect_student_closure_request(text)

    flags: list[str] = []
    if base_injection:
        flags.append("PROMPT_INJECTION")
    if redirect_attempt:
        flags.append("REDIRECT_ATTEMPT")
    if answer_reveal_attempt:
        flags.append("ANSWER_REVEAL_ATTEMPT")
    if score_alter_attempt:
        flags.append("SCORE_ALTER_ATTEMPT")

    injection_detected = bool(flags)
    return SecurityInspection(
        flags=flags,
        injection_detected=injection_detected,
        redirect_attempt=redirect_attempt,
        answer_reveal_attempt=answer_reveal_attempt,
        score_alter_attempt=score_alter_attempt,
        closure_requested=closure_requested,
        block_rubric_progress=injection_detected,
    )


def build_refusal_reply(
    *,
    inspection: SecurityInspection,
    next_criterion_name: str,
    language: str,
) -> str:
    """Return a neutral Socratic refusal that stays within approved lesson boundaries."""
    is_vi = language == "vi"
    if inspection.answer_reveal_attempt:
        if is_vi:
            return (
                "Mình không thể cung cấp đáp án mẫu hay tài liệu nội bộ của giảng viên. "
                f"Thay vào đó, bạn hãy giải thích bằng lời của mình về tiêu chí '{next_criterion_name}' dựa trên bài học nhé?"
            )
        return (
            "I cannot reveal official answer keys or teacher-only notes. "
            f"Could you instead explain '{next_criterion_name}' in your own words using concepts from the lesson?"
        )
    if inspection.score_alter_attempt:
        if is_vi:
            return (
                "Điểm số và đánh giá rubric chỉ dựa trên minh chứng giải thích của bạn và sẽ do giảng viên xem xét; "
                f"mình không thể thay đổi điểm theo yêu cầu. Hãy trình bày rõ hơn về '{next_criterion_name}' nhé."
            )
        return (
            "Rubric progress and provisional outcomes follow your actual explanation and remain subject to teacher review, "
            f"so scores cannot be altered directly. Please walk through '{next_criterion_name}' in your own words."
        )
    if is_vi:
        return (
            "Chúng ta cần bám sát nội dung bài học đang đánh giá mức Hiểu và không thể chuyển vai hay bỏ qua quy tắc phiên học. "
            f"Bạn hãy chia sẻ cách hiểu của mình về '{next_criterion_name}' nhé?"
        )
    return (
        "Let's stay focused on your approved lesson Understanding check rather than switching roles or rules. "
        f"How would you explain '{next_criterion_name}' based on the lesson?"
    )


def _has_prompt_shingle_overlap(reply: str, system_prompt: str) -> bool:
    prompt_tokens = _TOKEN_RE.findall((system_prompt or "").lower())
    reply_tokens = _TOKEN_RE.findall((reply or "").lower())
    if len(prompt_tokens) < SHINGLE_WORDS or len(reply_tokens) < SHINGLE_WORDS:
        return False
    prompt_shingles = {
        tuple(prompt_tokens[i : i + SHINGLE_WORDS])
        for i in range(len(prompt_tokens) - SHINGLE_WORDS + 1)
    }
    return any(
        tuple(reply_tokens[i : i + SHINGLE_WORDS]) in prompt_shingles
        for i in range(len(reply_tokens) - SHINGLE_WORDS + 1)
    )


def _safe_leak_fallback(*, next_criterion_name: str, language: str) -> str:
    if language == "vi":
        return (
            "Mình cần giữ nội dung hướng dẫn gọn theo bài học mà không tiết lộ đáp án chi tiết. "
            f"Bạn hãy thử phân tích hoặc nêu ví dụ về '{next_criterion_name}' nhé?"
        )
    return (
        "I need to avoid revealing direct answer keys during this Understanding check. "
        f"Could you explain '{next_criterion_name}' in your own words or with a concrete example?"
    )


def guard_assistant_reply(
    reply: str,
    *,
    system_prompt: str,
    forbidden_markers: Iterable[str] = (),
    next_criterion_name: str = "lesson concepts",
    language: str = "vi",
) -> tuple[str, bool]:
    """Enforce no system prompt leak and prevent leaking confidential answer-key markers.

    Returns ``(safe_reply, answer_leak_prevented)``. Raises ``AppError(LLM_OUTPUT_REJECTED)``
    when system-prompt leakage is detected.
    """
    cleaned = _SPACE_RE.sub(" ", (reply or "")).strip()
    enforce_no_prompt_leak(cleaned)
    if _has_prompt_shingle_overlap(cleaned, system_prompt):
        raise AppError(
            ErrorCode.LLM_OUTPUT_REJECTED,
            "Assistant reply leaked protected system prompt text.",
        )

    lowered = cleaned.lower()
    leak_detected = any(p.search(cleaned) for p in _ANSWER_LEAK_REGEXES)
    if not leak_detected:
        for marker in forbidden_markers:
            normalized_marker = _SPACE_RE.sub(" ", (marker or "")).strip().lower()
            if normalized_marker and normalized_marker in lowered:
                leak_detected = True
                break

    if leak_detected:
        return (
            clamp_text(
                _safe_leak_fallback(next_criterion_name=next_criterion_name, language=language),
                MAX_DIALOGUE_REPLY_CHARS,
            ),
            True,
        )

    if not cleaned:
        cleaned = _safe_leak_fallback(next_criterion_name=next_criterion_name, language=language)

    return clamp_text(cleaned, MAX_DIALOGUE_REPLY_CHARS), False
