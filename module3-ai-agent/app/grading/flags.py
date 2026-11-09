"""Cheap, deterministic review signals computed by the backend (independent of the model's own flags)."""

from __future__ import annotations

import difflib
import re

# Flags that put a homework result into ``needs_review`` (never auto-final).
REVIEW_FLAGS = frozenset({
    "injection_suspected",   # submission tries to instruct the grader
    "no_valid_files",        # Agent: nothing gradable after file filtering
    "unreadable",            # Agent / transcription: content could not be read
    "truncated",             # Agent: submission cut to fit the size limit, the model did not see everything
    "not_executed",          # Agent: a required test criterion could not be run
    "grading_error",         # system failure while grading
    "diagram_present",       # handwritten: a drawing/diagram the lecturer must grade by hand
})
# Informational only (result stays final for homework).
WARNING_FLAGS = frozenset({"empty_or_off_topic", "ungrounded_rubric"})

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|any\s+)?(previous|above|prior|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|the\s+)?(previous|above|prior)?\s*(instructions|rules)",
    r"(give|award|assign)\s+(me\s+|this\s+(answer|submission)\s+)?(full|maximum|max|perfect|10|100)\s*(/\s*10\s*)?"
    r"(marks?|points?|score|grade)",
    r"\b(mark|grade)\s+(this|me)\s+(as\s+)?(pass(ed)?|correct|full)",
    r"system\s+prompt",
    r"you\s+are\s+now\s+",
    r"\bas\s+an?\s+(ai|language\s+model|grader)\b.*\b(must|should)\b",
    r"bỏ\s+qua\s+(các\s+|mọi\s+|tất\s+cả\s+)?(hướng\s+dẫn|chỉ\s+dẫn|yêu\s+cầu|quy\s+tắc)",
    r"cho\s+(em|tôi|mình|bài\s+này)\s+(điểm\s+)?(10|tối\s+đa|max|đạt|qua)",
    r"chấm\s+(cho\s+)?(em|tôi|mình|bài\s+này)\s+(điểm\s+)?(10|tối\s+đa|đạt|qua)",
    r"(gửi|nhắn)\s+(ai|giám\s+khảo|người\s+chấm)\s*[:,]",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
MIN_ANSWER_CHARS = 10


def _norm(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def is_empty_or_copy(answer: str, question: str) -> bool:
    a = _norm(answer or "")
    if len(a) < MIN_ANSWER_CHARS:
        return True
    q = _norm(question or "")
    return bool(q) and (a == q or difflib.SequenceMatcher(None, a, q).ratio() >= 0.9)


def split_flags(flags: list[str]) -> tuple[list[str], list[str]]:
    """``(review_flags, warnings)``; unknown labels are treated as review flags (fail safe)."""
    review, warn = [], []
    for f in dict.fromkeys(flags):  # de-duplicate, keep order
        (warn if f in WARNING_FLAGS else review).append(f)
    return review, warn
