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

# Patterns are deliberately specific: they also run on SOURCE CODE (comments, identifiers), where phrases such as
# "assign max score" or "self.system: str" are normal and must not raise a review flag.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|any\s+)?(previous|above|prior|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|the\s+)?(previous|above|prior)?\s*(instructions|rules)",
    r"\b(give|award|assign)\s+(me|us|this\s+(answer|submission|solution|code|essay))\s+"
    r"(full|maximum|max|perfect|10|100)\s*(/\s*10\s*)?(marks?|points?|score|grade)",
    r"\b(mark|grade)\s+(this|me)\s+(as\s+)?(pass(ed)?|correct|full\s+marks?)\b",
    r"system\s+prompt",
    r"\b(the\s+)?(grader|examiner|assistant|model)\s+(must|should|has\s+to)\s+(give|award|mark|grade|assign)",
    r"you\s+are\s+now\s+(a|an|the)\s",
    r"\bas\s+an?\s+(ai|language\s+model|grader)\b.*\b(must|should)\b",
    r"bỏ\s+qua\s+(các\s+|mọi\s+|tất\s+cả\s+)?(hướng\s+dẫn|chỉ\s+dẫn|yêu\s+cầu|quy\s+tắc)",
    r"cho\s+(em|tôi|mình|bài\s+này)\s+(điểm\s+(10|tối\s+đa|max|cao)|đạt|qua\s+môn|qua)\b",
    r"chấm\s+(cho\s+)?(em|tôi|mình|bài\s+này)\s+(điểm\s+(10|tối\s+đa|max|cao)|đạt|qua)\b",
    r"(gửi|nhắn)\s+(ai|giám\s+khảo|người\s+chấm)\s*[:,]",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
COPY_RATIO = 0.9


def _norm(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def is_empty_or_copy(answer: str, question: str) -> bool:
    """True for an answer with no word at all, or one that only repeats the question.

    Short answers are NOT empty: "O(n log n)", "3NF" or "True" are valid short answers and must be graded.
    """
    a = _norm(answer or "")
    if not a:
        return True
    q = _norm(question or "")
    return bool(q) and len(a) >= 10 and (a == q or difflib.SequenceMatcher(None, a, q).ratio() >= COPY_RATIO)


def split_flags(flags: list[str]) -> tuple[list[str], list[str]]:
    """``(review_flags, warnings)``; unknown labels are treated as review flags (fail safe)."""
    review, warn = [], []
    for f in dict.fromkeys(flags):  # de-duplicate, keep order
        (warn if f in WARNING_FLAGS else review).append(f)
    return review, warn
