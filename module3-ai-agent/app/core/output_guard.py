"""Treat model output as untrusted (Phase 19): size cap, prompt-leak detection, bounded fields.

* The size cap stops a runaway model (or injected "repeat forever") from becoming a large response/parse.
* Prompt-leak detection compares the output with the SYSTEM prompt texts of every registered prompt using
  exact 8-word shingles. It is a tripwire for verbatim disclosure ("reveal your system prompt"), not a
  proof: a paraphrased leak is not detected. Shingles drawn from the sentences the system prompt tells the
  model to emit are never counted.
* ``clamp_*`` keep response fields bounded after Pydantic validation.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from app.core import prompt_registry as registry
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.prompts import INSUFFICIENT_CONTEXT_ANSWER

logger = logging.getLogger(__name__)

SHINGLE_WORDS = 8
MAX_FEEDBACK_CHARS = 4000
MAX_LIST_ITEMS = 20
MAX_LIST_ITEM_CHARS = 1000

_WORD = re.compile(r"\w+", re.UNICODE)
_cache: tuple[int, frozenset[tuple[str, ...]]] = (-1, frozenset())


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _shingles(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + SHINGLE_WORDS]) for i in range(len(w) - SHINGLE_WORDS + 1)}


def _system_shingles() -> frozenset[tuple[str, ...]]:
    global _cache
    specs = list(registry._REGISTRY.values())
    key = hash(tuple(sorted(s.fingerprint for s in specs)))
    if _cache[0] != key:
        grams: set[tuple[str, ...]] = set()
        for spec in specs:
            for text in spec.system_texts:
                grams |= _shingles(text)
        grams -= _shingles(INSUFFICIENT_CONTEXT_ANSWER)
        _cache = (key, frozenset(grams))
    return _cache[1]


def contains_prompt_leak(*texts: str) -> bool:
    grams = _system_shingles()
    return any(_shingles(t) & grams for t in texts if t)


def enforce_output_size(raw: str) -> None:
    if len(raw) > get_settings().max_llm_output_chars:
        raise AppError(ErrorCode.LLM_OUTPUT_REJECTED, "The language model output exceeded the allowed size.")


def enforce_no_prompt_leak(*texts: str) -> None:
    if contains_prompt_leak(*texts):
        logger.warning("model output rejected: verbatim system-prompt text detected")  # never the text itself
        raise AppError(ErrorCode.LLM_OUTPUT_REJECTED, "The language model output was rejected by the safety check.")


def clamp_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit].rstrip()


def clamp_list(items: Iterable[str], max_items: int = MAX_LIST_ITEMS, max_chars: int = MAX_LIST_ITEM_CHARS) -> list[str]:
    return [clamp_text(i, max_chars) for i in list(items)[:max_items]]
