"""RAG prompt rules and the tolerant JSON parser used on model output."""

from __future__ import annotations

import pytest

from app.core.prompts import (
    PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    build_rag_messages,
)
from app.utils.json_utils import safe_parse_json


# ── Prompts ─────────────────────────────────────────────────────────


def test_prompt_version_and_rules() -> None:
    assert PROMPT_VERSION == "v1"
    for rule in (
        "ONLY the provided reference context",
        "Do not use external knowledge",
        "Do not invent facts",
        "Do not invent source pages",
        "Do not invent chunk IDs",
        "Do not invent similarity scores",
        "conflict",
    ):
        assert rule in RAG_SYSTEM_PROMPT


def test_prompt_contains_question_and_context() -> None:
    system, user = build_rag_messages("CTX BODY", "THE QUESTION?")
    assert system == RAG_SYSTEM_PROMPT
    assert "CTX BODY" in user and "THE QUESTION?" in user
    assert "REFERENCE CONTEXT" in user and "USER QUESTION" in user


def test_prompt_survives_special_characters() -> None:
    nasty = 'He said "hi" {x} {0} {{y}} %s \\n\nline2 `code` ```json'
    _, user = build_rag_messages(nasty, nasty)
    assert user.count(nasty) == 2


def test_prompts_contain_no_secrets() -> None:
    _, user = build_rag_messages("c", "q")
    blob = (RAG_SYSTEM_PROMPT + user).lower()
    for word in ("api_key", "api key", "secret", "password", "token"):
        assert word not in blob


# ── json_utils ──────────────────────────────────────────────────────


def test_json_utils_variants() -> None:
    assert safe_parse_json('{"a": 1}') == {"a": 1}
    assert safe_parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert safe_parse_json('```\n{"a": 1}\n```') == {"a": 1}
    assert safe_parse_json('Sure! {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(ValueError):
        safe_parse_json("no json here")
    with pytest.raises(ValueError):
        safe_parse_json("{broken: json}")
