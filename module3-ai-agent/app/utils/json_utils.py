import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_json_from_text(text: str) -> str:
    """Return the JSON payload from raw LLM output.

    Handles plain JSON, Markdown-fenced JSON, and JSON surrounded by prose.
    """
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    # Prefer whichever bracket type appears first in the text.
    candidates = []
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append((start, end))
    if candidates:
        start, end = min(candidates)
        return text[start:end + 1]
    raise ValueError("No JSON block found in LLM output.")


def _reject_constant(name: str) -> Any:
    # json.loads accepts NaN / Infinity / -Infinity by default, which are not valid JSON.
    raise ValueError(f"JSON parse error: non-finite number {name} is not allowed.")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # With duplicate keys json.loads silently keeps the LAST one, so an attacker-influenced output
    # could hide a benign first value (e.g. a second "score"). Reject ambiguity instead.
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("JSON parse error: duplicate key in object.")
        out[key] = value
    return out


def safe_parse_json(text: str) -> Any:
    extracted = extract_json_from_text(text)
    try:
        return json.loads(extracted, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc.msg} (line {exc.lineno}, column {exc.colno}).") from None
    except RecursionError:
        raise ValueError("JSON parse error: nesting too deep.") from None
