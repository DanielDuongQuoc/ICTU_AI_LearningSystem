"""Server-Sent Events helpers — the single place where SSE frames are formatted.

Every frame is ``data: <compact JSON>\\n\\n``.  JSON is produced with ``json.dumps``
(never ``str(dict)``) and ``ensure_ascii=False`` so non-ASCII text stays readable.
"""

from __future__ import annotations

import json
from typing import Any

SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

EVENT_TYPES = ("status", "token", "result", "error", "done")


def format_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def status_event(message: str) -> str:
    return format_sse({"type": "status", "message": message})


def token_event(content: str) -> str:
    return format_sse({"type": "token", "content": content})


def result_event(result: dict[str, Any]) -> str:
    return format_sse({"type": "result", "result": result})


def error_event(code: str, message: str) -> str:
    return format_sse({"type": "error", "code": code, "message": message})


def done_event() -> str:
    return format_sse({"type": "done"})
