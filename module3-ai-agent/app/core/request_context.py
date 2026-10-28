"""Request-scoped context (request id) shared by middleware, handlers and logging."""

from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"

# Accept UUIDs and other short, log-safe tokens; anything else is replaced by a new UUID.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def is_valid_request_id(value: Optional[str]) -> bool:
    return bool(value) and _VALID_REQUEST_ID.match(value) is not None


def resolve_request_id(client_value: Optional[str]) -> str:
    """Reuse the client's X-Request-ID when it is safe, otherwise generate a UUID."""
    if client_value is not None and is_valid_request_id(client_value.strip()):
        return client_value.strip()
    return str(uuid.uuid4())


def get_request_id(request: Optional[Request] = None) -> str:
    """Request id for ``request`` (or the current context); generates one as a last resort."""
    if request is not None:
        value = getattr(request.state, "request_id", None)
        if value:
            return value
    current = request_id_var.get()
    if current and current != "-":
        return current
    return str(uuid.uuid4())


# ---- Phase 20: request time budget and per-request stage timings -----------------------------
# Both are set by the request middleware; outside a request (tests, scripts) they are unset and
# every helper degrades to "no budget / not recorded".
deadline_var: ContextVar[Optional[float]] = ContextVar("request_deadline", default=None)
stage_times_var: ContextVar[Optional[dict]] = ContextVar("request_stage_times", default=None)


def remaining_s() -> Optional[float]:
    """Seconds left in this request's time budget (may be <= 0), or None when no budget is set."""
    deadline = deadline_var.get()
    return None if deadline is None else deadline - time.monotonic()


def record_stage(stage: str, seconds: float) -> None:
    """Accumulate a pipeline stage duration for the current request (used by slow-request logs)."""
    times = stage_times_var.get()
    if times is not None:
        times[stage] = times.get(stage, 0.0) + seconds


def slowest_stage() -> Optional[tuple[str, float]]:
    times = stage_times_var.get()
    if not times:
        return None
    name = max(times, key=times.get)
    return name, times[name]
