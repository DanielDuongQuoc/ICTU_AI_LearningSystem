"""Timeout chain (Phase 20).

    REQUEST_TIMEOUT_S (whole request)  >=  LLM_TIMEOUT (one model call)  >=  LLM_CONNECT_TIMEOUT
                                       ^ validated at startup (config.py)

The request budget is a *deadline*, not a cancellation: the middleware records ``now + REQUEST_TIMEOUT_S``
and every LLM / embedding call (a) is given a tighter httpx timeout when the remaining budget is smaller than its own
timeout, (b) is refused when the budget is already spent, and (c) is not retried when too little is left.
So a request can never keep issuing model calls long after its client has given up, and retries cannot
multiply the worst case beyond the budget. Outside a request (scripts, unit tests) there is no deadline and
behaviour is exactly the per-call timeouts.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.request_context import remaining_s

MIN_USEFUL_ATTEMPT_S = 1.0  # a call (or retry) with less time than this left is pointless


def call_timeout(default_s: float, connect_s: float, *, code: str = ErrorCode.LLM_TIMEOUT) -> Optional[httpx.Timeout]:
    """``None`` when the call's own timeout applies; a tightened ``httpx.Timeout`` when the budget is binding.

    Raises ``AppError(code)`` when the budget is exhausted.
    """
    left = remaining_s()
    if left is None or left >= default_s:
        return None
    if left < MIN_USEFUL_ATTEMPT_S:
        metrics.event("request_budget", "exhausted")
        raise AppError(code, "The request time budget was exhausted.")
    return httpx.Timeout(left, connect=min(connect_s, left))


def can_retry(backoff_s: float, expected_call_s: float = MIN_USEFUL_ATTEMPT_S) -> bool:
    """True when a retry (backoff + one more attempt) still fits in the request budget."""
    left = remaining_s()
    if left is None or left > backoff_s + max(expected_call_s, MIN_USEFUL_ATTEMPT_S):
        return True
    metrics.event("request_budget", "retry_skipped")
    return False

