"""In-process abuse controls (Phase 19): sliding-window rate limiting and a concurrent-upload gate.

Scope and limits (also in docs/SECURITY.md):

* Single instance only. State lives in this process; several workers/containers each enforce their own
  budget and a restart resets it. Distributed limiting needs shared state (Redis/gateway) and was
  deliberately not added.
* The client key is the TCP peer address. ``X-Forwarded-For`` is NOT trusted (it is client-controlled), so
  clients behind one reverse proxy share one bucket - put real per-client limiting in the gateway.
* This bounds cost/abuse; it is not authentication.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Request

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.request_context import get_request_id

logger = logging.getLogger("app.security")

MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str, limit: int, window_s: float) -> Optional[int]:
        """Record a hit; return ``None`` if allowed, else the seconds to wait (Retry-After)."""
        if limit <= 0:
            return None
        now = time.monotonic()
        full_key = f"{bucket}:{key}"
        with self._lock:
            if len(self._events) > MAX_TRACKED_CLIENTS:  # bound memory under address-spraying
                self._events = {k: v for k, v in self._events.items() if v and now - v[-1] < window_s}
            q = self._events.setdefault(full_key, deque())
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(window_s - (now - q[0])) + 1)
            q.append(now)
            return None

    def count(self, bucket: str, key: str, window_s: float) -> tuple[int, int]:
        """(events in window, seconds until the oldest expires) without recording anything."""
        now = time.monotonic()
        with self._lock:
            q = self._events.get(f"{bucket}:{key}")
            if not q:
                return 0, 0
            while q and now - q[0] >= window_s:
                q.popleft()
            return (len(q), max(1, int(window_s - (now - q[0])) + 1)) if q else (0, 0)

    def add(self, bucket: str, key: str) -> None:
        with self._lock:
            self._events.setdefault(f"{bucket}:{key}", deque()).append(time.monotonic())

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


limiter = SlidingWindowLimiter()


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RateLimited(AppError):
    def __init__(self, retry_after: int) -> None:
        super().__init__(ErrorCode.RATE_LIMITED, "Too many requests; please slow down.")
        self.retry_after = retry_after


def rate_limit(bucket: str = "expensive"):
    """FastAPI dependency: ``RATE_LIMIT_PER_MINUTE`` requests per client per minute (0 = off)."""

    async def dependency(request: Request) -> None:
        wait = limiter.check(bucket, client_key(request), get_settings().rate_limit_per_minute, 60.0)
        if wait is not None:
            logger.warning("rate limited request_id=%s bucket=%s retry_after=%s",
                           get_request_id(request), bucket, wait)
            raise RateLimited(wait)

    return dependency


# ---- concurrent uploads -----------------------------------------------------------------------
_upload_lock = threading.Lock()
_uploads_active = 0


@asynccontextmanager
async def upload_slot() -> AsyncIterator[None]:
    """Reject (never queue) when ``MAX_CONCURRENT_UPLOADS`` uploads are already being processed."""
    global _uploads_active
    limit = get_settings().max_concurrent_uploads
    if limit > 0:
        with _upload_lock:
            if _uploads_active >= limit:
                raise AppError(ErrorCode.SERVICE_OVERLOADED, "Too many uploads are being processed; retry shortly.")
            _uploads_active += 1
    try:
        yield
    finally:
        if limit > 0:
            with _upload_lock:
                _uploads_active -= 1
