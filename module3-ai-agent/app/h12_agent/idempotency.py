"""In-memory idempotency store for repeated-request handling (H12 PI 2.2)."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.core.exceptions import AppError, ErrorCode


@dataclass
class _Entry:
    fingerprint: str
    response: dict[str, Any]
    created_at: float


class IdempotencyStore:
    """Bounded, TTL-aware store mapping ``(endpoint, key)`` to a canonical request fingerprint."""

    def __init__(self, *, max_entries: int = 2048) -> None:
        self._max_entries = max(16, max_entries)
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def compute_fingerprint(endpoint: str, payload: dict[str, Any]) -> str:
        cleaned = {k: v for k, v in payload.items() if k != "idempotency_key"}
        canonical = json.dumps(
            {"endpoint": endpoint, "payload": cleaned},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _prune(self, now: float, ttl_s: int) -> None:
        expired = [k for k, entry in self._entries.items() if now - entry.created_at >= ttl_s]
        for k in expired:
            del self._entries[k]
        while len(self._entries) >= self._max_entries:
            oldest_key = min(self._entries, key=lambda k: self._entries[k].created_at)
            del self._entries[oldest_key]

    def lookup_or_conflict(
        self,
        *,
        endpoint: str,
        key: str,
        fingerprint: str,
        ttl_s: int,
    ) -> Optional[dict[str, Any]]:
        """Return cached response on exact replay, None on miss, or raise 409 on payload conflict."""
        composite = f"{endpoint}:{key}"
        now = time.monotonic()
        with self._lock:
            self._prune(now, ttl_s)
            entry = self._entries.get(composite)
            if entry is None:
                return None
            if entry.fingerprint != fingerprint:
                raise AppError(
                    ErrorCode.IDEMPOTENCY_KEY_CONFLICT,
                    "Idempotency-Key was already used with a different request payload.",
                )
            return copy.deepcopy(entry.response)

    def save(
        self,
        *,
        endpoint: str,
        key: str,
        fingerprint: str,
        response_data: dict[str, Any],
        ttl_s: int,
    ) -> None:
        composite = f"{endpoint}:{key}"
        now = time.monotonic()
        with self._lock:
            self._prune(now, ttl_s)
            self._entries[composite] = _Entry(
                fingerprint=fingerprint,
                response=copy.deepcopy(response_data),
                created_at=now,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_idempotency_store = IdempotencyStore()


def get_idempotency_store() -> IdempotencyStore:
    return _idempotency_store
