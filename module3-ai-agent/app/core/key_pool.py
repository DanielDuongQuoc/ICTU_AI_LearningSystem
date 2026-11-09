"""Round-robin API key pool with automatic per-key quota cooldown (failover on HTTP 429/5xx).

``GEMINI_API_KEY`` may contain a single key or multiple keys separated by commas, semicolons,
or whitespace::

    GEMINI_API_KEY=key_1,key_2,key_3

Behaviour:
1. Healthy keys are chosen in round-robin order so RPM/TPM/RPD limits are spread evenly.
2. When a key receives HTTP 429 (quota / rate limit) or 500/502/503/504, it is placed in a
   temporary cooldown (default 60s for 429, 10s for 5xx) and the caller immediately fails over
   to the next healthy key in the pool.
3. If every key is currently cooling down, the key whose cooldown expires soonest is selected.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Optional

_SPLIT_RE = re.compile(r"[,;\s]+")
DEFAULT_RATE_LIMIT_COOLDOWN_S = 60.0
DEFAULT_TRANSIENT_COOLDOWN_S = 10.0


def parse_api_keys(raw: Optional[str]) -> list[str]:
    """Split a raw ``GEMINI_API_KEY`` string into a deduplicated list of non-empty keys."""
    if not raw:
        return []
    seen: list[str] = []
    for part in _SPLIT_RE.split(raw.strip()):
        cleaned = part.strip().strip("'\"")
        if cleaned.startswith("="):
            cleaned = cleaned.lstrip("=").strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


class GeminiKeyPool:
    """Round-robin pool of Gemini API keys with per-key cooldown tracking."""

    def __init__(
        self,
        keys: str | Iterable[str] | None,
        *,
        rate_limit_cooldown_s: float = DEFAULT_RATE_LIMIT_COOLDOWN_S,
        transient_cooldown_s: float = DEFAULT_TRANSIENT_COOLDOWN_S,
    ) -> None:
        if isinstance(keys, str) or keys is None:
            self._keys: tuple[str, ...] = tuple(parse_api_keys(keys))
        else:
            deduped: list[str] = []
            for k in keys:
                for p in parse_api_keys(str(k)):
                    if p not in deduped:
                        deduped.append(p)
            self._keys = tuple(deduped)
        self._index: int = 0
        self._cooldown_until: dict[str, float] = {}
        self._rate_limit_cooldown_s = max(0.0, rate_limit_cooldown_s)
        self._transient_cooldown_s = max(0.0, transient_cooldown_s)

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @property
    def size(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def key_slot(self, key: str) -> str:
        """Return a safe 1-based ``slot/total`` label for logging (never logs the key itself)."""
        if not self._keys or key not in self._keys:
            return f"?/{len(self._keys)}"
        return f"{self._keys.index(key) + 1}/{len(self._keys)}"

    def next_key(self, *, exclude: Optional[set[str]] = None) -> str:
        """Return the next available key in round-robin order, skipping cooling-down keys."""
        if not self._keys:
            return ""
        n = len(self._keys)
        now = time.monotonic()
        excluded = exclude or set()

        # Pass 1: healthy keys not yet tried in this call
        for offset in range(n):
            idx = (self._index + offset) % n
            key = self._keys[idx]
            if key in excluded:
                continue
            if self._cooldown_until.get(key, 0.0) <= now:
                self._index = (idx + 1) % n
                return key

        # Pass 2: any untried key with the earliest cooldown expiry
        candidates = [k for k in self._keys if k not in excluded]
        if candidates:
            best = min(candidates, key=lambda k: self._cooldown_until.get(k, 0.0))
            self._index = (self._keys.index(best) + 1) % n
            return best

        # Pass 3: all keys were tried; pick the key whose cooldown expires soonest
        best = min(self._keys, key=lambda k: self._cooldown_until.get(k, 0.0))
        self._index = (self._keys.index(best) + 1) % n
        return best

    def report_status(self, key: str, status_code: int) -> None:
        """Update cooldown state for ``key`` based on the upstream HTTP status code."""
        if not key or key not in self._keys:
            return
        if status_code == 200:
            self._cooldown_until.pop(key, None)
        elif status_code == 429:
            self._cooldown_until[key] = time.monotonic() + self._rate_limit_cooldown_s
        elif status_code in (500, 502, 503, 504):
            self._cooldown_until[key] = time.monotonic() + self._transient_cooldown_s

