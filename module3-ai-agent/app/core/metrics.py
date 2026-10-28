"""Minimal in-process operational metrics (Phase 16, additive, no new dependency).

* Counters/histograms live in process memory (thread-safe) and reset on restart.
* Rendered in the Prometheus text format by ``GET /metrics`` (internal use, see docs).
* Label values are always bounded: HTTP method, *route template* (never the raw path, so
  document ids cannot create series), status code, a fixed set of pipeline stage names, and
  error codes from ``ErrorCode``. ``request_id``, ``document_id``, prompts, answers and PDF
  content are never recorded.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Optional

# Seconds; a final +Inf bucket is implicit.
BUCKETS_S: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

# Fixed allow-list of pipeline stages (bounds cardinality; unknown names are ignored).
STAGES: frozenset[str] = frozenset({
    "embedding", "chroma_query", "llm_generate", "llm_first_token", "llm_stream_total",
    # Phase 20 additions (same bounded allow-list, no dynamic names):
    "pdf_extract", "chunking", "index_write", "context_build", "json_parse", "validation",
})
# Phase 20: bounded event counters. Both the event names and the outcomes are fixed here; anything else
# is ignored, so no request/document/user value can ever become a label.
EVENTS: dict[str, frozenset[str]] = {
    "llm_request": frozenset({"ok", "error", "timeout", "rejected"}),
    "llm_retry": frozenset({"transient", "connection_reset"}),
    "embedding_request": frozenset({"ok", "error", "timeout", "rejected"}),
    "embedding_retry": frozenset({"transient", "connection_reset"}),
    "rag_retrieval": frozenset({"ok", "no_context", "error"}),
    "question_generation": frozenset({"ok", "error"}),
    "evaluation": frozenset({"ok", "error"}),
    "document_processing": frozenset({"ok", "failed", "timeout", "limit_exceeded"}),
    "index": frozenset({"ok", "failed", "cleaned_up", "verification_failed"}),
    "request_budget": frozenset({"exhausted", "retry_skipped"}),
}
_SLOW_KIND = {
    "llm_generate": "slow_llm", "llm_first_token": "slow_llm", "llm_stream_total": "slow_llm",
    "chroma_query": "slow_retrieval", "context_build": "slow_retrieval",
    "pdf_extract": "slow_pdf", "chunking": "slow_pdf", "embedding": "slow_embedding",
    "index_write": "slow_index",
}
logger = logging.getLogger("app.slow")
STREAM_OUTCOMES: frozenset[str] = frozenset({"ok", "error", "client_disconnected"})


class _Histogram:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.buckets = [0] * (len(BUCKETS_S) + 1)

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for i, bound in enumerate(BUCKETS_S):
            if value <= bound:
                self.buckets[i] += 1
        self.buckets[-1] += 1


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def process_memory_bytes() -> Optional[int]:
    """Resident set size when the platform exposes it cheaply (Linux /proc); else None."""
    try:
        with open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError, IndexError):
        return None


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._durations: dict[tuple[str, str], _Histogram] = defaultdict(_Histogram)
        self._stages: dict[str, _Histogram] = defaultdict(_Histogram)
        self._errors: dict[str, int] = defaultdict(int)
        self._streams: dict[str, int] = defaultdict(int)
        self._events: dict[tuple[str, str], int] = defaultdict(int)
        self._inflight = 0
        self._cancelled = 0
        self._llm_active = 0
        self._llm_waiting = 0
        self._llm_rejected = 0

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    # ---- recording -------------------------------------------------------------------
    def request_started(self) -> None:
        with self._lock:
            self._inflight += 1

    def request_finished(self, method: str, route: str, status: int, duration_s: float) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._requests[(method, route, str(status))] += 1
            self._durations[(method, route)].observe(duration_s)

    def request_cancelled(self) -> None:
        with self._lock:
            self._cancelled += 1

    def error(self, code: str) -> None:
        with self._lock:
            self._errors[str(code)] += 1

    def observe_stage(self, stage: str, duration_s: float) -> None:
        if stage not in STAGES:
            return
        with self._lock:
            self._stages[stage].observe(duration_s)
        self._after_stage(stage, duration_s)

    @staticmethod
    def _after_stage(stage: str, duration_s: float) -> None:
        """Per-request stage totals (for slow-request logs) and slow-stage warnings. Never raises."""
        try:
            from app.core.config import get_settings
            from app.core.request_context import record_stage

            record_stage(stage, duration_s)
            kind = _SLOW_KIND.get(stage, "slow_stage")
            s = get_settings()
            limit_ms = s.slow_llm_threshold_ms if kind == "slow_llm" else s.slow_stage_threshold_ms
            if duration_s * 1000.0 > limit_ms:
                logger.warning("%s stage=%s duration_ms=%.1f threshold_ms=%d",
                               kind, stage, duration_s * 1000.0, limit_ms)
        except Exception:  # observability must never break a request
            pass

    def event(self, name: str, outcome: str) -> None:
        if outcome not in EVENTS.get(name, frozenset()):
            return
        with self._lock:
            self._events[(name, outcome)] += 1

    def stream_finished(self, outcome: str) -> None:
        with self._lock:
            self._streams[outcome if outcome in STREAM_OUTCOMES else "error"] += 1

    def llm_waiting(self, delta: int) -> None:
        with self._lock:
            self._llm_waiting = max(0, self._llm_waiting + delta)

    def llm_active(self, delta: int) -> None:
        with self._lock:
            self._llm_active = max(0, self._llm_active + delta)

    def llm_rejected(self) -> None:
        with self._lock:
            self._llm_rejected += 1

    # ---- reading ---------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime": time.time() - self._started,
                "inflight": self._inflight,
                "cancelled": self._cancelled,
                "llm_active": self._llm_active,
                "llm_waiting": self._llm_waiting,
                "llm_rejected": self._llm_rejected,
                "requests": dict(self._requests),
                "errors": dict(self._errors),
                "streams": dict(self._streams),
                "events": dict(self._events),
                "durations": {k: (h.sum, h.count, list(h.buckets)) for k, h in self._durations.items()},
                "stages": {k: (h.sum, h.count, list(h.buckets)) for k, h in self._stages.items()},
            }

    @staticmethod
    def _hist_lines(name: str, label_sets: dict, lines: list[str]) -> None:
        for labels, (total, count, buckets) in sorted(label_sets.items()):
            for bound, n in zip(BUCKETS_S, buckets):
                lines.append(f'{name}_bucket{{{labels},le="{bound}"}} {n}')
            lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {buckets[-1]}')
            lines.append(f"{name}_sum{{{labels}}} {total:.6f}")
            lines.append(f"{name}_count{{{labels}}} {count}")

    def render(self) -> str:
        s = self.snapshot()
        out: list[str] = []

        def gauge(name: str, help_: str, value) -> None:
            out.extend([f"# HELP {name} {help_}", f"# TYPE {name} gauge", f"{name} {value}"])

        def counter(name: str, help_: str, value) -> None:
            out.extend([f"# HELP {name} {help_}", f"# TYPE {name} counter", f"{name} {value}"])

        gauge("module3_uptime_seconds", "Seconds since the process started.", f"{s['uptime']:.1f}")
        gauge("module3_http_requests_inflight", "Requests currently being served.", s["inflight"])
        gauge("module3_llm_active", "LLM calls currently holding a concurrency slot.", s["llm_active"])
        gauge("module3_llm_waiting", "LLM calls waiting for a concurrency slot.", s["llm_waiting"])
        rss = process_memory_bytes()
        if rss is not None:
            gauge("module3_process_resident_memory_bytes", "Resident memory (Linux only).", rss)
        counter("module3_http_requests_cancelled_total", "Requests aborted by cancellation/disconnect.", s["cancelled"])
        counter("module3_llm_rejected_total", "LLM calls rejected because no slot became free in time.", s["llm_rejected"])

        out += ["# HELP module3_http_requests_total Completed requests.", "# TYPE module3_http_requests_total counter"]
        for (method, route, status), n in sorted(s["requests"].items()):
            out.append(f'module3_http_requests_total{{method="{_esc(method)}",route="{_esc(route)}",status="{status}"}} {n}')
        out += ["# HELP module3_errors_total API errors by error code.", "# TYPE module3_errors_total counter"]
        for code, n in sorted(s["errors"].items()):
            out.append(f'module3_errors_total{{code="{_esc(code)}"}} {n}')
        out += ["# HELP module3_streams_total SSE evaluation streams by outcome.", "# TYPE module3_streams_total counter"]
        for outcome, n in sorted(s["streams"].items()):
            out.append(f'module3_streams_total{{outcome="{_esc(outcome)}"}} {n}')
        out += ["# HELP module3_events_total Dependency/pipeline events by fixed name and outcome.",
                "# TYPE module3_events_total counter"]
        for (name, outcome), n in sorted(s["events"].items()):
            out.append(f'module3_events_total{{event="{_esc(name)}",outcome="{_esc(outcome)}"}} {n}')
        out += ["# HELP module3_http_request_duration_seconds Request duration (SSE: until stream end).",
                "# TYPE module3_http_request_duration_seconds histogram"]
        self._hist_lines("module3_http_request_duration_seconds",
                         {f'method="{_esc(m)}",route="{_esc(r)}"': v for (m, r), v in s["durations"].items()}, out)
        out += ["# HELP module3_stage_duration_seconds Pipeline stage duration.",
                "# TYPE module3_stage_duration_seconds histogram"]
        self._hist_lines("module3_stage_duration_seconds",
                         {f'stage="{_esc(k)}"': v for k, v in s["stages"].items()}, out)
        return "\n".join(out) + "\n"


metrics = MetricsRegistry()
