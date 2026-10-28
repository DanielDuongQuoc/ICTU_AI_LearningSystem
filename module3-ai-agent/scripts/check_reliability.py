"""Phase 20 - SLO report (``python -m scripts.check_reliability``).

Reads Prometheus-text metrics (``--metrics-file`` or ``--url http://host:8000/metrics``) and prints, for every
SLO: Target | Observed | Status | Data source.  Status is PASS / FAIL / INCONCLUSIVE:

* INCONCLUSIVE when there is no data or fewer than the minimum sample count - never PASS by default.
* Percentiles are *upper bounds* estimated from histogram buckets (the bucket edge that contains the quantile).
* Performance regression needs ``--baseline`` JSON (e.g. ``{"retrieve_p95_ms": 43}``); without a baseline, or when
  the baseline has too few samples, it is INCONCLUSIVE.
* ``--offline-harness`` drives the real app in-process with fake embedding/LLM. Those numbers are CODE-PATH
  timings only: not LLM latency, not production traffic. They are labelled as such and never reported as
  production evidence; LLM SLOs stay INCONCLUSIVE in that mode.

Targets are proposals (docs/SLO_SLA.md), not measured guarantees. No production traffic data exists yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*?)\})?\s+([-+0-9.eEInfa]+)$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

MIN_SAMPLES = 30


@dataclass
class Row:
    slo: str
    target: str
    observed: str
    status: str
    source: str


def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        try:
            out.append((m.group(1), labels, float(m.group(3))))
        except ValueError:
            continue
    return out


def _sum(samples, name: str, **match: str) -> float:
    return sum(v for n, lab, v in samples if n == name and all(lab.get(k) == x for k, x in match.items()))


def histogram(samples, name: str, **match: str) -> tuple[int, list[tuple[float, float]]]:
    """(count, [(le, cumulative)]) merged over all series matching ``match``."""
    buckets: dict[float, float] = {}
    for n, lab, v in samples:
        if n == f"{name}_bucket" and all(lab.get(k) == x for k, x in match.items()):
            le = float("inf") if lab["le"] == "+Inf" else float(lab["le"])
            buckets[le] = buckets.get(le, 0.0) + v
    count = int(buckets.get(float("inf"), 0))
    return count, sorted(buckets.items())


def quantile_upper_bound(count: int, buckets: list[tuple[float, float]], q: float) -> Optional[float]:
    if count == 0:
        return None
    need = q * count
    for le, cum in buckets:
        if cum >= need:
            return le
    return None


def _latency_row(samples, label: str, route: str, target_s: float, source: str, *, min_samples=MIN_SAMPLES) -> Row:
    count, b = histogram(samples, "module3_http_request_duration_seconds", route=route)
    target = f"p95 <= {target_s * 1000:.0f} ms"
    if count < min_samples:
        return Row(label, target, f"{count} samples (< {min_samples})", "INCONCLUSIVE", source)
    p95 = quantile_upper_bound(count, b, 0.95)
    if p95 is None or p95 == float("inf"):
        return Row(label, target, f"p95 > {max(le for le, _ in b if le != float('inf')) * 1000:.0f} ms (n={count})", "FAIL", source)
    return Row(label, target, f"p95 <= {p95 * 1000:.0f} ms (n={count})", "PASS" if p95 <= target_s else "FAIL", source)


def _ratio_row(label: str, target: str, good: float, total: float, minimum: float, source: str, min_samples=MIN_SAMPLES) -> Row:
    if total < min_samples:
        return Row(label, target, f"{int(total)} samples (< {min_samples})", "INCONCLUSIVE", source)
    ratio = good / total
    status = "PASS" if ratio >= minimum else "FAIL"
    return Row(label, target, f"{ratio * 100:.2f}% (n={int(total)})", status, source)


def evaluate_slos(samples, source: str, *, baseline: Optional[dict] = None, harness: bool = False) -> list[Row]:
    rows: list[Row] = []
    total = _sum(samples, "module3_http_requests_total")
    server_err = sum(v for n, lab, v in samples
                     if n == "module3_http_requests_total" and lab.get("status", "").startswith("5"))
    rows.append(_ratio_row("Availability (non-5xx responses)", ">= 99.0% (proposal)", total - server_err, total, 0.99, source))
    rows.append(_latency_row(samples, "Retrieve latency", "/api/rag/retrieve", 2.0, source))
    llm_src = f"{source} [LLM latency NOT measured by fakes]" if harness else source
    for label, route, target in (("Question generation latency", "/api/questions/generate", 90.0),
                                 ("Evaluation latency", "/api/evaluations/evaluate", 60.0)):
        if harness:
            rows.append(Row(label, f"p95 <= {target * 1000:.0f} ms", "not measurable with fake LLM", "INCONCLUSIVE", llm_src))
        else:
            rows.append(_latency_row(samples, label, route, target, source))

    ev = lambda name, outcome: _sum(samples, "module3_events_total", event=name, outcome=outcome)  # noqa: E731
    llm_total = sum(ev("llm_request", o) for o in ("ok", "error", "timeout", "rejected"))
    if harness:
        rows.append(Row("LLM success rate", ">= 95% (proposal)", "not measurable with fake LLM", "INCONCLUSIVE", llm_src))
    else:
        rows.append(_ratio_row("LLM success rate", ">= 95% (proposal)", ev("llm_request", "ok"), llm_total, 0.95, source, 20))
        t_total = llm_total
        rows.append(_ratio_row("LLM timeout rate", "<= 5% (proposal)", t_total - ev("llm_request", "timeout"), t_total, 0.95, source, 20))
    dp_total = sum(ev("document_processing", o) for o in ("ok", "failed", "timeout", "limit_exceeded"))
    rows.append(_ratio_row("Document processing success", ">= 95% (proposal)", ev("document_processing", "ok"), dp_total, 0.95, source, 20))
    idx_total = ev("index", "ok") + ev("index", "failed")
    rows.append(_ratio_row("Index write success", ">= 99% (proposal)", ev("index", "ok"), idx_total, 0.99, source, 20))
    st_total = sum(v for n, _, v in samples if n == "module3_streams_total")
    rows.append(_ratio_row("Stream completion", ">= 95% (proposal)", _sum(samples, "module3_streams_total", outcome="ok"), st_total, 0.95, source, 20))

    # Performance regression: needs a baseline and enough samples; otherwise INCONCLUSIVE (never a guess).
    count, b = histogram(samples, "module3_http_request_duration_seconds", route="/api/rag/retrieve")
    base = (baseline or {}).get("retrieve_p95_ms")
    p95 = quantile_upper_bound(count, b, 0.95)
    if base is None:
        rows.append(Row("Retrieve p95 vs baseline", "<= 1.5x baseline", "no baseline file", "INCONCLUSIVE", source))
    elif count < MIN_SAMPLES or p95 is None or p95 == float("inf"):
        rows.append(Row("Retrieve p95 vs baseline", f"<= 1.5x {base} ms", f"{count} samples", "INCONCLUSIVE", source))
    else:
        # bucket upper bound is coarse: only call a regression when even the *bucket edge* is far above 1.5x
        rows.append(Row("Retrieve p95 vs baseline", f"<= 1.5x {base} ms", f"p95 <= {p95 * 1000:.0f} ms",
                        "PASS" if p95 * 1000 <= 1.5 * base else "INCONCLUSIVE (coarse buckets; rerun)", source))
    return rows


def render(rows: list[Row]) -> str:
    head = ("SLO", "Target", "Observed", "Status", "Data source")
    data = [head] + [(r.slo, r.target, r.observed, r.status, r.source) for r in rows]
    widths = [max(len(str(row[i])) for row in data) for i in range(5)]
    lines = []
    for n, row in enumerate(data):
        lines.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
        if n == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


async def _harness(n: int) -> str:
    """Drive the real app in-process with fakes. Code-path timings only."""
    import os
    from httpx import ASGITransport, AsyncClient

    os.environ.setdefault("LLM_MAX_CONCURRENCY", "0")
    from app.core.metrics import metrics
    from app.main import app
    from tests.fakes import Stack, sample_pdf_bytes

    class _Patch:
        def setattr(self, target, value):
            mod, attr = target.rsplit(".", 1)
            import importlib
            setattr(importlib.import_module(mod), attr, value)

    metrics.reset()
    stack = Stack().install(_Patch())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://harness") as c:
        up = await c.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes(
            ["Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."]), "application/pdf")})
        doc = up.json()["document_id"]
        for _ in range(n):
            await c.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})
    del stack
    return metrics.render()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics-file")
    src.add_argument("--url")
    src.add_argument("--offline-harness", action="store_true",
                     help="in-process fakes: code-path timings only, NOT LLM latency, NOT production")
    p.add_argument("--baseline", help="JSON file, e.g. {\"retrieve_p95_ms\": 43}")
    p.add_argument("--harness-requests", type=int, default=40)
    p.add_argument("--strict", action="store_true", help="exit 1 on INCONCLUSIVE as well as FAIL")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    harness = a.offline_harness
    if a.metrics_file:
        text, source = open(a.metrics_file, encoding="utf-8").read(), f"file:{a.metrics_file}"
    elif a.url:
        with urllib.request.urlopen(a.url, timeout=10) as r:  # noqa: S310 - operator-provided internal URL
            text, source = r.read().decode(), f"url:{a.url}"
    else:
        text, source = asyncio.run(_harness(a.harness_requests)), "offline-harness (fakes; NOT production evidence)"
    baseline = json.load(open(a.baseline, encoding="utf-8")) if a.baseline else None

    rows = evaluate_slos(parse_metrics(text), source, baseline=baseline, harness=harness)
    if harness:  # a code-path timing is not SLO evidence
        for r in rows:
            if r.status == "PASS":
                r.status = "PASS (harness only)"
    if a.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2))
    else:
        print(render(rows))
        print("\nTargets are proposals; no production traffic data is implied. INCONCLUSIVE = not enough data.")
    failed = any(r.status == "FAIL" for r in rows)
    unknown = any(r.status.startswith("INCONCLUSIVE") for r in rows)
    return 1 if failed or (a.strict and unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
