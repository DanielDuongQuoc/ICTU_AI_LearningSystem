"""Operational check: read /ready and /metrics of a running instance and compare with alert thresholds.

    python scripts/ops_check.py [--base-url http://localhost:8000]
        [--max-5xx-ratio 0.05] [--max-p95-seconds 60] [--min-requests 20] [--max-llm-rejections 10]

Exit code 0 = OK, 1 = at least one alert, 2 = instance unreachable.
Read-only (GET only). It sends nothing anywhere; wire its exit code into cron / your monitor.
Thresholds are CLI options, not hardcoded policy.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

import httpx

_LINE = re.compile(r'^(?P<name>[a-z0-9_]+)(?:\{(?P<labels>[^}]*)\})? (?P<value>\S+)$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            labels = dict(_LABEL.findall(m.group("labels") or ""))
            samples.append((m.group("name"), labels, float(m.group("value"))))
    return samples


def evaluate(samples, *, max_5xx_ratio: float, max_p95_seconds: float, min_requests: int,
             max_llm_rejections: int = 10) -> list[str]:
    """Return a list of alert messages (empty = healthy)."""
    alerts: list[str] = []
    total = errors = 0.0
    for name, labels, value in samples:
        if name == "module3_http_requests_total":
            total += value
            if labels.get("status", "").startswith("5"):
                errors += value
        elif name == "module3_llm_rejected_total" and value > max_llm_rejections:
            alerts.append(f"LLM_OVERLOAD: {value:.0f} rejected calls > {max_llm_rejections}")
    if total >= min_requests and errors / total > max_5xx_ratio:
        alerts.append(f"HIGH_5XX_RATIO: {errors:.0f}/{total:.0f} = {errors / total:.1%} > {max_5xx_ratio:.1%}")

    buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for name, labels, value in samples:
        if name == "module3_http_request_duration_seconds_bucket":
            le = float("inf") if labels["le"] == "+Inf" else float(labels["le"])
            buckets[(labels["method"], labels["route"])].append((le, value))
    for (method, route), items in buckets.items():
        items.sort()
        count = items[-1][1]
        if count < min_requests:
            continue
        target = 0.95 * count
        p95 = next((le for le, n in items if n >= target), float("inf"))
        if p95 > max_p95_seconds:
            alerts.append(f"SLOW_P95: {method} {route} p95<={p95}s > {max_p95_seconds}s (n={count:.0f})")
    return alerts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--max-5xx-ratio", type=float, default=0.05)
    p.add_argument("--max-p95-seconds", type=float, default=60.0)
    p.add_argument("--min-requests", type=int, default=20)
    p.add_argument("--max-llm-rejections", type=int, default=10)
    args = p.parse_args(argv)
    try:
        with httpx.Client(base_url=args.base_url, timeout=10.0) as c:
            ready = c.get("/ready")
            metrics_text = c.get("/metrics").text
    except httpx.HTTPError as exc:
        print(f"UNREACHABLE: {type(exc).__name__}", file=sys.stderr)
        return 2
    alerts = []
    if ready.status_code != 200:
        try:
            services = ready.json().get("services", {})
        except ValueError:
            services = {}
        alerts.append(f"NOT_READY: /ready={ready.status_code} services={services}")
    alerts += evaluate(parse_metrics(metrics_text), max_5xx_ratio=args.max_5xx_ratio,
                       max_p95_seconds=args.max_p95_seconds, min_requests=args.min_requests,
                       max_llm_rejections=args.max_llm_rejections)
    for a in alerts:
        print(f"ALERT {a}")
    print("OK" if not alerts else f"{len(alerts)} alert(s)")
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
