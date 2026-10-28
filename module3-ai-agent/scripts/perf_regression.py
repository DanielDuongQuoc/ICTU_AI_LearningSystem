"""Phase 22 - performance regression gate (``python -m scripts.perf_regression BASELINE.json CURRENT.json``).

Both files are ``scripts.benchmark_scale`` outputs. The verdict keeps failure classes apart instead of a generic
"test failed":

* ``PASS``                    - every comparable metric is within the threshold
* ``PERFORMANCE REGRESSION``  - a comparable metric slowed by more than the threshold
* ``INSUFFICIENT DATA``       - no baseline, too few samples, or the two runs are not comparable
                                (different dataset / embedding / chunk config / RAG config)
* ``INFRASTRUCTURE FAILURE``  - a file is missing or unreadable

Functional failures and quality regressions are reported by pytest and ``scripts.quality_gate`` respectively;
this tool never turns those into a performance verdict. Exit code: 0 PASS, 1 PERFORMANCE REGRESSION,
2 INSUFFICIENT DATA, 3 INFRASTRUCTURE FAILURE. Threshold: ``--threshold`` or PERFORMANCE_REGRESSION_THRESHOLD_PERCENT.
Absolute values under ``--noise-floor-ms`` are ignored (a 0.4 ms -> 0.6 ms change is timer noise, not a regression).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

PASS = "PASS"
REGRESSION = "PERFORMANCE REGRESSION"
INSUFFICIENT = "INSUFFICIENT DATA"
INFRA = "INFRASTRUCTURE FAILURE"

COMPARABLE_KEYS = ("dataset_version", "embedding", "configured_embedding_model", "configured_llm_model",
                   "prompt_version", "chunk_size", "chunk_overlap", "top_k", "similarity_threshold",
                   "max_context_chars")


def comparable(base_meta: dict[str, Any], cur_meta: dict[str, Any]) -> list[str]:
    """Return the configuration keys that differ (empty list = comparable)."""
    return [k for k in COMPARABLE_KEYS if base_meta.get(k) != cur_meta.get(k)]


def compare(baseline: dict[str, Any], current: dict[str, Any], threshold_percent: float,
            min_samples: int = 1, noise_floor_ms: float = 1.0) -> dict[str, Any]:
    differing = comparable(baseline.get("metadata", {}), current.get("metadata", {}))
    if differing:
        return {"verdict": INSUFFICIENT, "reason": "runs are not comparable; differing: " + ", ".join(differing),
                "rows": []}
    rows: list[dict[str, Any]] = []
    for name, cur in current.get("metrics", {}).items():
        base = baseline.get("metrics", {}).get(name)
        row: dict[str, Any] = {"metric": name, "baseline_ms": None, "current_ms": cur.get("value_ms"),
                               "regression_percent": None, "status": INSUFFICIENT}
        if base is None or base.get("value_ms") is None or cur.get("value_ms") is None:
            row["note"] = "no baseline value"
        elif min(base.get("samples", 0), cur.get("samples", 0)) < min_samples:
            row["baseline_ms"] = base["value_ms"]
            row["note"] = f"fewer than {min_samples} samples"
        else:
            b, c = float(base["value_ms"]), float(cur["value_ms"])
            row["baseline_ms"] = b
            row["regression_percent"] = round((c - b) / b * 100, 1) if b > 0 else None
            if max(b, c) < noise_floor_ms:
                row["status"], row["note"] = PASS, "below noise floor"
            elif b <= 0:
                row["note"] = "baseline is zero"
            elif (c - b) / b * 100 > threshold_percent:
                row["status"] = REGRESSION
            else:
                row["status"] = PASS
        rows.append(row)
    return {"verdict": overall(rows), "threshold_percent": threshold_percent, "rows": rows}


def overall(rows: list[dict[str, Any]]) -> str:
    statuses = {r["status"] for r in rows}
    if REGRESSION in statuses:
        return REGRESSION          # a proven regression is never hidden by missing data elsewhere
    if not rows or statuses == {INSUFFICIENT} or INSUFFICIENT in statuses:
        return INSUFFICIENT        # never PASS on partial evidence
    return PASS


EXIT = {PASS: 0, REGRESSION: 1, INSUFFICIENT: 2, INFRA: 3}


def _load(path: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two benchmark_scale runs.")
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--threshold", type=float, default=None, help="percent; default from settings")
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--noise-floor-ms", type=float, default=1.0)
    args = ap.parse_args(argv)
    base, cur = _load(args.baseline), _load(args.current)
    if base is None or cur is None:
        print(f"{INFRA}: cannot read {'baseline' if base is None else 'current'} file")
        return EXIT[INFRA]
    threshold = args.threshold
    if threshold is None:
        ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ROOT))
        from app.core.config import get_settings
        threshold = get_settings().performance_regression_threshold_percent
    result = compare(base, cur, threshold, args.min_samples, args.noise_floor_ms)
    print(f"threshold: {threshold}%")
    for r in result["rows"]:
        print(f"{r['metric']:<34}{str(r['baseline_ms']):>10}{str(r['current_ms']):>10}"
              f"{str(r['regression_percent']):>9}%  {r['status']}  {r.get('note', '')}")
    print(f"VERDICT: {result['verdict']}" + (f" ({result['reason']})" if "reason" in result else ""))
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
