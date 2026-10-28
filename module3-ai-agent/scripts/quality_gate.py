"""Phase 17 - AI quality gate.

Reads the newest evaluation report of each suite (``rag``, ``qgen``, ``answer_eval``) and prints
exactly one verdict.  Functional test failures are the job of ``pytest`` (a separate CI step); this
gate covers data integrity, safety invariants and quality regression.

Verdicts and exit codes::

    0   PASS                       quality verified against a comparable baseline (or, with
                                   --deterministic-only, deterministic checks passed - the quality
                                   line then still says INSUFFICIENT EVIDENCE, never PASS)
    10  INSUFFICIENT EVIDENCE      no report / offline-only / dataset too small / no comparable baseline
    11  QUALITY REGRESSION         a metric fell more than --tolerance below the baseline
    12  DATA INTEGRITY FAILURE     a dataset fails schema validation
    13  SEVERE: SOURCE/ISOLATION   a source-integrity or document-isolation violation was recorded
    14  INFRASTRUCTURE FAILURE     a report could not be read

Rules: it never reports PASS when the evaluator did not run; reports from different dataset
versions/models/prompts/top-k/threshold are NOT compared (they are reported as not comparable);
it never changes the production model or prompt.

Usage::

    python scripts/quality_gate.py [--reports evaluation/reports] [--baseline-dir evaluation/baselines]
        [--tolerance 0.05] [--deterministic-only] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_common as ec  # noqa: E402

EXIT_PASS, EXIT_INSUFFICIENT, EXIT_REGRESSION, EXIT_DATA, EXIT_SEVERE, EXIT_INFRA = 0, 10, 11, 12, 13, 14

SUITES = ("rag", "qgen", "answer_eval")
# Minimum cases before a quality conclusion is meaningful. Smaller sets can only show gross failures.
MIN_CASES = {"rag": 30, "qgen": 30, "answer_eval": 30}
# metric -> direction (+1: higher is better, -1: lower is better)
TRACKED = {
    "rag": {"hit@5": 1, "mrr": 1, "insufficient_context_detection_rate": 1, "incorrect_abstention_rate": -1},
    "qgen": {"answerability_rate": 1, "groundedness_rate": 1, "duplication_rate": -1},
    "answer_eval": {"within_band_rate": 1, "correct_label_accuracy": 1,
                    "over_scoring_rate_wrong_answers": -1, "under_scoring_rate_paraphrases": -1},
}
COMPARABLE_KEYS = ("dataset_version", "model_name", "embedding_model", "top_k", "similarity_threshold", "prompt_version")


def latest_reports(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            run_id = data["metadata"]["evaluation_run_id"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"unreadable report {path.name}: {exc}") from exc
        suite = run_id.split("-")[0] if not run_id.startswith("answer_eval") else "answer_eval"
        if suite in SUITES:
            old = found.get(suite)
            if old is None or data["metadata"]["timestamp"] > old[1]["metadata"]["timestamp"]:
                found[suite] = (path, data)
    return found


def validate_datasets() -> list[str]:
    problems = []
    checks = (
        ("rag_eval.jsonl", ec.validate_rag_cases),
        ("question_generation_eval.jsonl", ec.validate_qgen_cases),
        ("answer_evaluation_eval.jsonl", ec.validate_eval_cases),
    )
    try:
        corpus = ec.load_corpus()
    except ec.DatasetError as exc:
        return [str(exc)]
    for name, fn in checks:
        try:
            fn(ec.load_jsonl(ec.DATASET_DIR / name), corpus)
        except ec.DatasetError as exc:
            problems.append(str(exc))
    return problems


def not_comparable(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    return [
        f"{k}: baseline={baseline.get(k)!r} current={current.get(k)!r}"
        for k in COMPARABLE_KEYS if baseline.get(k) != current.get(k)
    ]


def regressions(suite: str, current: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> list[str]:
    out = []
    for metric, direction in TRACKED[suite].items():
        cur, base = current["metrics_flat"].get(metric), baseline["metrics_flat"].get(metric)
        if cur is None or base is None:
            continue
        if (base - cur) * direction > tolerance:
            out.append(f"{suite}.{metric}: baseline {base:.4f} -> current {cur:.4f} (tolerance {tolerance})")
    return out


def run_gate(
    reports_dir: Path, baseline_dir: Optional[Path], tolerance: float, deterministic_only: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {"suites": {}, "reasons": [], "quality": "INSUFFICIENT EVIDENCE"}

    problems = validate_datasets()
    if problems:
        result.update(verdict="DATA INTEGRITY FAILURE", exit_code=EXIT_DATA, reasons=problems)
        return result
    try:
        reports = latest_reports(reports_dir) if reports_dir.is_dir() else {}
        baselines = latest_reports(baseline_dir) if baseline_dir and baseline_dir.is_dir() else {}
    except RuntimeError as exc:
        result.update(verdict="INFRASTRUCTURE FAILURE", exit_code=EXIT_INFRA, reasons=[str(exc)])
        return result

    severe = []
    for suite, (path, rep) in reports.items():
        inv = rep.get("invariants", {})
        for name in ("source_integrity_violations", "document_isolation_violations"):
            if inv.get(name):
                severe.append(f"{suite} ({path.name}): {name}={inv[name]}")
    if severe:
        result.update(verdict="SEVERE FAILURE: SOURCE/ISOLATION", exit_code=EXIT_SEVERE, reasons=severe)
        return result

    missing = [s for s in SUITES if s not in reports]
    if missing:
        result["reasons"].append(f"no evaluation report for: {', '.join(missing)} (the evaluator did not run)")

    regress: list[str] = []
    verified: list[str] = []
    for suite, (path, rep) in reports.items():
        meta = rep["metadata"]
        info: dict[str, Any] = {"report": path.name, "mode": meta["mode"], "cases": rep.get("cases") or
                                len(rep.get("per_case", rep.get("per_question", [])))}
        result["suites"][suite] = info
        if meta["mode"] != "ollama":
            info["quality"] = "NOT EVALUATED (offline harness run; no model quality measured)"
            result["reasons"].append(f"{suite}: offline report - quality not measured")
            continue
        if info["cases"] < MIN_CASES[suite]:
            info["quality"] = f"INSUFFICIENT EVIDENCE ({info['cases']} < {MIN_CASES[suite]} cases)"
            result["reasons"].append(f"{suite}: dataset too small for a quality conclusion")
            continue
        base = baselines.get(suite)
        if base is None:
            info["quality"] = "INSUFFICIENT EVIDENCE (no baseline)"
            result["reasons"].append(f"{suite}: no baseline report in {baseline_dir}")
            continue
        diffs = not_comparable(meta, base[1]["metadata"])
        if diffs:
            info["quality"] = "NOT COMPARABLE"
            result["reasons"].append(f"{suite}: configuration differs from baseline -> {'; '.join(diffs)}")
            continue
        found = regressions(suite, rep, base[1], tolerance)
        info["quality"] = "REGRESSION" if found else "VERIFIED vs baseline"
        regress += found
        if not found:
            verified.append(suite)

    if regress:
        result.update(verdict="QUALITY REGRESSION", exit_code=EXIT_REGRESSION, reasons=regress + result["reasons"])
        return result
    if len(verified) == len(SUITES):
        result.update(verdict="PASS", exit_code=EXIT_PASS, quality="VERIFIED vs baseline")
        return result
    if deterministic_only and not missing:
        result.update(
            verdict="DETERMINISTIC CHECKS PASS (datasets valid, no source/isolation violation); "
                    "QUALITY: INSUFFICIENT EVIDENCE",
            exit_code=EXIT_PASS,
        )
        return result
    result.update(verdict="INSUFFICIENT EVIDENCE", exit_code=EXIT_INSUFFICIENT)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AI quality gate (never PASSes without evidence).")
    p.add_argument("--reports", default=str(ec.DEFAULT_REPORT_DIR))
    p.add_argument("--baseline-dir", default=str(ROOT / "evaluation" / "baselines"))
    p.add_argument("--tolerance", type=float, default=0.05, help="Absolute metric drop allowed vs baseline")
    p.add_argument("--deterministic-only", action="store_true",
                   help="CI mode: exit 0 when datasets are valid and no safety invariant failed")
    p.add_argument("--json", default=None, help="Also write the verdict as JSON")
    args = p.parse_args(argv)
    result = run_gate(Path(args.reports), Path(args.baseline_dir), args.tolerance, args.deterministic_only)
    print(f"QUALITY GATE: {result['verdict']}")
    for suite, info in result["suites"].items():
        print(f"  {suite}: {info['mode']} | {info['cases']} cases | {info['quality']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
