"""Prompt evaluation against golden cases (calls the REAL Gemini API, uses quota).

    python scripts/prompt_eval.py                      # all tests/golden/*.yaml, 1 run per case
    python scripts/prompt_eval.py --repeat 3           # stability: same verdict / score spread
    python scripts/prompt_eval.py --dry-run            # validate the dataset only (no model calls)
    python scripts/prompt_eval.py --only hw_ --out reports/eval.md

Metrics: expectation pass rate, homework PASS/FAIL agreement, in-class MAE vs lecturer_score,
stability over repeats (homework verdict unchanged, in-class spread <= 0.5), injection resistance.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.grading.rubric_validator import validate_rubric  # noqa: E402
from app.schemas.grading import GradeRequest  # noqa: E402
from app.schemas.rubric import Rubric  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden"
STABLE_SPREAD = 0.5


@dataclass
class Case:
    id: str
    request: GradeRequest
    expect: dict[str, Any]


@dataclass
class CaseResult:
    case: Case
    runs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ctx(self) -> str:
        return self.case.request.assessment_context

    def check(self) -> list[str]:
        """Expectation failures of the FIRST successful run (empty list = pass)."""
        if not self.runs:
            return ["no successful run"] + self.errors
        r, e, fails = self.runs[0], self.case.expect, []
        if "result" in e and r.get("result") != e["result"]:
            fails.append(f"result {r.get('result')} != {e['result']}")
        if "status" in e and r.get("status") != e["status"]:
            fails.append(f"status {r.get('status')} != {e['status']}")
        for f in e.get("flags_include", []):
            if f not in r.get("flags", []):
                fails.append(f"missing flag {f}")
        if "total_range" in e and r.get("suggested_total") is not None:
            lo, hi = e["total_range"]
            if not lo <= r["suggested_total"] <= hi:
                fails.append(f"total {r['suggested_total']} not in [{lo}, {hi}]")
        return fails

    def stable(self) -> Optional[bool]:
        if len(self.runs) < 2:
            return None
        if self.ctx == "homework":
            return len({(r.get("result"), r.get("status")) for r in self.runs}) == 1
        totals = [r["suggested_total"] for r in self.runs]
        return max(totals) - min(totals) <= STABLE_SPREAD


def load_cases(directory: Path = GOLDEN_DIR, only: str = "") -> list[Case]:
    cases: list[Case] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        defaults = data.get("defaults", {})
        for raw in data.get("cases", []):
            item = {**defaults, **raw}
            if only and only not in item["id"]:
                continue
            rubric_path = (path.parent / item.pop("rubric_file")) if "rubric_file" in item else None
            rubric = Rubric.model_validate(
                json.loads(rubric_path.read_text(encoding="utf-8")) if rubric_path else item.pop("rubric"))
            problems = validate_rubric(rubric, item["assessment_context"])
            if problems:
                raise ValueError(f"{path.name}:{item['id']}: invalid rubric: {problems}")
            expect = item.pop("expect", {})
            case_id = item.pop("id")
            req = GradeRequest(rubric=rubric, submission_ref=case_id.replace(" ", "_")[:64],
                               **{k: v for k, v in item.items() if k in GradeRequest.model_fields})
            cases.append(Case(case_id, req, expect))
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


async def evaluate(cases: list[Case], service: Any, repeat: int = 1) -> list[CaseResult]:
    from app.core.exceptions import AppError

    results = []
    for case in cases:
        cr = CaseResult(case)
        for _ in range(repeat):
            try:
                res = await service.grade(case.request)
                cr.runs.append(res.model_dump(mode="json"))
            except AppError as exc:
                cr.errors.append(f"{exc.code}: {exc.message}")
        results.append(cr)
    return results


def metrics(results: list[CaseResult]) -> dict[str, Any]:
    passed = sum(1 for r in results if not r.check())
    hw = [r for r in results if r.ctx == "homework" and r.runs and "result" in r.case.expect]
    ic = [r for r in results if r.ctx == "in_class" and r.runs and "lecturer_score" in r.case.expect]
    stab = [s for s in (r.stable() for r in results) if s is not None]
    inj = [r for r in results if "injection_suspected" in r.case.expect.get("flags_include", [])]
    return {
        "cases": len(results),
        "expectations_passed": passed,
        "homework_agreement": (sum(r.runs[0]["result"] == r.case.expect["result"] for r in hw) / len(hw)) if hw else None,
        "in_class_mae": (statistics.mean(abs(r.runs[0]["suggested_total"] - r.case.expect["lecturer_score"])
                                         for r in ic)) if ic else None,
        "stability": (sum(stab) / len(stab)) if stab else None,
        "injection_resisted": (sum(1 for r in inj if not r.check()) / len(inj)) if inj else None,
        "errors": sum(len(r.errors) for r in results),
    }


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def report_markdown(results: list[CaseResult], m: dict[str, Any], repeat: int, elapsed_s: float) -> str:
    mae = "-" if m["in_class_mae"] is None else f"{m['in_class_mae']:.2f}"
    lines = [
        f"# Prompt evaluation - {datetime.now():%Y-%m-%d %H:%M}", "",
        f"Cases: {m['cases']} | repeat: {repeat} | time: {elapsed_s:.0f}s | errors: {m['errors']}", "",
        "| Metric | Value | Target |", "|---|---|---|",
        f"| Expectations passed | {m['expectations_passed']}/{m['cases']} | all |",
        f"| Homework PASS/FAIL agreement | {_pct(m['homework_agreement'])} | >= 90% |",
        f"| In-class MAE vs lecturer | {mae} | <= 1.0 |",
        f"| Stability over repeats | {_pct(m['stability'])} | 100% |",
        f"| Injection resisted | {_pct(m['injection_resisted'])} | 100% |", "",
        "| Case | Context | Result | Status | Total | Flags | Check |", "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        first = r.runs[0] if r.runs else {}
        fails = r.check()
        lines.append(f"| {r.case.id} | {r.ctx} | {first.get('result') or ''} | {first.get('status', 'ERROR')} | "
                     f"{'' if first.get('suggested_total') is None else first['suggested_total']} | "
                     f"{','.join(first.get('flags', []))} | {'OK' if not fails else '; '.join(fails)} |")
    return "\n".join(lines) + "\n"


def write_csv(results: list[CaseResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "context", "run", "result", "status", "suggested_total", "flags", "levels"])
        for r in results:
            for i, run in enumerate(r.runs, 1):
                w.writerow([r.case.id, r.ctx, i, run.get("result"), run.get("status"), run.get("suggested_total"),
                            ",".join(run.get("flags", [])),
                            " ".join(f"{c['criterion_id']}={c['level']}" for c in run.get("criteria", []))])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="", help="substring filter on case ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="markdown report path (default reports/prompt_eval_<ts>.md)")
    args = ap.parse_args(argv)
    cases = load_cases(only=args.only)
    print(f"{len(cases)} golden cases loaded")
    if args.dry_run:
        return 0

    from app.services.grading_service import GradingService
    from app.services.llm_service import get_llm_service

    async def go() -> list[CaseResult]:
        try:
            return await evaluate(cases, GradingService(rag_service=object()), args.repeat)  # type: ignore[arg-type]
        finally:
            await get_llm_service().close()

    started = time.perf_counter()
    results = asyncio.run(go())
    m = metrics(results)
    out = Path(args.out) if args.out else ROOT / "reports" / f"prompt_eval_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_markdown(results, m, args.repeat, time.perf_counter() - started), encoding="utf-8")
    write_csv(results, out.with_suffix(".csv"))
    print(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"report: {out}")
    return 0 if m["expectations_passed"] == m["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
