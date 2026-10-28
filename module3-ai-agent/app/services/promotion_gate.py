"""Promotion gate (Phase 18): turns Phase 17 evaluation evidence into ONE verdict for a candidate version.

Pure functions, no I/O and no model calls, so the decision is deterministic and unit-testable.  The
evidence itself (Phase 17 report JSON + a functional-test result) is produced host-side by
``scripts/ai_version_cli.py evaluate``; the Docker image contains only ``app/``, which is why the gate
logic lives here and not under ``scripts/``.

Verdicts (only ``PASS`` allows ``validated`` -> ``approved`` -> ``active``)::

    PASS             every required check ran on real (ollama-mode) data and is satisfied
    FAIL_SAFETY      a source-integrity or document-isolation invariant was violated
    FAIL_INTEGRITY   evidence does not belong to this candidate / is malformed / dataset mismatch
    FAIL_FUNCTIONAL  the functional test run failed
    FAIL_QUALITY     a tracked metric regressed more than the tolerance vs the baseline, or an
                     operator-defined absolute floor is not met
    INCONCLUSIVE     evidence exists but cannot support a conclusion (offline harness run, too few
                     cases, no baseline and no floors, baseline not comparable)
    BLOCKED          required evidence is missing or the evaluation could not run (infrastructure)

Statistics: the tolerance is a fixed absolute margin, NOT a significance test.  With 30-40 cases per
suite, sampling noise on a rate is several percentage points, so a PASS means "no regression larger than
the margin was observed", never "statistically equivalent".  Nothing here claims otherwise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

SUITES = ("rag", "qgen", "answer_eval")
# metric -> direction (+1 higher is better, -1 lower is better). Same metrics as scripts/quality_gate.py.
TRACKED: dict[str, dict[str, int]] = {
    "rag": {"hit@5": 1, "mrr": 1, "insufficient_context_detection_rate": 1, "incorrect_abstention_rate": -1},
    "qgen": {"answerability_rate": 1, "groundedness_rate": 1, "duplication_rate": -1},
    "answer_eval": {"within_band_rate": 1, "correct_label_accuracy": 1,
                    "over_scoring_rate_wrong_answers": -1, "under_scoring_rate_paraphrases": -1},
}
INVARIANTS = ("source_integrity_violations", "document_isolation_violations")
# Report prompt_version key (Phase 17) -> prompt registry task
_PROMPT_KEYS = {"rag": "rag_answer", "question_generation": "question_generation", "evaluation": "answer_evaluation"}


class GatePolicy(BaseModel):
    """Configurable promotion criteria.  Defaults are deliberately conservative and EXPLAINED:

    * ``min_cases`` 30: Phase 17 already treats <30 cases as insufficient for a quality conclusion.
    * ``tolerance`` 0.05: absolute drop allowed vs the baseline; same default as the Phase 17 gate.
    * ``floors``: EMPTY by default.  The repository has no model-backed baseline, so any absolute
      number would be arbitrary.  Operators may add floors (e.g. after measuring a real baseline).
    * ``require_baseline``: when true and no floors cover a suite, the verdict is INCONCLUSIVE.
    """

    model_config = ConfigDict(extra="forbid")

    min_cases: dict[str, int] = Field(default_factory=lambda: {s: 30 for s in SUITES})
    tolerance: float = Field(0.05, ge=0.0, le=0.5)
    floors: dict[str, dict[str, float]] = Field(default_factory=dict)
    require_baseline: bool = True
    require_functional_tests: bool = True

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class GateResult(BaseModel):
    verdict: str
    reasons: list[str]
    suites: dict[str, dict[str, Any]] = {}
    baseline_used: bool = False


def _cases(report: dict[str, Any]) -> int:
    for key in ("per_case", "per_question"):
        if isinstance(report.get(key), list):
            return len(report[key])
    value = report.get("cases")
    return int(value) if isinstance(value, int) else 0


def _metric(report: dict[str, Any], name: str) -> Optional[float]:
    value = (report.get("metrics_flat") or {}).get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mismatch(candidate: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    """Report metadata must describe THIS candidate (model, embedding, retrieval, prompts)."""
    problems = []
    checks = (
        ("model_name", candidate["llm"]["model"]),
        ("embedding_model", candidate["embedding_model"]),
        ("top_k", candidate["rag"]["top_k"]),
        ("similarity_threshold", candidate["rag"]["similarity_threshold"]),
    )
    for key, expected in checks:
        if meta.get(key) != expected:
            problems.append(f"{key}: report={meta.get(key)!r} candidate={expected!r}")
    for key, version in (meta.get("prompt_version") or {}).items():
        task = _PROMPT_KEYS.get(key)
        if task and candidate["prompt_versions"].get(task) != version:
            problems.append(f"prompt {task}: report={version!r} candidate={candidate['prompt_versions'].get(task)!r}")
    return problems


def evaluate_evidence(
    candidate_config: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    functional: Optional[dict[str, Any]],
    baseline_reports: Optional[dict[str, dict[str, Any]]],
    policy: GatePolicy,
    infrastructure_errors: Optional[list[str]] = None,
) -> GateResult:
    """Decide the verdict.  ``reports`` maps suite -> Phase 17 report dict (may be partial)."""
    suites: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    # 1. hard safety invariants (any report present, even offline / tiny)
    safety = [
        f"{s}: {name}={rep.get('invariants', {}).get(name)}"
        for s, rep in reports.items() for name in INVARIANTS if rep.get("invariants", {}).get(name)
    ]
    if safety:
        return GateResult(verdict="FAIL_SAFETY", reasons=safety)

    # 2. evidence integrity: must belong to this candidate and share one dataset version
    integrity: list[str] = []
    for s, rep in reports.items():
        meta = rep.get("metadata") or {}
        if not meta or "metrics_flat" not in rep:
            integrity.append(f"{s}: report is malformed (no metadata/metrics)")
            continue
        if meta.get("mode") == "ollama":  # offline reports use scripted stand-ins by design
            integrity += [f"{s}: {p}" for p in _mismatch(candidate_config, meta)]
    versions = {(rep.get("metadata") or {}).get("dataset_version") for rep in reports.values()}
    if len(versions) > 1:
        integrity.append(f"reports use different dataset versions: {sorted(str(v) for v in versions)}")
    if integrity:
        return GateResult(verdict="FAIL_INTEGRITY", reasons=integrity)

    # 3. functional tests
    if functional is not None and not functional.get("passed", False):
        return GateResult(verdict="FAIL_FUNCTIONAL", reasons=[
            f"functional tests failed ({functional.get('failed', '?')} failed of {functional.get('total', '?')})"])

    # 4. missing evidence / infrastructure -> BLOCKED
    blocked = list(infrastructure_errors or [])
    blocked += [f"no evaluation evidence for suite '{s}'" for s in SUITES if s not in reports]
    if policy.require_functional_tests and functional is None:
        blocked.append("no functional test result")
    if blocked:
        return GateResult(verdict="BLOCKED", reasons=blocked)

    # 5. evidence that cannot support a conclusion -> INCONCLUSIVE
    inconclusive: list[str] = []
    for s in SUITES:
        rep = reports[s]
        mode = (rep.get("metadata") or {}).get("mode")
        n = _cases(rep)
        suites[s] = {"mode": mode, "cases": n, "run_id": (rep.get("metadata") or {}).get("evaluation_run_id", ""),
                     "metrics": {m: _metric(rep, m) for m in TRACKED[s]}}
        if mode != "ollama":
            inconclusive.append(f"{s}: report mode is {mode!r}; only a real-model (ollama) run measures quality")
        elif n < policy.min_cases.get(s, 30):
            inconclusive.append(f"{s}: {n} cases < required {policy.min_cases.get(s, 30)}")
    if inconclusive:
        return GateResult(verdict="INCONCLUSIVE", reasons=inconclusive, suites=suites)

    # 6. quality: floors and baseline regression
    failures: list[str] = []
    gaps: list[str] = []
    base_used = False
    for s in SUITES:
        rep = reports[s]
        floors = policy.floors.get(s, {})
        base = (baseline_reports or {}).get(s)
        covered = bool(floors) or base is not None
        if not covered and policy.require_baseline:
            gaps.append(f"{s}: no comparable baseline and no absolute floors configured")
        for metric, floor in floors.items():
            value = _metric(rep, metric)
            direction = TRACKED[s].get(metric, 1)
            if value is None:
                gaps.append(f"{s}.{metric}: metric missing from report")
            elif (floor - value) * direction > 0:
                failures.append(f"{s}.{metric}: {value:.4f} violates floor {floor}")
        if base is not None:
            if (base.get("metadata") or {}).get("dataset_version") != (rep.get("metadata") or {}).get("dataset_version"):
                gaps.append(f"{s}: baseline used another dataset version; not comparable")
                continue
            if (base.get("metadata") or {}).get("mode") != "ollama":
                gaps.append(f"{s}: baseline is not a real-model run; not comparable")
                continue
            base_used = True
            for metric, direction in TRACKED[s].items():
                cur, ref = _metric(rep, metric), _metric(base, metric)
                if cur is None or ref is None:
                    continue
                if (ref - cur) * direction > policy.tolerance:
                    failures.append(
                        f"{s}.{metric}: baseline {ref:.4f} -> candidate {cur:.4f} (tolerance {policy.tolerance})")
    if failures:
        return GateResult(verdict="FAIL_QUALITY", reasons=failures, suites=suites, baseline_used=base_used)
    if gaps:
        return GateResult(verdict="INCONCLUSIVE", reasons=gaps, suites=suites, baseline_used=base_used)
    reasons.append(f"all suites evaluated on real-model data; no regression beyond tolerance {policy.tolerance} "
                   "(fixed margin, not a significance test)")
    return GateResult(verdict="PASS", reasons=reasons, suites=suites, baseline_used=base_used)
