"""Grading policy: the assessment context decides HOW a submission is graded and whether the result is final.

* ``homework``  -> ``pass_fail``: AI / Agent grade automatically; the verdict is PASS or FAIL and is ``final``,
  unless a review flag was raised (then ``needs_review``).
* ``in_class``  -> ``score``: AI proposes per-criterion points; the result is ALWAYS ``pending_review`` and the
  lecturer decides the final score.

The caller never chooses the grading mode: it is derived here from ``assessment_context`` only.
"""

from __future__ import annotations

from typing import Iterable, Literal, Mapping

from app.schemas.rubric import AssessmentContext, Rubric

GradingMode = Literal["pass_fail", "score"]
ResultStatus = Literal["final", "needs_review", "pending_review"]
Verdict = Literal["PASS", "FAIL"]

_EPS = 1e-9


def grading_mode(ctx: AssessmentContext) -> GradingMode:
    if ctx == "homework":
        return "pass_fail"
    if ctx == "in_class":
        return "score"
    raise ValueError(f"unknown assessment_context {ctx!r}")


def met_percentage(criteria_met: Mapping[str, bool], rubric: Rubric) -> float:
    """Share (0..100) of the rubric weight whose criteria are met."""
    total = sum(c.max_points for c in rubric.criteria)
    if total <= 0:
        return 0.0
    met = sum(c.max_points for c in rubric.criteria if criteria_met.get(c.criterion_id, False))
    return round(met / total * 100, 2)


def unmet_required(criteria_met: Mapping[str, bool], rubric: Rubric) -> list[str]:
    return [c.criterion_id for c in rubric.criteria if c.required and not criteria_met.get(c.criterion_id, False)]


def decide_pass(criteria_met: Mapping[str, bool], rubric: Rubric) -> Verdict:
    """Backend-owned homework verdict (the model only says met / not met per criterion)."""
    rule = rubric.pass_rule
    if rule.all_required_met and unmet_required(criteria_met, rubric):
        return "FAIL"
    if met_percentage(criteria_met, rubric) + _EPS < rule.min_percentage:
        return "FAIL"
    return "PASS"


def result_status(ctx: AssessmentContext, flags: Iterable[str]) -> ResultStatus:
    if ctx == "in_class":
        return "pending_review"  # never final: the lecturer decides
    return "needs_review" if any(True for _ in flags) else "final"
