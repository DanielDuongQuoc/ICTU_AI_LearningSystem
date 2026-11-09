"""Code-checked rubric rules (no LLM). R3 (coverage) and R5 (observable descriptors) need judgement and are
checked by the AI review step in :mod:`app.services.rubric_service`.

R1  sum of criterion max_points == rubric max_score
R2  every criterion has >= 3 levels, strictly decreasing points, top == max_points, bottom == 0, unique labels
R4  no two criteria assess the same thing (near-duplicate names / key points)
R6  a grounded rubric carries source_chunks on every criterion
ID  criterion ids are unique
PASS_RULE  (homework) the pass rule can actually fail a submission
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from app.schemas.rubric import AssessmentContext, Rubric, RubricViolation

MIN_LEVELS = 3
NEAR_DUPLICATE_RATIO = 0.9
_TOL = 1e-6
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def _similar(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= NEAR_DUPLICATE_RATIO)


def validate_rubric(rubric: Rubric, assessment_context: Optional[AssessmentContext] = None) -> list[RubricViolation]:
    out: list[RubricViolation] = []

    def add(rule: str, message: str, cid: Optional[str] = None) -> None:
        out.append(RubricViolation(rule=rule, criterion_id=cid, message=message))

    # ID
    seen_ids: set[str] = set()
    for c in rubric.criteria:
        if c.criterion_id in seen_ids:
            add("ID", f"criterion_id '{c.criterion_id}' is used more than once", c.criterion_id)
        seen_ids.add(c.criterion_id)

    # R1
    total = sum(c.max_points for c in rubric.criteria)
    if abs(total - rubric.max_score) > _TOL:
        add("R1", f"criteria max_points sum to {total:g} but max_score is {rubric.max_score:g}")

    # R2
    for c in rubric.criteria:
        pts = [lv.points for lv in c.levels]
        if len(c.levels) < MIN_LEVELS:
            add("R2", f"needs at least {MIN_LEVELS} levels, has {len(c.levels)}", c.criterion_id)
        if any(b >= a - _TOL for a, b in zip(pts, pts[1:])):
            add("R2", "level points must be strictly decreasing from the first to the last level", c.criterion_id)
        if pts and abs(pts[0] - c.max_points) > _TOL:
            add("R2", f"top level is worth {pts[0]:g} but max_points is {c.max_points:g}", c.criterion_id)
        if pts and abs(pts[-1]) > _TOL:
            add("R2", f"lowest level must be worth 0, is {pts[-1]:g}", c.criterion_id)
        if any(p > c.max_points + _TOL for p in pts):
            add("R2", "a level is worth more than max_points", c.criterion_id)
        labels = [_norm(lv.label) for lv in c.levels]
        if len(set(labels)) != len(labels):
            add("R2", "level labels must be unique within a criterion", c.criterion_id)

    # R4
    crits = rubric.criteria
    for i in range(len(crits)):
        for j in range(i + 1, len(crits)):
            a, b = crits[i], crits[j]
            same_name = _similar(_norm(a.name), _norm(b.name))
            pa, pb = {_norm(p) for p in a.key_points}, {_norm(p) for p in b.key_points}
            same_points = bool(pa) and pa == pb
            if same_name or same_points:
                add("R4", f"criteria '{a.criterion_id}' and '{b.criterion_id}' assess the same thing", b.criterion_id)

    # R6
    if rubric.grounded:
        for c in rubric.criteria:
            if not c.source_chunks:
                add("R6", "grounded rubric: criterion has no source_chunks", c.criterion_id)

    # PASS_RULE
    if assessment_context == "homework":
        has_required = any(c.required for c in rubric.criteria)
        rule = rubric.pass_rule
        if not ((rule.all_required_met and has_required) or rule.min_percentage > 0):
            add("PASS_RULE", "homework rubric needs a required criterion or min_percentage > 0, "
                             "otherwise every submission would pass")
    return out
