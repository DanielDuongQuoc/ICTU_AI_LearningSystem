"""Rubric schema, code-checked rubric rules (R1/R2/R4/R6/ID/PASS_RULE) and the grading policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.grading.policy import decide_pass, grading_mode, met_percentage, result_status, unmet_required
from app.grading.rubric_validator import validate_rubric
from app.schemas.rubric import Rubric


def _levels(maxp: float) -> list[dict]:
    return [
        {"label": "Đầy đủ", "points": maxp, "descriptor": "Nêu đủ 3 ý, đúng thuật ngữ"},
        {"label": "Một phần", "points": round(maxp / 2, 2), "descriptor": "Nêu đúng 1-2 ý"},
        {"label": "Không đạt", "points": 0, "descriptor": "Sai hoặc không đề cập"},
    ]


def make_rubric(**over) -> Rubric:
    data = {
        "max_score": 10,
        "grounded": False,
        "criteria": [
            {"criterion_id": "C1", "name": "Nêu đúng khái niệm", "max_points": 4, "required": True,
             "key_points": ["định nghĩa"], "levels": _levels(4)},
            {"criterion_id": "C2", "name": "Giải thích cơ chế hoạt động", "max_points": 4,
             "key_points": ["cơ chế"], "levels": _levels(4)},
            {"criterion_id": "C3", "name": "Cho ví dụ minh họa", "max_points": 2,
             "key_points": ["ví dụ"], "levels": _levels(2)},
        ],
    }
    data.update(over)
    return Rubric.model_validate(data)


def rules(rubric: Rubric, ctx=None) -> set[str]:
    return {v.rule for v in validate_rubric(rubric, ctx)}


# ---------------------------------------------------------------- validator

def test_valid_rubric_has_no_violations():
    assert validate_rubric(make_rubric(), "homework") == []
    assert validate_rubric(make_rubric(), "in_class") == []


def test_r1_points_must_sum_to_max_score():
    assert "R1" in rules(make_rubric(max_score=9))


def test_r2_level_structure():
    r = make_rubric()
    r.criteria[0].levels = r.criteria[0].levels[:2]                       # too few
    r.criteria[1].levels[1].points = 4                                     # not strictly decreasing
    r.criteria[2].levels[-1].points = 0.5                                  # bottom != 0
    vs = [v for v in validate_rubric(r) if v.rule == "R2"]
    assert {v.criterion_id for v in vs} == {"C1", "C2", "C3"}


def test_r2_top_level_must_equal_max_points_and_labels_unique():
    r = make_rubric()
    r.criteria[0].levels[0].points = 3
    r.criteria[1].levels[1].label = "Đầy đủ"
    msgs = " ".join(v.message for v in validate_rubric(r) if v.rule == "R2")
    assert "top level" in msgs and "unique" in msgs


def test_r4_duplicate_criteria_detected():
    r = make_rubric()
    r.criteria[1].name = "Nêu đúng khái niệm."
    assert "R4" in rules(r)
    r = make_rubric()
    r.criteria[2].key_points = ["cơ chế"]
    assert "R4" in rules(r)


def test_r6_grounded_rubric_needs_sources():
    assert "R6" in rules(make_rubric(grounded=True))
    r = make_rubric(grounded=True)
    for c in r.criteria:
        c.source_chunks = ["doc_chunk_1"]
    assert "R6" not in rules(r)


def test_duplicate_criterion_ids():
    r = make_rubric()
    r.criteria[2].criterion_id = "C1"
    assert "ID" in rules(r)


def test_homework_pass_rule_must_be_able_to_fail():
    r = make_rubric(pass_rule={"all_required_met": True, "min_percentage": 0})
    for c in r.criteria:
        c.required = False
    assert "PASS_RULE" in rules(r, "homework")
    assert "PASS_RULE" not in rules(r, "in_class")


def test_schema_rejects_bad_basic_fields():
    with pytest.raises(ValidationError):
        make_rubric(criteria=[])
    with pytest.raises(ValidationError):
        Rubric.model_validate({"criteria": [{"criterion_id": "bad id!", "name": "x", "max_points": 1,
                                             "levels": _levels(1)}]})


# ---------------------------------------------------------------- policy

def test_grading_mode_is_derived_from_context():
    assert grading_mode("homework") == "pass_fail"
    assert grading_mode("in_class") == "score"
    with pytest.raises(ValueError):
        grading_mode("exam")  # type: ignore[arg-type]


def test_pass_requires_required_criteria_and_threshold():
    r = make_rubric()
    assert decide_pass({"C1": True, "C2": True, "C3": False}, r) == "PASS"      # 80%
    assert decide_pass({"C1": False, "C2": True, "C3": True}, r) == "FAIL"      # required C1 unmet (60%)
    assert unmet_required({"C1": False}, r) == ["C1"]
    assert decide_pass({"C1": True, "C2": False, "C3": False}, r) == "FAIL"     # 40% < 50%
    assert met_percentage({"C1": True, "C3": True}, r) == 60.0
    assert decide_pass({"C1": True, "C3": True}, r) == "PASS"                    # exactly above threshold


def test_threshold_is_inclusive():
    r = make_rubric(pass_rule={"all_required_met": True, "min_percentage": 40})
    assert decide_pass({"C1": True}, r) == "PASS"                                # 40% == 40%


@pytest.mark.parametrize("ctx,flags,expected", [
    ("homework", [], "final"),
    ("homework", ["injection_suspected"], "needs_review"),
    ("in_class", [], "pending_review"),
    ("in_class", ["injection_suspected"], "pending_review"),
])
def test_result_status(ctx, flags, expected):
    assert result_status(ctx, flags) == expected


def test_in_class_is_never_final():
    for flags in ([], ["x"], ["a", "b"]):
        assert result_status("in_class", flags) != "final"
