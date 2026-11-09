"""Rubric-based grading with the LLM: the model picks ONE level label per criterion; everything numeric
(points, totals, PASS/FAIL) is computed by the backend from the rubric.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core import prompt_registry, runtime_config
from app.core.exceptions import AppError, ErrorCode
from app.core.output_guard import MAX_FEEDBACK_CHARS, clamp_text, enforce_no_prompt_leak
from app.schemas.rubric import Rubric, RubricCriterion, RubricLevel
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

MAX_LLM_ATTEMPTS = 2
MAX_REASON_CHARS = 600
MAX_EVIDENCE_CHARS = 400
REQUIRED_KEYS = ["criteria", "feedback"]


def _label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def rubric_prompt_json(rubric: Rubric) -> str:
    """Compact rubric view for the prompt (no backend-only fields such as source_chunks)."""
    return json.dumps(
        [{
            "criterion_id": c.criterion_id, "name": c.name, "description": c.description,
            "key_points": c.key_points,
            "levels": [{"label": lv.label, "points": lv.points, "descriptor": lv.descriptor} for lv in c.levels],
        } for c in rubric.criteria],
        ensure_ascii=False, indent=1,
    )


@dataclass
class CriterionJudgement:
    criterion: RubricCriterion
    level: RubricLevel
    reason: str
    evidence: str


@dataclass
class GradingJudgement:
    criteria: list[CriterionJudgement]
    feedback: str
    injection_suspected: bool = False
    off_topic: bool = False
    prompt_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def parse_judgement(raw: Any, rubric: Rubric) -> GradingJudgement:
    """Validate untrusted model JSON against the rubric; raise GRADING_INVALID_RESPONSE when unusable."""
    bad = AppError(ErrorCode.GRADING_INVALID_RESPONSE, "The language model returned an invalid grading result.")
    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        raise bad
    by_id = {c.criterion_id: c for c in rubric.criteria}
    seen: dict[str, CriterionJudgement] = {}
    for item in raw["criteria"]:
        if not isinstance(item, dict):
            raise bad
        cid, level = item.get("criterion_id"), item.get("level")
        if not isinstance(cid, str) or cid not in by_id or cid in seen or not isinstance(level, str):
            raise bad
        crit = by_id[cid]
        match = next((lv for lv in crit.levels if _label_key(lv.label) == _label_key(level)), None)
        if match is None:
            raise bad
        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        evidence = item.get("evidence") if isinstance(item.get("evidence"), str) else ""
        seen[cid] = CriterionJudgement(crit, match, clamp_text(reason.strip(), MAX_REASON_CHARS),
                                       clamp_text(evidence.strip(), MAX_EVIDENCE_CHARS))
    if set(seen) != set(by_id):
        raise bad
    feedback = raw.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise bad
    judgements = [seen[c.criterion_id] for c in rubric.criteria]
    enforce_no_prompt_leak(feedback, *(j.reason for j in judgements))
    return GradingJudgement(
        criteria=judgements,
        feedback=clamp_text(feedback.strip(), MAX_FEEDBACK_CHARS),
        injection_suspected=raw.get("injection_suspected") is True,
        off_topic=raw.get("off_topic") is True,
    )


class RubricGrader:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    async def judge(
        self,
        *,
        question: str,
        rubric: Rubric,
        answer: str,
        expected_answer: str = "",
        context: str = "",
        submission_kind: str = "typed",
        language: str = "vi",
        test_results: str = "",
        request_id: Optional[str] = None,
    ) -> GradingJudgement:
        version = runtime_config.current().prompt_version(prompt_registry.TASK_RUBRIC_GRADING)
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_RUBRIC_GRADING, version,
            question=question, rubric_json=rubric_prompt_json(rubric), answer=answer,
            expected_answer=expected_answer, context=context, submission_kind=submission_kind,
            language=language, test_results=test_results,
        )
        last: Optional[AppError] = None
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            raw = await self._llm.generate_json(system_prompt, user_prompt, required_keys=REQUIRED_KEYS)
            try:
                judgement = parse_judgement(raw, rubric)
                judgement.prompt_version = f"{prompt_registry.TASK_RUBRIC_GRADING}:{version}"
                return judgement
            except AppError as exc:
                last = exc
                logger.warning("Invalid grading output | request_id=%s attempt=%d/%d code=%s",
                               request_id, attempt, MAX_LLM_ATTEMPTS, exc.code)
        assert last is not None
        raise last
