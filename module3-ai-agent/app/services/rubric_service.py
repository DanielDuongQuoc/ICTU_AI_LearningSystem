"""Rubric generation: (optional) retrieval -> LLM draft -> code rules -> one repair -> optional AI review.

The model writes criteria text and level descriptors only. ``source_chunks``, ``grounded``, ``status`` and the
pass rule are set by the backend; for in-class tests every criterion is forced to ``required=False``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from pydantic import ValidationError

from app.core import prompt_registry, runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.output_guard import clamp_text, contains_prompt_leak
from app.grading.rubric_validator import validate_rubric
from app.schemas.grading import RubricGenerateRequest, RubricGenerateResponse
from app.schemas.rubric import Rubric, RubricViolation
from app.services.context_builder import build_context, pick_sources
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

MAX_REPAIRS = 1
_TEXT_FIELDS = ("name", "description", "key_points")


def _check_lengths(req: RubricGenerateRequest) -> None:
    s = get_settings()
    if len(req.question.strip()) > s.max_question_chars:
        raise AppError(ErrorCode.INPUT_TOO_LONG, f"question must be at most {s.max_question_chars} characters.")
    if len(req.expected_answer.strip()) > s.max_answer_chars:
        raise AppError(ErrorCode.INPUT_TOO_LONG, f"expected_answer must be at most {s.max_answer_chars} characters.")


class RubricService:
    def __init__(self, rag_service: Optional[RAGService] = None, llm_service: Optional[LLMService] = None) -> None:
        self._rag = rag_service
        self._llm = llm_service or get_llm_service()

    def _rag_service(self) -> RAGService:
        if self._rag is None:
            self._rag = get_rag_service()
        return self._rag

    # ── public ──────────────────────────────────────────────────────

    async def generate(self, req: RubricGenerateRequest, request_id: Optional[str] = None) -> RubricGenerateResponse:
        request_id = request_id or str(uuid.uuid4())
        _check_lengths(req)
        started = time.perf_counter()

        context, chunks = "", []
        if req.document_ids:
            context, chunks = await build_context(
                self._rag_service(), req.document_ids, f"{req.question}\n{req.expected_answer}", request_id)
            if not chunks:
                raise AppError(ErrorCode.INSUFFICIENT_CONTEXT,
                               "The reference documents do not contain enough information for this question.")
        grounded = bool(chunks)
        course_data = req.course_context.as_data_block() if req.course_context else ""

        version = runtime_config.current().prompt_version(prompt_registry.TASK_RUBRIC_GENERATION)
        violations: list[str] = []
        rubric: Optional[Rubric] = None
        for attempt in range(MAX_REPAIRS + 1):
            system_prompt, user_prompt = prompt_registry.build(
                prompt_registry.TASK_RUBRIC_GENERATION, version,
                question=req.question.strip(), expected_answer=req.expected_answer.strip(),
                max_score=req.max_score, assessment_context=req.assessment_context, context=context,
                course_data=course_data, language=req.language, violations=violations or None,
            )
            raw = await self._llm.generate_json(system_prompt, user_prompt, required_keys=["criteria"])
            candidate, problems = self._build_rubric(raw, req, grounded, chunks)
            if candidate is not None and not problems:
                rubric = candidate
                break
            violations = problems
            logger.warning("Rubric draft rejected | request_id=%s attempt=%d problems=%d",
                           request_id, attempt + 1, len(problems))
        if rubric is None:
            raise AppError(ErrorCode.RUBRIC_INVALID,
                           "The generated rubric violates the rubric rules: " + "; ".join(violations[:8]))

        review_applied, issues = False, []
        if get_settings().rubric_review_enabled:
            rubric, review_applied, issues = await self._review(rubric, req, request_id)

        logger.info("Rubric generated | request_id=%s context=%s criteria=%d grounded=%s review_applied=%s "
                    "total_ms=%.1f", request_id, req.assessment_context, len(rubric.criteria), grounded,
                    review_applied, (time.perf_counter() - started) * 1000)
        return RubricGenerateResponse(
            rubric=rubric, review_applied=review_applied, review_issues=issues,
            model=self._llm.model_name, prompt_version=f"{prompt_registry.TASK_RUBRIC_GENERATION}:{version}",
        )

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _build_rubric(
        raw: Any, req: RubricGenerateRequest, grounded: bool, chunks: list[dict[str, Any]],
    ) -> tuple[Optional[Rubric], list[str]]:
        if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
            return None, ["the output must be a JSON object with a criteria list"]
        criteria = []
        for item in raw["criteria"]:
            if not isinstance(item, dict):
                return None, ["every criterion must be a JSON object"]
            item = {k: v for k, v in item.items() if k != "source_chunks"}  # backend-owned
            if req.assessment_context == "in_class":
                item["required"] = False
            criteria.append(item)
        try:
            rubric = Rubric.model_validate({
                "max_score": req.max_score, "criteria": criteria, "grounded": grounded, "status": "draft",
            })
        except ValidationError as exc:
            msgs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]]
            return None, msgs
        texts = [c.name + " " + c.description + " " + " ".join(c.key_points) for c in rubric.criteria]
        texts += [lv.descriptor for c in rubric.criteria for lv in c.levels]
        if contains_prompt_leak(*texts):
            return None, ["the rubric must not repeat the instructions"]
        if grounded:
            for c in rubric.criteria:
                c.source_chunks = pick_sources(f"{c.name} {c.description} {' '.join(c.key_points)}", chunks)
        problems = [f"[{v.rule}{'/' + v.criterion_id if v.criterion_id else ''}] {v.message}"
                    for v in validate_rubric(rubric, req.assessment_context)]
        return rubric, problems

    async def _review(
        self, rubric: Rubric, req: RubricGenerateRequest, request_id: str,
    ) -> tuple[Rubric, bool, list[RubricViolation]]:
        """Best effort: any failure keeps the already valid rubric."""
        try:
            version = runtime_config.current().prompt_version(prompt_registry.TASK_RUBRIC_REVIEW)
            rubric_json = json.dumps(
                [c.model_dump(exclude={"source_chunks"}) for c in rubric.criteria], ensure_ascii=False, indent=1)
            system_prompt, user_prompt = prompt_registry.build(
                prompt_registry.TASK_RUBRIC_REVIEW, version, question=req.question.strip(),
                expected_answer=req.expected_answer.strip(), rubric_json=rubric_json, language=req.language)
            raw = await self._llm.generate_json(system_prompt, user_prompt, required_keys=["issues", "criteria"])
        except AppError as exc:
            logger.warning("Rubric review skipped | request_id=%s code=%s", request_id, exc.code)
            return rubric, False, []

        issues: list[RubricViolation] = []
        for it in raw.get("issues", []) if isinstance(raw, dict) and isinstance(raw.get("issues"), list) else []:
            if isinstance(it, dict) and isinstance(it.get("message"), str) and it["message"].strip():
                rule = it.get("rule") if it.get("rule") in ("R3", "R4", "R5") else "R5"
                cid = it.get("criterion_id") if isinstance(it.get("criterion_id"), str) else None
                issues.append(RubricViolation(rule=rule, criterion_id=cid, message=clamp_text(it["message"].strip(), 500)))
        issues = issues[:20]

        revised = self._merge_review(rubric, raw.get("criteria") if isinstance(raw, dict) else None)
        if revised is None:
            return rubric, False, issues
        if validate_rubric(revised, req.assessment_context):
            logger.warning("Rubric review output broke the rubric rules; keeping the draft | request_id=%s", request_id)
            return rubric, False, issues
        changed = revised.model_dump() != rubric.model_dump()
        return (revised if changed else rubric), changed, issues

    @staticmethod
    def _merge_review(rubric: Rubric, items: Any) -> Optional[Rubric]:
        """Take only text from the review; ids, points, required flags and sources stay backend-controlled."""
        if not isinstance(items, list):
            return None
        by_id = {i.get("criterion_id"): i for i in items if isinstance(i, dict)}
        data = rubric.model_dump()
        for c in data["criteria"]:
            new = by_id.get(c["criterion_id"])
            if not isinstance(new, dict):
                continue
            for f in _TEXT_FIELDS:
                if f in new:
                    c[f] = new[f]
            new_levels = new.get("levels")
            if isinstance(new_levels, list) and len(new_levels) == len(c["levels"]):
                for old, nl in zip(c["levels"], new_levels):
                    if isinstance(nl, dict) and isinstance(nl.get("descriptor"), str) and nl["descriptor"].strip():
                        old["descriptor"] = nl["descriptor"]
        try:
            merged = Rubric.model_validate(data)
        except ValidationError:
            return None
        texts = [lv.descriptor for c in merged.criteria for lv in c.levels] + [c.name for c in merged.criteria]
        return None if contains_prompt_leak(*texts) else merged


_rubric_service: Optional[RubricService] = None


def get_rubric_service() -> RubricService:
    global _rubric_service
    if _rubric_service is None:
        _rubric_service = RubricService()
    return _rubric_service
