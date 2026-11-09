"""Rubric-based grading orchestration.

homework  -> pass_fail: per-criterion met / not met -> backend PASS/FAIL -> ``final`` (``needs_review`` if flagged)
in_class  -> score:     per-criterion level -> backend points + total   -> always ``pending_review``

The submission is untrusted: it only appears in the STUDENT ANSWER block and is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.grading.flags import detect_injection, is_empty_or_copy, split_flags
from app.grading.grader import CriterionJudgement, GradingJudgement, RubricGrader
from app.grading.policy import decide_pass, grading_mode, result_status, unmet_required
from app.grading.rubric_validator import validate_rubric
from app.schemas.grading import (
    BatchError, BatchItemResult, CriterionResult, GradeBatchRequest, GradeBatchResponse, GradeRequest,
    GradeResult, GradeTrace,
)
from app.schemas.rubric import AssessmentContext, Rubric
from app.services.context_builder import build_context
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

BATCH_CONCURRENCY = 4
_EPS = 1e-9
EMPTY_REASON = "Không có nội dung bài làm để đánh giá."
EMPTY_FEEDBACK = "Bài làm trống hoặc không có nội dung liên quan đến câu hỏi."


def check_rubric_usable(rubric: Rubric, ctx: AssessmentContext) -> None:
    if rubric.status != "approved":
        raise AppError(ErrorCode.RUBRIC_NOT_APPROVED, "Only an approved rubric can be used for grading.")
    violations = validate_rubric(rubric, ctx)
    if violations:
        raise AppError(ErrorCode.RUBRIC_INVALID, "The rubric violates the rubric rules: "
                       + "; ".join(f"[{v.rule}] {v.message}" for v in violations[:8]))


def _empty_judgement(rubric: Rubric) -> GradingJudgement:
    return GradingJudgement(
        criteria=[CriterionJudgement(c, c.levels[-1], EMPTY_REASON, "") for c in rubric.criteria],
        feedback=EMPTY_FEEDBACK, off_topic=True, prompt_version="none:empty-submission",
    )


def build_result(
    *,
    ctx: AssessmentContext,
    rubric: Rubric,
    judgement: GradingJudgement,
    flags: list[str],
    model: str,
    request_id: str,
    submission_ref: Optional[str] = None,
) -> GradeResult:
    """Apply the grading policy. Pure function: everything numeric is derived from the rubric here."""
    ratio = get_settings().grading_criterion_met_ratio
    mode = grading_mode(ctx)
    met = {j.criterion.criterion_id: j.level.points + _EPS >= ratio * j.criterion.max_points
           for j in judgement.criteria}
    score_mode = mode == "score"
    criteria = [
        CriterionResult(
            criterion_id=j.criterion.criterion_id, name=j.criterion.name, required=j.criterion.required,
            level=j.level.label, met=met[j.criterion.criterion_id],
            points=j.level.points if score_mode else None,
            max_points=j.criterion.max_points if score_mode else None,
            reason=j.reason, evidence=j.evidence,
        )
        for j in judgement.criteria
    ]
    review, warnings = split_flags(flags)
    status = result_status(ctx, review)
    common = dict(
        submission_ref=submission_ref, assessment_context=ctx, grading_mode=mode, status=status,
        criteria=criteria, feedback=judgement.feedback, flags=review, warnings=warnings,
        trace=GradeTrace(model=model, prompt_version=judgement.prompt_version, request_id=request_id,
                         grounded=rubric.grounded),
    )
    if score_mode:
        total = round(sum(j.level.points for j in judgement.criteria), 2)
        return GradeResult(**common, suggested_total=total, max_score=rubric.max_score)
    return GradeResult(**common, result=decide_pass(met, rubric), unmet_required=unmet_required(met, rubric))


class GradingService:
    def __init__(self, rag_service: Optional[RAGService] = None, llm_service: Optional[LLMService] = None) -> None:
        self._rag = rag_service
        self._llm = llm_service or get_llm_service()
        self._grader = RubricGrader(self._llm)

    def _rag_service(self) -> RAGService:
        if self._rag is None:
            self._rag = get_rag_service()
        return self._rag

    @staticmethod
    def _check_inputs(question: str, answer: str) -> None:
        s = get_settings()
        if not question.strip():
            raise AppError(ErrorCode.EMPTY_QUESTION, "question must not be empty.")
        if len(question.strip()) > s.max_question_chars:
            raise AppError(ErrorCode.INPUT_TOO_LONG, f"question must be at most {s.max_question_chars} characters.")
        if len(answer) > s.max_submission_chars:
            raise AppError(ErrorCode.INPUT_TOO_LONG,
                           f"the submission must be at most {s.max_submission_chars} characters.")

    async def _context(self, document_ids, question: str, expected: str, request_id: str) -> str:
        if not document_ids:
            return ""
        context, _ = await build_context(self._rag_service(), document_ids, f"{question}\n{expected}", request_id)
        return context

    async def _grade_one(
        self, *, ctx: AssessmentContext, question: str, rubric: Rubric, answer: str, expected: str,
        context: str, kind: str, language: str, test_results: str, extra_flags: list[str],
        submission_ref: Optional[str], request_id: str,
    ) -> GradeResult:
        self._check_inputs(question, answer)
        started = time.perf_counter()
        flags = list(extra_flags)
        if not rubric.grounded and ctx == "in_class":
            flags.append("ungrounded_rubric")
        if not answer.strip() or (kind != "code" and is_empty_or_copy(answer, question)):
            judgement = _empty_judgement(rubric)
            llm_called = False
        else:
            judgement = await self._grader.judge(
                question=question, rubric=rubric, answer=answer, expected_answer=expected, context=context,
                submission_kind=kind, language=language, test_results=test_results, request_id=request_id)
            llm_called = True
        if judgement.injection_suspected or detect_injection(answer):
            flags.append("injection_suspected")
        if judgement.off_topic:
            flags.append("empty_or_off_topic")
        result = build_result(ctx=ctx, rubric=rubric, judgement=judgement, flags=flags,
                              model=self._llm.model_name, request_id=request_id, submission_ref=submission_ref)
        logger.info("Grading finished | request_id=%s context=%s kind=%s status=%s result=%s total=%s flags=%s "
                    "llm_called=%s answer_chars=%d total_ms=%.1f", request_id, ctx, kind, result.status,
                    result.result, result.suggested_total, ",".join(result.flags) or "-", llm_called,
                    len(answer), (time.perf_counter() - started) * 1000)
        return result

    async def grade(self, req: GradeRequest, request_id: Optional[str] = None) -> GradeResult:
        request_id = request_id or str(uuid.uuid4())
        check_rubric_usable(req.rubric, req.assessment_context)
        expected = (req.expected_answer or "").strip()
        context = await self._context(req.document_ids, req.question, expected, request_id)
        return await self._grade_one(
            ctx=req.assessment_context, question=req.question.strip(), rubric=req.rubric, answer=req.answer.strip(),
            expected=expected, context=context, kind=req.submission_kind, language=req.language,
            test_results=(req.test_results or "").strip(), extra_flags=req.extra_flags,
            submission_ref=req.submission_ref, request_id=request_id)

    async def grade_batch(self, req: GradeBatchRequest, request_id: Optional[str] = None) -> GradeBatchResponse:
        request_id = request_id or str(uuid.uuid4())
        limit = get_settings().max_batch_submissions
        if len(req.submissions) > limit:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"at most {limit} submissions per batch.")
        check_rubric_usable(req.rubric, req.assessment_context)
        expected = (req.expected_answer or "").strip()
        context = await self._context(req.document_ids, req.question, expected, request_id)
        sem = asyncio.Semaphore(BATCH_CONCURRENCY)

        async def one(sub) -> BatchItemResult:
            async with sem:
                try:
                    res = await self._grade_one(
                        ctx=req.assessment_context, question=req.question.strip(), rubric=req.rubric,
                        answer=sub.answer.strip(), expected=expected, context=context, kind=sub.submission_kind,
                        language=req.language, test_results=(sub.test_results or "").strip(),
                        extra_flags=sub.extra_flags, submission_ref=sub.submission_ref,
                        request_id=f"{request_id}:{sub.submission_ref}")
                    return BatchItemResult(submission_ref=sub.submission_ref, ok=True, result=res)
                except AppError as exc:
                    return BatchItemResult(submission_ref=sub.submission_ref, ok=False,
                                           error=BatchError(code=exc.code, message=exc.message))
                except Exception:
                    logger.exception("Batch grading item failed | request_id=%s", request_id)
                    return BatchItemResult(submission_ref=sub.submission_ref, ok=False,
                                           error=BatchError(code=ErrorCode.GRADING_ERROR, message="Grading failed."))

        results = await asyncio.gather(*(one(s) for s in req.submissions))
        ok = sum(1 for r in results if r.ok)
        return GradeBatchResponse(total=len(results), succeeded=ok, failed=len(results) - ok, results=list(results))


_grading_service: Optional[GradingService] = None


def get_grading_service() -> GradingService:
    global _grading_service
    if _grading_service is None:
        _grading_service = GradingService()
    return _grading_service
