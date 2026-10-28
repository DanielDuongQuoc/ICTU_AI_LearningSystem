"""Phase 7 — Answer evaluation.

Pipeline
~~~~~~~~
request → validate (no RAG/LLM on empty input) → RAGService.retrieve (document-scoped,
query = question + expected answer) → insufficient-context check (LLM NOT called)
→ evaluation prompt → LLMService.generate_json → strict Pydantic validation
→ score range / consistency checks → backend percentage → backend sources.

The LLM provides only score / correct / feedback / strengths / missing_points / errors.
``percentage`` and ``sources`` are always computed by the backend; anything else the
LLM returns (e.g. fake ``sources``) is ignored.  The student answer is untrusted and is
only ever placed in the STUDENT ANSWER block of the user prompt.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, Optional
from uuid import UUID

from app.core import prompt_registry, runtime_config
from app.core.execution_trace import trace_execution
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.output_guard import (
    MAX_FEEDBACK_CHARS, clamp_list, clamp_text, enforce_no_prompt_leak,
)
from app.schemas.evaluation import EvaluationResponse, EvaluationSource, LLMEvaluationResult
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service
from app.utils.json_utils import safe_parse_json
from app.utils.sse import done_event, error_event, result_event, status_event, token_event

logger = logging.getLogger(__name__)

MAX_ALLOWED_SCORE = 100.0
MAX_LLM_ATTEMPTS = 2  # first try + exactly one retry
MAX_RETRIEVAL_QUERY_CHARS = 1000
REQUIRED_KEYS = ["score", "correct", "feedback", "strengths", "missing_points", "errors"]


def build_retrieval_query(question: str, expected_answer: str) -> str:
    """Retrieval query: the question plus the reference answer.

    The student answer is deliberately NOT used (untrusted; could steer retrieval).
    """
    return f"{question.strip()}\n{expected_answer.strip()}"[:MAX_RETRIEVAL_QUERY_CHARS]


def validate_llm_result(raw: Any, max_score: float) -> LLMEvaluationResult:
    started = time.perf_counter()
    try:
        return _validate_llm_result(raw, max_score)
    finally:
        metrics.observe_stage("validation", time.perf_counter() - started)


def _validate_llm_result(raw: Any, max_score: float) -> LLMEvaluationResult:
    """Validate raw LLM JSON; raise AppError (EVALUATION_INVALID_RESPONSE / INVALID_SCORE)."""
    if not isinstance(raw, dict):
        raise AppError(ErrorCode.EVALUATION_INVALID_RESPONSE, "The language model returned an invalid evaluation.")
    try:
        result = LLMEvaluationResult.model_validate(raw)
    except Exception as exc:
        raise AppError(
            ErrorCode.EVALUATION_INVALID_RESPONSE, "The language model returned an invalid evaluation."
        ) from exc

    score = float(result.score)
    if not math.isfinite(score) or score < 0 or score > max_score:
        raise AppError(
            ErrorCode.INVALID_SCORE,
            "The language model returned a score outside the allowed range.",
            status_code=502,
        )
    # Reject clearly contradictory verdicts rather than silently trusting them.
    if (result.correct and score == 0) or (not result.correct and score == max_score):
        raise AppError(
            ErrorCode.EVALUATION_INVALID_RESPONSE,
            "The language model returned an inconsistent evaluation.",
        )
    # Phase 19: model text is untrusted - reject verbatim system-prompt disclosure, bound field sizes.
    enforce_no_prompt_leak(result.feedback, *result.strengths, *result.missing_points, *result.errors)
    return result.model_copy(update={
        "feedback": clamp_text(result.feedback, MAX_FEEDBACK_CHARS),
        "strengths": clamp_list(result.strengths),
        "missing_points": clamp_list(result.missing_points),
        "errors": clamp_list(result.errors),
    })


class EvaluationService:
    """Orchestrates RAG retrieval + LLM evaluation + backend validation."""

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self._rag = rag_service or get_rag_service()
        self._llm = llm_service or get_llm_service()

    @staticmethod
    def _validate_request(
        document_id: Any, question: Any, student_answer: Any, expected_answer: Any, max_score: Any
    ) -> tuple[str, str, str, str, float]:
        if document_id is None or not str(document_id).strip():
            raise AppError(ErrorCode.EMPTY_DOCUMENT_ID, "document_id must not be empty.")
        if not isinstance(question, str) or not question.strip():
            raise AppError(ErrorCode.EMPTY_QUESTION, "question must not be empty.")
        if not isinstance(student_answer, str) or not student_answer.strip():
            raise AppError(ErrorCode.EMPTY_STUDENT_ANSWER, "student_answer must not be empty.")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise AppError(ErrorCode.EMPTY_EXPECTED_ANSWER, "expected_answer must not be empty.")
        limits = get_settings()
        for name, value, limit in (
            ("question", question, limits.max_question_chars),
            ("student_answer", student_answer, limits.max_answer_chars),
            ("expected_answer", expected_answer, limits.max_answer_chars),
        ):
            if len(value.strip()) > limit:
                raise AppError(ErrorCode.INPUT_TOO_LONG, f"{name} must be at most {limit} characters.")
        if (
            isinstance(max_score, bool)
            or not isinstance(max_score, (int, float))
            or not math.isfinite(max_score)
            or not 0 < max_score <= MAX_ALLOWED_SCORE
        ):
            raise AppError(ErrorCode.INVALID_SCORE, "max_score must be greater than 0 and at most 100.")
        return (
            str(document_id).strip(), question.strip(), student_answer.strip(),
            expected_answer.strip(), float(max_score),
        )

    async def evaluate_answer(
        self,
        document_id: UUID | str,
        question: str,
        student_answer: str,
        expected_answer: str,
        max_score: float = 10,
        request_id: Optional[str] = None,
    ) -> EvaluationResponse:
        request_id = request_id or str(uuid.uuid4())
        # Validation happens before any RAG / LLM call.
        doc_id, question, student_answer, expected_answer, max_score = self._validate_request(
            document_id, question, student_answer, expected_answer, max_score
        )
        started = time.perf_counter()

        try:
            # 1. Retrieval (document-scoped; also verifies the document exists)
            t0 = time.perf_counter()
            retrieval = await self._rag.retrieve(
                doc_id,
                build_retrieval_query(question, expected_answer),
                top_k=runtime_config.current().top_k,
                request_id=request_id,
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000

            # 2. Insufficient context → never call the LLM
            if retrieval["insufficient_context"] or not retrieval["chunks"]:
                logger.info(
                    "Evaluation skipped | request_id=%s document_id=%s insufficient_context=True "
                    "llm_called=False retrieval_ms=%.1f",
                    request_id, doc_id, retrieval_ms,
                )
                raise AppError(
                    ErrorCode.INSUFFICIENT_CONTEXT,
                    "The document does not contain enough information to evaluate this answer.",
                )

            # 3. Source integrity: every chunk must belong to the requested document
            sources = self._build_sources(doc_id, retrieval["chunks"])

            # 4. LLM evaluation + validation (one retry on invalid output)
            t0 = time.perf_counter()
            result = await self._evaluate_with_llm(
                retrieval["context"], question, student_answer, expected_answer, max_score, request_id
            )
            llm_ms = (time.perf_counter() - t0) * 1000
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "Evaluation failed | request_id=%s document_id=%s: %s",
                request_id, doc_id, exc, exc_info=True,
            )
            raise AppError(ErrorCode.EVALUATION_ERROR, "Answer evaluation failed.") from exc

        score = float(result.score)
        response = self._build_response(result, max_score, sources)
        logger.info(
            "Evaluation finished | request_id=%s document_id=%s model=%s prompt_version=%s "
            "max_score=%s score=%s retrieved_chunks=%d retrieval_ms=%.1f llm_ms=%.1f total_ms=%.1f",
            request_id, doc_id, self._llm.model_name,
            runtime_config.current().prompt_version(prompt_registry.TASK_ANSWER_EVALUATION),
            max_score, score, len(sources), retrieval_ms, llm_ms,
            (time.perf_counter() - started) * 1000,
        )
        trace_execution(
            prompt_registry.TASK_ANSWER_EVALUATION, request_id, doc_id, "ok",
            (time.perf_counter() - started) * 1000,
            source_chunk_ids=[s.chunk_id for s in sources], llm_called=True,
        )
        return response

    @staticmethod
    def _build_response(
        result: LLMEvaluationResult, max_score: float, sources: list[EvaluationSource]
    ) -> EvaluationResponse:
        """Final response: percentage / max_score / sources are backend-owned."""
        score = float(result.score)
        return EvaluationResponse(
            score=score,
            max_score=max_score,
            percentage=round(score / max_score * 100, 2),
            correct=result.correct,
            feedback=result.feedback,
            strengths=[s.strip() for s in result.strengths if s.strip()],
            missing_points=[s.strip() for s in result.missing_points if s.strip()],
            errors=[s.strip() for s in result.errors if s.strip()],
            sources=sources,
        )

    # ── Phase 8: streaming (SSE) ────────────────────────────────────

    async def evaluate_answer_stream(
        self,
        document_id: UUID | str,
        question: str,
        student_answer: str,
        expected_answer: str,
        max_score: float = 10,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield SSE frames: status… token… result, or error, and always one final ``done``.

        Every expected failure becomes an ``error`` event (code + safe message only);
        stack traces stay in the server log.  Cancellation (client disconnect) is logged
        and re-raised so Starlette can stop the stream; no ``done`` is sent then.
        """
        request_id = request_id or str(uuid.uuid4())
        stats: dict[str, Any] = {
            "started": time.perf_counter(), "first_token_ms": None, "chunks": 0,
            "retrieval_ms": None, "score": None, "retrieved": 0,
        }
        outcome = "ok"
        try:
            events = self._stream_events(
                document_id, question, student_answer, expected_answer, max_score,
                request_id, stats,
            )
            # aclosing: on client disconnect the inner generator (and with it the upstream
            # LLM stream) is closed immediately instead of waiting for garbage collection.
            async with aclosing(events):
                async for frame in events:
                    yield frame
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "client_disconnected"
            logger.warning(
                "Evaluation stream cancelled | request_id=%s document_id=%s", request_id, document_id
            )
            raise
        except AppError as exc:
            outcome = exc.code
            metrics.error(exc.code)
            yield error_event(exc.code, exc.message)
        except Exception as exc:
            outcome = ErrorCode.EVALUATION_ERROR
            metrics.error(ErrorCode.EVALUATION_ERROR)
            logger.error(
                "Evaluation stream failed | request_id=%s document_id=%s: %s",
                request_id, document_id, exc, exc_info=True,
            )
            yield error_event(ErrorCode.EVALUATION_ERROR, "Answer evaluation failed.")
        finally:
            first = stats["first_token_ms"]
            last = stats.get("last_token_ms")
            total_ms = (time.perf_counter() - stats["started"]) * 1000
            metrics.stream_finished(
                "ok" if outcome == "ok" else "client_disconnected" if outcome == "client_disconnected" else "error"
            )
            stream_ms = (last - first) if (first is not None and last is not None) else None
            logger.info(
                "Evaluation stream | request_id=%s document_id=%s endpoint=/api/evaluations/evaluate/stream "
                "outcome=%s model=%s prompt_version=%s retrieved_chunks=%d retrieval_ms=%s "
                "first_token_latency_ms=%s last_token_ms=%s stream_duration_ms=%s "
                "token_chunks=%d token_count=%d score=%s total_latency_ms=%.1f",
                request_id, document_id, outcome, self._llm.model_name,
                runtime_config.current().prompt_version(prompt_registry.TASK_ANSWER_EVALUATION),
                stats["retrieved"],
                "null" if stats["retrieval_ms"] is None else f"{stats['retrieval_ms']:.1f}",
                "null" if first is None else f"{first:.1f}",
                "null" if last is None else f"{last:.1f}",
                "null" if stream_ms is None else f"{stream_ms:.1f}",
                stats["chunks"], stats["chunks"], stats["score"],
                total_ms,
            )
            trace_execution(
                prompt_registry.TASK_ANSWER_EVALUATION, request_id, str(document_id), str(outcome),
                total_ms, source_chunk_ids=None, llm_called=stats["first_token_ms"] is not None,
            )
            threshold = get_settings().slow_request_threshold_ms
            if total_ms > threshold:
                logger.warning(
                    "Slow stream | request_id=%s total_latency_ms=%.1f threshold_ms=%s",
                    request_id, total_ms, threshold,
                )
        yield done_event()

    async def _stream_events(
        self,
        document_id: Any,
        question: Any,
        student_answer: Any,
        expected_answer: Any,
        max_score: Any,
        request_id: str,
        stats: dict[str, Any],
    ) -> AsyncIterator[str]:
        yield status_event("Validating request")
        doc_id, question, student_answer, expected_answer, max_score = self._validate_request(
            document_id, question, student_answer, expected_answer, max_score
        )
        try:
            UUID(doc_id)
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_DOCUMENT_ID, "document_id must be a valid UUID.") from exc

        yield status_event("Retrieving relevant context")
        t0 = time.perf_counter()
        retrieval = await self._rag.retrieve(
            doc_id,
            build_retrieval_query(question, expected_answer),
            top_k=runtime_config.current().top_k,
            request_id=request_id,
        )
        stats["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        stats["retrieved"] = len(retrieval["chunks"])

        if retrieval["insufficient_context"] or not retrieval["chunks"]:
            yield status_event("Insufficient context")
            raise AppError(
                ErrorCode.INSUFFICIENT_CONTEXT,
                "The document does not contain enough information to evaluate this answer.",
            )
        sources = self._build_sources(doc_id, retrieval["chunks"])

        yield status_event("Evaluating answer")
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_ANSWER_EVALUATION,
            runtime_config.current().prompt_version(prompt_registry.TASK_ANSWER_EVALUATION),
            retrieval["context"], question, student_answer, expected_answer, max_score
        )

        # Forward each piece immediately AND keep it so the complete output can be parsed.
        buffer: list[str] = []
        received_chars = 0
        limits = get_settings()
        stream = self._llm.generate_stream(system_prompt, user_prompt, json_mode=True)
        try:
            async for piece in stream:
                if not isinstance(piece, str) or not piece:
                    continue
                # Phase 19: a stream cannot run (or grow) without bound.
                received_chars += len(piece)
                if received_chars > limits.max_llm_output_chars:
                    raise AppError(ErrorCode.LLM_OUTPUT_REJECTED, "The language model output exceeded the allowed size.")
                if time.perf_counter() - stats["started"] > limits.stream_max_duration_s:
                    raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.")
                now_ms = (time.perf_counter() - stats["started"]) * 1000
                if stats["first_token_ms"] is None:
                    stats["first_token_ms"] = now_ms
                stats["last_token_ms"] = now_ms
                stats["chunks"] += 1
                buffer.append(piece)
                yield token_event(piece)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

        full_output = "".join(buffer)
        if not full_output.strip():
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty response.")

        # Streamed tokens are never the evaluation: parse + validate the complete output.
        try:
            result = validate_llm_result(safe_parse_json(full_output), max_score)
        except (ValueError, AppError) as exc:
            if isinstance(exc, AppError) and exc.code not in (
                ErrorCode.EVALUATION_INVALID_RESPONSE, ErrorCode.INVALID_SCORE
            ):
                raise
            logger.warning(
                "Streamed evaluation invalid, retrying once non-streaming | request_id=%s", request_id
            )
            yield status_event("Retrying structured evaluation")
            raw = await self._llm.generate_json(
                system_prompt, user_prompt, required_keys=REQUIRED_KEYS
            )
            result = validate_llm_result(raw, max_score)  # still invalid → error event

        response = self._build_response(result, max_score, sources)
        stats["score"] = response.score
        yield result_event(response.model_dump(mode="json"))

    @staticmethod
    def _build_sources(document_id: str, chunks: list[dict[str, Any]]) -> list[EvaluationSource]:
        sources: list[EvaluationSource] = []
        for c in chunks:
            if c.get("document_id") != document_id:
                logger.error(
                    "Source integrity violation: chunk %s belongs to another document",
                    c.get("chunk_id"),
                )
                raise AppError(ErrorCode.SOURCE_INTEGRITY_ERROR, "Source verification failed.")
            sources.append(
                EvaluationSource(page=c.get("page"), chunk_id=c["chunk_id"], score=c["score"])
            )
        return sources

    async def _evaluate_with_llm(
        self,
        context: str,
        question: str,
        student_answer: str,
        expected_answer: str,
        max_score: float,
        request_id: str,
    ) -> LLMEvaluationResult:
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_ANSWER_EVALUATION,
            runtime_config.current().prompt_version(prompt_registry.TASK_ANSWER_EVALUATION),
            context, question, student_answer, expected_answer, max_score
        )
        last_error: Optional[AppError] = None
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            # Ollama / empty-response / invalid-JSON AppErrors propagate unchanged.
            raw = await self._llm.generate_json(
                system_prompt, user_prompt, required_keys=REQUIRED_KEYS
            )
            try:
                return validate_llm_result(raw, max_score)
            except AppError as exc:
                last_error = exc
                logger.warning(
                    "Invalid evaluation from LLM | request_id=%s attempt=%d/%d code=%s",
                    request_id, attempt, MAX_LLM_ATTEMPTS, exc.code,
                )
        assert last_error is not None
        raise last_error


# ── Module-level singleton ──────────────────────────────────────────

_evaluation_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    """Return a lazily-created service reusing the shared RAG and LLM singletons."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
