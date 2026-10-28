"""Phase 6 — Question generation.

Pipeline
~~~~~~~~
request → validate → RAGService.retrieve (document-scoped) → insufficient-context
check (LLM NOT called) → prompt → LLMService.generate_json → Pydantic validation
→ per-type validation → duplicate removal → backend question_id
→ backend-derived source_chunks (retrieval by question text) → response.

The LLM only produces question text/answers.  ``question_id`` and ``source_chunks``
are always created by the backend; any such keys in the LLM output are ignored.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
import uuid
from typing import Any, Optional
from uuid import UUID

from app.core import prompt_registry, runtime_config
from app.core.execution_trace import trace_execution
from app.core.output_guard import contains_prompt_leak
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.prompts import (
    QUESTION_GENERATION_RETRIEVAL_QUERY,
)
from app.schemas.question import (
    ExamMatrixItem,
    ExamMatrixItemResult,
    ExamMatrixResponse,
    GeneratedQuestion,
    LLMGeneratedQuestion,
    LLMQuestionGenerationResult,
    MatrixQuestion,
    QuestionGenerationResponse,
)
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_QUESTION_TYPES = ("multiple_choice", "true_false", "short_answer", "essay")

MIN_QUESTION_CHARS = 10
MIN_ANSWER_CHARS = 3
MAX_SOURCE_CHUNKS_PER_QUESTION = 2
NEAR_DUPLICATE_RATIO = 0.9
MAX_LLM_ATTEMPTS = 2  # first try + exactly one retry
SOURCE_RETRIEVAL_CONCURRENCY = 4
MIN_LEXICAL_OVERLAP = 0.2

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


# ── Pure helpers ────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def is_duplicate(candidate: str, seen: list[str]) -> bool:
    """``candidate`` / ``seen`` are already normalised."""
    for other in seen:
        if candidate == other:
            return True
        if difflib.SequenceMatcher(None, candidate, other).ratio() >= NEAR_DUPLICATE_RATIO:
            return True
    return False


def validate_question_item(
    item: LLMGeneratedQuestion, difficulty: str, question_type: str
) -> Optional[dict[str, Any]]:
    """Return cleaned fields for a valid question, or ``None`` to reject it.

    Invalid data is rejected, never silently repaired (the only exception is the
    constant ``["True", "False"]`` option list of true/false questions).
    """
    question = item.question.strip()
    answer = item.expected_answer.strip()
    if len(question) < MIN_QUESTION_CHARS or len(answer) < MIN_ANSWER_CHARS:
        return None
    if item.difficulty != difficulty or item.type != question_type:
        return None

    cleaned: dict[str, Any] = {
        "question": question,
        "expected_answer": answer,
        "options": None,
        "correct_answer": None,
    }

    if question_type == "multiple_choice":
        options = [o.strip() for o in (item.options or []) if isinstance(o, str)]
        correct = (item.correct_answer or "").strip()
        normalized = [normalize_text(o) for o in options]
        if len(options) != 4 or any(not n for n in normalized) or len(set(normalized)) != 4:
            return None
        if not correct or normalize_text(correct) not in normalized:
            return None
        if normalize_text(correct) not in normalize_text(answer):
            return None
        # Keep the canonical option text as the correct answer.
        cleaned["options"] = options
        cleaned["correct_answer"] = options[normalized.index(normalize_text(correct))]
    elif question_type == "true_false":
        correct = normalize_text(item.correct_answer or "")
        if correct not in ("true", "false"):
            return None
        if item.options is not None and [normalize_text(o) for o in item.options] != ["true", "false"]:
            return None
        cleaned["options"] = ["True", "False"]
        cleaned["correct_answer"] = "True" if correct == "true" else "False"
    return cleaned


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def lexical_source_fallback(
    question: str, answer: str, chunks: list[dict[str, Any]]
) -> list[str]:
    """Pick supporting chunk ids from the generation context by word overlap."""
    q_tokens = _tokens(f"{question} {answer}")
    if not q_tokens:
        return []
    scored = []
    for c in chunks:
        overlap = len(q_tokens & _tokens(c["text"])) / len(q_tokens)
        scored.append((overlap, c["chunk_id"]))
    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored or scored[0][0] < MIN_LEXICAL_OVERLAP:
        return []
    return [cid for ov, cid in scored if ov >= MIN_LEXICAL_OVERLAP][:MAX_SOURCE_CHUNKS_PER_QUESTION]


# ── Service ─────────────────────────────────────────────────────────


class QuestionGenerationService:
    """Generates grounded questions on top of RAGService + LLMService."""

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self._rag = rag_service or get_rag_service()
        self._llm = llm_service or get_llm_service()

    # ── Validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_request(
        document_id: Any, number_of_questions: Any, difficulty: Any, question_type: Any
    ) -> str:
        if document_id is None or not str(document_id).strip():
            raise AppError(ErrorCode.EMPTY_DOCUMENT_ID, "document_id must not be empty.")
        if (
            not isinstance(number_of_questions, int)
            or isinstance(number_of_questions, bool)
            or not 1 <= number_of_questions <= 20
        ):
            raise AppError(
                ErrorCode.INVALID_NUMBER_OF_QUESTIONS,
                "number_of_questions must be between 1 and 20.",
            )
        if difficulty not in VALID_DIFFICULTIES:
            raise AppError(
                ErrorCode.INVALID_DIFFICULTY,
                f"difficulty must be one of {list(VALID_DIFFICULTIES)}.",
            )
        if question_type not in VALID_QUESTION_TYPES:
            raise AppError(
                ErrorCode.INVALID_QUESTION_TYPE,
                f"question_type must be one of {list(VALID_QUESTION_TYPES)}.",
            )
        return str(document_id).strip()

    # ── Main entry point ────────────────────────────────────────────

    async def generate_questions(
        self,
        document_id: UUID | str,
        number_of_questions: int = 5,
        difficulty: str = "medium",
        question_type: str = "short_answer",
        request_id: Optional[str] = None,
    ) -> QuestionGenerationResponse:
        request_id = request_id or str(uuid.uuid4())
        doc_id = self._validate_request(document_id, number_of_questions, difficulty, question_type)
        started = time.perf_counter()
        prompt_version = runtime_config.current().prompt_version(prompt_registry.TASK_QUESTION_GENERATION)

        logger.info(
            "Question generation started | request_id=%s document_id=%s prompt_version=%s "
            "question_type=%s difficulty=%s number_of_questions=%d",
            request_id, doc_id, prompt_version,
            question_type, difficulty, number_of_questions,
        )

        try:
            # 1. RAG retrieval (document-scoped; checks the document exists)
            settings = get_settings()
            top_k = min(max(number_of_questions * 2, runtime_config.current().top_k), settings.max_top_k)
            t0 = time.perf_counter()
            retrieval = await self._rag.retrieve(
                doc_id, QUESTION_GENERATION_RETRIEVAL_QUERY, top_k=top_k, request_id=request_id
            )
            rag_ms = (time.perf_counter() - t0) * 1000

            # 2. Insufficient context → never call the LLM
            if retrieval["insufficient_context"] or not retrieval["chunks"]:
                logger.info(
                    "Question generation skipped | request_id=%s document_id=%s "
                    "insufficient_context=True llm_called=False rag_ms=%.1f",
                    request_id, doc_id, rag_ms,
                )
                raise AppError(
                    ErrorCode.INSUFFICIENT_CONTEXT,
                    "The document does not contain enough information to generate questions.",
                )
            context_chunks: list[dict[str, Any]] = retrieval["chunks"]

            # 3. LLM generation + validation (one retry on invalid structure)
            t0 = time.perf_counter()
            valid = await self._generate_valid_questions(
                retrieval["context"], number_of_questions, difficulty, question_type, request_id
            )
            llm_ms = (time.perf_counter() - t0) * 1000

            # 4. Backend-derived source chunks, then backend question ids
            questions = await self._build_questions(
                doc_id, valid, context_chunks, difficulty, question_type, request_id
            )
            if not questions:
                raise AppError(
                    ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE,
                    "The language model did not return usable questions.",
                )
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "Question generation failed | request_id=%s document_id=%s: %s",
                request_id, doc_id, exc, exc_info=True,
            )
            raise AppError(
                ErrorCode.QUESTION_GENERATION_ERROR, "Question generation failed."
            ) from exc

        logger.info(
            "Question generation finished | request_id=%s document_id=%s model=%s "
            "prompt_version=%s retrieved_chunks=%d generated_questions=%d "
            "rag_ms=%.1f llm_ms=%.1f total_ms=%.1f",
            request_id, doc_id, self._llm.model_name, prompt_version,
            len(context_chunks), len(questions), rag_ms, llm_ms,
            (time.perf_counter() - started) * 1000,
        )
        trace_execution(
            prompt_registry.TASK_QUESTION_GENERATION, request_id, doc_id, "ok",
            (time.perf_counter() - started) * 1000,
            source_chunk_ids=[c["chunk_id"] for c in context_chunks], llm_called=True,
        )
        return QuestionGenerationResponse(document_id=UUID(doc_id), questions=questions)

    # ── Exam matrix ─────────────────────────────────────────────────

    async def generate_exam(
        self,
        document_id: UUID | str,
        items: list[ExamMatrixItem],
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ExamMatrixResponse:
        """Generate one batch of questions per matrix row, sequentially (each row is one gated LLM call).

        Questions of different rows are de-duplicated against each other. A row that cannot be satisfied is
        reported (``partial`` / ``failed`` with ``error_code``) instead of being silently padded, so the instructor
        sees exactly what is missing. If the first row fails, the whole request fails with that error (a missing
        document, insufficient context or an unavailable model affects every row). If no row produced a question the
        first error is raised.
        """
        request_id = request_id or str(uuid.uuid4())
        results: list[ExamMatrixItemResult] = []
        questions: list[MatrixQuestion] = []
        seen: list[str] = []
        first_error: Optional[AppError] = None
        doc_id = str(document_id)

        for index, item in enumerate(items):
            try:
                batch = await self.generate_questions(
                    document_id=document_id,
                    number_of_questions=item.count,
                    difficulty=item.difficulty,
                    question_type=item.question_type,
                    request_id=request_id,
                )
            except AppError as exc:
                if index == 0:
                    raise
                first_error = first_error or exc
                results.append(ExamMatrixItemResult(
                    index=index, question_type=item.question_type, difficulty=item.difficulty,
                    requested=item.count, generated=0, status="failed", error_code=str(exc.code),
                ))
                continue
            kept = 0
            for q in batch.questions:
                norm = normalize_text(q.question)
                if is_duplicate(norm, seen):
                    continue
                seen.append(norm)
                questions.append(MatrixQuestion(**q.model_dump(), matrix_item=index))
                kept += 1
            results.append(ExamMatrixItemResult(
                index=index, question_type=item.question_type, difficulty=item.difficulty,
                requested=item.count, generated=kept,
                status="ok" if kept == item.count else ("partial" if kept else "failed"),
                error_code=None if kept else str(ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE),
            ))

        if not questions and first_error is not None:
            raise first_error
        requested = sum(i.count for i in items)
        logger.info(
            "Exam matrix finished | request_id=%s document_id=%s items=%d requested=%d generated=%d",
            request_id, doc_id, len(items), requested, len(questions),
        )
        return ExamMatrixResponse(
            document_id=UUID(doc_id), subject=subject, requested_total=requested,
            generated_total=len(questions), complete=len(questions) == requested and all(r.status == "ok" for r in results),
            items=results, questions=questions,
        )

    # ── LLM step ────────────────────────────────────────────────────

    async def _generate_valid_questions(
        self,
        context: str,
        number_of_questions: int,
        difficulty: str,
        question_type: str,
        request_id: str,
    ) -> list[dict[str, Any]]:
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_QUESTION_GENERATION,
            runtime_config.current().prompt_version(prompt_registry.TASK_QUESTION_GENERATION),
            context, number_of_questions, difficulty, question_type
        )
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            # AppErrors (Ollama down, LLM_INVALID_JSON, empty response...) propagate as-is.
            raw = await self._llm.generate_json(
                system_prompt, user_prompt, required_keys=["questions"]
            )
            valid = self._parse_and_filter(raw, number_of_questions, difficulty, question_type)
            if valid:
                return valid
            logger.warning(
                "Question generation produced no valid questions | request_id=%s attempt=%d/%d",
                request_id, attempt, MAX_LLM_ATTEMPTS,
            )
        raise AppError(
            ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE,
            "The language model did not return valid questions.",
        )

    @staticmethod
    def _parse_and_filter(
        raw: Any, number_of_questions: int, difficulty: str, question_type: str
    ) -> list[dict[str, Any]]:
        if isinstance(raw, list):  # tolerate a bare array
            raw = {"questions": raw}
        if not isinstance(raw, dict):
            return []
        try:
            parsed = LLMQuestionGenerationResult.model_validate(raw)
        except Exception:
            return []

        valid: list[dict[str, Any]] = []
        seen: list[str] = []
        for item in parsed.questions:
            cleaned = validate_question_item(item, difficulty, question_type)
            if cleaned is None:
                continue
            if contains_prompt_leak(cleaned["question"], cleaned["expected_answer"], *(cleaned.get("options") or [])):
                continue  # Phase 19: verbatim system-prompt text in an item drops that item
            norm = normalize_text(cleaned["question"])
            if not norm or is_duplicate(norm, seen):
                continue
            seen.append(norm)
            valid.append(cleaned)
            if len(valid) >= number_of_questions:
                break
        return valid

    # ── Source chunks ───────────────────────────────────────────────

    async def _build_questions(
        self,
        document_id: str,
        valid: list[dict[str, Any]],
        context_chunks: list[dict[str, Any]],
        difficulty: str,
        question_type: str,
        request_id: str,
    ) -> list[GeneratedQuestion]:
        sem = asyncio.Semaphore(SOURCE_RETRIEVAL_CONCURRENCY)

        async def sources_for(q: dict[str, Any]) -> list[str]:
            async with sem:
                try:
                    res = await self._rag.retrieve(
                        document_id,
                        q["question"],
                        top_k=MAX_SOURCE_CHUNKS_PER_QUESTION,
                        request_id=request_id,
                    )
                    ids = [
                        c["chunk_id"]
                        for c in res["chunks"]
                        if c.get("document_id", document_id) == document_id
                    ]
                except Exception as exc:  # fall back to the generation context
                    logger.warning(
                        "Source retrieval failed | request_id=%s: %s", request_id, exc
                    )
                    ids = []
            if not ids:
                ids = lexical_source_fallback(q["question"], q["expected_answer"], context_chunks)
            return ids[:MAX_SOURCE_CHUNKS_PER_QUESTION]

        all_sources = await asyncio.gather(*(sources_for(q) for q in valid))

        questions: list[GeneratedQuestion] = []
        for q, sources in zip(valid, all_sources):
            if not sources:  # no verifiable support in the document → drop
                logger.info("Dropped question without document support | request_id=%s", request_id)
                continue
            questions.append(
                GeneratedQuestion(
                    question_id=str(uuid.uuid4()),
                    question=q["question"],
                    expected_answer=q["expected_answer"],
                    difficulty=difficulty,  # type: ignore[arg-type]
                    type=question_type,  # type: ignore[arg-type]
                    source_chunks=sources,
                    options=q["options"],
                    correct_answer=q["correct_answer"],
                )
            )
        return questions


# ── Module-level singleton ──────────────────────────────────────────

_question_service: Optional[QuestionGenerationService] = None


def get_question_service() -> QuestionGenerationService:
    """Return a lazily-created service reusing the shared RAG and LLM singletons."""
    global _question_service
    if _question_service is None:
        _question_service = QuestionGenerationService()
    return _question_service
