"""Phase 6 — Question generation tests.

EmbeddingService, ChromaRepository (FakeRepo) and LLMService are mocked, so the
backend logic is tested without Ollama or ChromaDB.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.core.prompts import (
    QUESTION_GENERATION_PROMPT_VERSION,
    QUESTION_GENERATION_SYSTEM_PROMPT,
    build_question_generation_prompt,
)
from app.schemas.question import QuestionGenerateRequest
from app.services.llm_service import LLMService
from app.services.question_service import (
    QuestionGenerationService,
    is_duplicate,
    normalize_text,
)
from app.services.rag_service import RAGService
from tests.test_rag import FakeRepo, hit, make_embedding

DOC_A = str(uuid.UUID(int=1))
DOC_B = str(uuid.UUID(int=2))
UNKNOWN = str(uuid.UUID(int=99))


# ── Helpers ─────────────────────────────────────────────────────────


def q(question: str, answer: str = "An expected answer.", difficulty="medium",
      type_="short_answer", **extra: Any) -> dict[str, Any]:
    return {"question": question, "expected_answer": answer,
            "difficulty": difficulty, "type": type_, **extra}


def make_llm(response: Any = None, side_effect: Any = None) -> MagicMock:
    llm = MagicMock(spec=LLMService)
    llm.model_name = "qwen-test"
    llm.generate_json = AsyncMock(
        return_value=response if response is not None else {"questions": [q("What is machine learning?")]},
        side_effect=side_effect,
    )
    llm.generate = AsyncMock(return_value="unused")
    return llm


def make_stack(hits, llm, threshold: float = 0.3):
    repo = FakeRepo(hits)
    emb = make_embedding()
    rag = RAGService(
        embedding_service=emb, chroma_repo=repo,  # type: ignore[arg-type]
        similarity_threshold=threshold, max_context_chars=12000, max_top_k=20,
    )
    return QuestionGenerationService(rag_service=rag, llm_service=llm), repo, emb


def default_hits():
    return [
        hit("chunk_001", 0.9, document_id=DOC_A, text="Machine learning lets systems learn from data."),
        hit("chunk_002", 0.8, document_id=DOC_A, text="Neural networks are inspired by the brain."),
        hit("chunk_b1", 0.95, document_id=DOC_B, text="SECRET B CONTENT about databases."),
    ]


async def gen(svc, doc=DOC_A, n=5, difficulty="medium", qtype="short_answer"):
    return await svc.generate_questions(doc, n, difficulty, qtype)


# ── Happy paths ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_short_answer_questions() -> None:
    llm = make_llm()
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    assert str(res.document_id) == DOC_A
    assert len(res.questions) == 1
    item = res.questions[0]
    assert item.question == "What is machine learning?"
    assert item.type == "short_answer" and item.difficulty == "medium"
    llm.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_multiple_questions() -> None:
    llm = make_llm({"questions": [
        q("What is machine learning?"),
        q("How are neural networks inspired?"),
        q("Which systems learn from data?"),
    ]})
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc, n=3)
    assert len(res.questions) == 3
    assert len({x.question_id for x in res.questions}) == 3


@pytest.mark.asyncio
async def test_llm_returns_more_than_requested_is_capped() -> None:
    llm = make_llm({"questions": [
        q("What is machine learning?"),
        q("How are neural networks inspired by biology?"),
        q("Which approach lets systems improve from experience?"),
        q("Describe the role of training data in models."),
    ]})
    svc, _, _ = make_stack(default_hits(), llm)
    assert len((await gen(svc, n=2)).questions) == 2


@pytest.mark.asyncio
async def test_llm_returns_fewer_than_requested_not_padded() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    assert len((await gen(svc, n=5)).questions) == 1


# ── Request validation ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [0, -1, 21])
async def test_invalid_number_of_questions(n: int) -> None:
    llm = make_llm()
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc, n=n)
    assert exc.value.code == ErrorCode.INVALID_NUMBER_OF_QUESTIONS
    llm.generate_json.assert_not_called()
    with pytest.raises(Exception):
        QuestionGenerateRequest(document_id=DOC_A, number_of_questions=n)


@pytest.mark.asyncio
async def test_invalid_difficulty() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    with pytest.raises(AppError) as exc:
        await gen(svc, difficulty="extreme")
    assert exc.value.code == ErrorCode.INVALID_DIFFICULTY
    with pytest.raises(Exception):
        QuestionGenerateRequest(document_id=DOC_A, difficulty="extreme")


@pytest.mark.asyncio
async def test_invalid_question_type() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    with pytest.raises(AppError) as exc:
        await gen(svc, qtype="poem")
    assert exc.value.code == ErrorCode.INVALID_QUESTION_TYPE
    with pytest.raises(Exception):
        QuestionGenerateRequest(document_id=DOC_A, question_type="poem")


@pytest.mark.asyncio
async def test_empty_document_id() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    with pytest.raises(AppError) as exc:
        await gen(svc, doc="  ")
    assert exc.value.code == ErrorCode.EMPTY_DOCUMENT_ID


@pytest.mark.asyncio
async def test_document_not_found() -> None:
    llm = make_llm()
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc, doc=UNKNOWN)
    assert exc.value.code == ErrorCode.DOCUMENT_NOT_FOUND
    llm.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_insufficient_context_does_not_call_llm() -> None:
    llm = make_llm()
    hits = [hit("c1", 0.1, document_id=DOC_A), hit("c2", 0.05, document_id=DOC_A)]
    svc, _, _ = make_stack(hits, llm, threshold=0.3)
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.INSUFFICIENT_CONTEXT
    assert llm.method_calls == []


# ── LLM JSON handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_valid_json() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    res = await gen(svc)
    assert res.questions[0].expected_answer == "An expected answer."


@pytest.mark.asyncio
async def test_llm_invalid_json() -> None:
    llm = make_llm(side_effect=AppError(ErrorCode.LLM_INVALID_JSON, "bad json"))
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.LLM_INVALID_JSON


@pytest.mark.asyncio
async def test_llm_invalid_structure_raises_after_single_retry() -> None:
    llm = make_llm(side_effect=[{"questions": "nope"}, {"wrong": 1}])
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE
    assert llm.generate_json.await_count == 2  # never more than one retry


@pytest.mark.asyncio
async def test_llm_retry_once() -> None:
    llm = make_llm(side_effect=[{"questions": []}, {"questions": [q("What is machine learning?")]}])
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    assert len(res.questions) == 1
    assert llm.generate_json.await_count == 2


@pytest.mark.asyncio
async def test_llm_unexpected_error_wrapped_without_details() -> None:
    llm = make_llm(side_effect=RuntimeError("secret internals"))
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.QUESTION_GENERATION_ERROR
    assert "secret" not in exc.value.message


@pytest.mark.asyncio
async def test_ollama_errors_propagate() -> None:
    llm = make_llm(side_effect=AppError(ErrorCode.OLLAMA_UNAVAILABLE, "down"))
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE


# ── Duplicates ──────────────────────────────────────────────────────


def test_normalize_and_duplicate_helpers() -> None:
    assert normalize_text("  What  is Machine-Learning?? ") == "what is machine learning"
    assert is_duplicate(normalize_text("what is machine learning"),
                        [normalize_text("What is machine learning?")])
    assert not is_duplicate("what is python", ["how do neural networks work"])


@pytest.mark.asyncio
async def test_duplicate_questions_removed() -> None:
    llm = make_llm({"questions": [
        q("What is machine learning?"),
        q("what is machine learning"),
        q("  WHAT IS   machine learning?!  "),
        q("How are neural networks inspired?"),
    ]})
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    assert [x.question for x in res.questions] == [
        "What is machine learning?", "How are neural networks inspired?"]


# ── IDs and sources ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_question_id_generated_by_backend() -> None:
    llm = make_llm({"questions": [q("What is machine learning?", question_id="abc123")]})
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    qid = res.questions[0].question_id
    assert qid != "abc123"
    uuid.UUID(qid)  # backend-generated UUID


@pytest.mark.asyncio
async def test_source_chunks_are_backend_derived() -> None:
    llm = make_llm({"questions": [q("What is machine learning?", source_chunks=["fake_chunk"])]})
    svc, repo, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    sources = res.questions[0].source_chunks
    assert "fake_chunk" not in sources
    assert sources and set(sources) <= {"chunk_001", "chunk_002"}
    # One retrieval for generation context + one per question (by question text).
    assert len(repo.query_calls) == 2


@pytest.mark.asyncio
async def test_source_chunks_belong_to_document() -> None:
    llm = make_llm({"questions": [q("What is machine learning?"), q("How are neural networks inspired?")]})
    svc, repo, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    for item in res.questions:
        assert item.source_chunks
        assert all(c.startswith("chunk_00") for c in item.source_chunks)
        assert "chunk_b1" not in item.source_chunks
    assert all(c["document_id"] == DOC_A for c in repo.query_calls)


@pytest.mark.asyncio
async def test_no_external_source_chunks() -> None:
    llm = make_llm({"questions": [q("What is machine learning?",
                                     source_chunks=["chunk_b1", "ghost"])]})
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc)
    assert "chunk_b1" not in res.questions[0].source_chunks
    assert "ghost" not in res.questions[0].source_chunks


@pytest.mark.asyncio
async def test_question_without_document_support_is_dropped() -> None:
    """Retrieval by question returns nothing and there is no lexical support."""
    svc, _, _ = make_stack(default_hits(), make_llm())
    calls = {"n": 0}
    real = svc._rag.retrieve

    async def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return await real(*a, **kw)
        raise AppError(ErrorCode.RAG_RETRIEVAL_ERROR, "boom")

    svc._rag.retrieve = flaky  # type: ignore[method-assign]
    llm_q = {"questions": [q("Zebra quantum xylophone velocity?", "Zebra quantum xylophone")]}
    svc._llm.generate_json = AsyncMock(return_value=llm_q)  # type: ignore[attr-defined]
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE


# ── Integrity ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_question_type_integrity() -> None:
    llm = make_llm({"questions": [q("What is machine learning?", type_="essay")]})
    svc, _, _ = make_stack(default_hits(), llm)
    with pytest.raises(AppError) as exc:
        await gen(svc, qtype="short_answer")
    assert exc.value.code == ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE


@pytest.mark.asyncio
async def test_difficulty_integrity() -> None:
    llm = make_llm({"questions": [
        q("What is machine learning?", difficulty="hard"),
        q("How are neural networks inspired?", difficulty="easy"),
    ]})
    svc, _, _ = make_stack(default_hits(), llm)
    res = await gen(svc, difficulty="easy")
    assert [x.question for x in res.questions] == ["How are neural networks inspired?"]
    assert all(x.difficulty == "easy" for x in res.questions)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    q("What is machine learning?", answer="   "),
    q("Too short", answer="Some answer"),
    q("What is machine learning?", answer="x"),
])
async def test_expected_answer_not_empty(bad: dict) -> None:
    svc, _, _ = make_stack(default_hits(), make_llm({"questions": [bad]}))
    with pytest.raises(AppError) as exc:
        await gen(svc)
    assert exc.value.code == ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE


# ── Multiple choice / true-false / essay ────────────────────────────


def mc(**over: Any) -> dict[str, Any]:
    base = q("Which field lets systems learn from data?", "Machine learning, because it learns.",
             type_="multiple_choice",
             options=["Machine learning", "Databases", "Networking", "Compilers"],
             correct_answer="Machine learning")
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_multiple_choice_validation() -> None:
    svc, _, _ = make_stack(default_hits(), make_llm({"questions": [mc()]}))
    res = await gen(svc, qtype="multiple_choice")
    item = res.questions[0]
    assert item.options and len(item.options) == 4
    assert item.correct_answer == "Machine learning"

    bad_cases = [
        mc(options=["A one", "B two", "C three"]),                       # 3 options
        mc(options=["Same", "same", "Other", "Another"]),                # duplicates
        mc(correct_answer="Quantum"),                                    # not in options
        mc(expected_answer="Databases are the answer here."),            # inconsistent
        mc(options=None),
    ]
    for bad in bad_cases:
        svc, _, _ = make_stack(default_hits(), make_llm({"questions": [bad]}))
        with pytest.raises(AppError) as exc:
            await gen(svc, qtype="multiple_choice")
        assert exc.value.code == ErrorCode.QUESTION_GENERATION_INVALID_RESPONSE


@pytest.mark.asyncio
async def test_true_false_validation() -> None:
    good = q("Machine learning lets systems learn from data.", "True, per the document.",
             difficulty="easy", type_="true_false", options=["True", "False"], correct_answer="true")
    svc, _, _ = make_stack(default_hits(), make_llm({"questions": [good]}))
    res = await gen(svc, difficulty="easy", qtype="true_false")
    assert res.questions[0].options == ["True", "False"]
    assert res.questions[0].correct_answer == "True"

    for bad in (
        {**good, "correct_answer": "maybe"},
        {**good, "options": ["Yes", "No"]},
    ):
        svc, _, _ = make_stack(default_hits(), make_llm({"questions": [bad]}))
        with pytest.raises(AppError):
            await gen(svc, difficulty="easy", qtype="true_false")


@pytest.mark.asyncio
async def test_essay_question() -> None:
    essay = q("Explain how machine learning uses data.", "Key points: learns from data.",
              difficulty="hard", type_="essay")
    svc, _, _ = make_stack(default_hits(), make_llm({"questions": [essay]}))
    res = await gen(svc, difficulty="hard", qtype="essay")
    assert res.questions[0].type == "essay" and res.questions[0].options is None


# ── Prompt ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("qtype", ["short_answer", "multiple_choice", "true_false", "essay"])
def test_question_generation_prompt(qtype: str) -> None:
    system, user = build_question_generation_prompt("CTX {with} braces", 7, "hard", qtype)
    assert system == QUESTION_GENERATION_SYSTEM_PROMPT
    assert "CONTEXT" in user and "CTX {with} braces" in user
    assert "7" in user and "hard" in user and qtype in user
    assert "SOURCE OF TRUTH = CONTEXT" in system
    assert "external knowledge" in system
    assert "valid JSON" in system and "Markdown" in system
    assert "question_id" in system and "source_chunks" in system
    assert QUESTION_GENERATION_PROMPT_VERSION == "v1"


def test_prompt_values_are_dynamic() -> None:
    _, u1 = build_question_generation_prompt("c", 3, "easy", "short_answer")
    _, u2 = build_question_generation_prompt("c", 9, "hard", "essay")
    assert u1 != u2 and "hard" not in u1 and "easy" not in u2


# ── RAG + LLM integration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_to_llm_integration_isolation_and_context() -> None:
    hits = default_hits() + [
        hit("chunk_low", 0.1, document_id=DOC_A, text="IRRELEVANT BELOW THRESHOLD"),
    ]
    llm = make_llm({"questions": [q("What is machine learning?"), q("How are neural networks inspired?")]})
    svc, repo, _ = make_stack(hits, llm)
    res = await gen(svc, n=2)

    assert str(res.document_id) == DOC_A
    assert 1 <= len(res.questions) <= 2
    system_prompt, user_prompt = llm.generate_json.await_args.args
    assert "Machine learning lets systems learn from data." in user_prompt
    assert "SECRET B CONTENT" not in user_prompt
    assert "IRRELEVANT BELOW THRESHOLD" not in user_prompt  # not the whole document
    all_sources = {c for item in res.questions for c in item.source_chunks}
    assert all_sources <= {"chunk_001", "chunk_002"}


@pytest.mark.asyncio
async def test_hallucination_python_example() -> None:
    hits = [hit("chunk_py", 0.9, document_id=DOC_A,
                text="Python is a programming language created by Guido van Rossum.")]
    llm = make_llm({"questions": [
        q("Who created Python?", "Guido van Rossum", source_chunks=["fake_chunk"])]})
    svc, _, _ = make_stack(hits, llm)
    res = await gen(svc, n=1)
    item = res.questions[0]
    assert item.expected_answer == "Guido van Rossum"
    assert item.source_chunks == ["chunk_py"]
    assert "fake_chunk" not in item.source_chunks


# ── API ─────────────────────────────────────────────────────────────

URL = "/api/questions/generate"


@pytest.mark.asyncio
async def test_api_generate_ok(client) -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    with patch("app.api.questions._get_question_service", return_value=svc):
        resp = await client.post(URL, json={"document_id": DOC_A, "number_of_questions": 3,
                                            "difficulty": "medium", "question_type": "short_answer"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == DOC_A
    item = body["questions"][0]
    assert set(item) == {"question_id", "question", "expected_answer", "difficulty",
                         "type", "source_chunks"}
    assert item["source_chunks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,code", [
    ({"document_id": DOC_A, "number_of_questions": 0}, "INVALID_NUMBER_OF_QUESTIONS"),
    ({"document_id": DOC_A, "number_of_questions": 21}, "INVALID_NUMBER_OF_QUESTIONS"),
    ({"document_id": DOC_A, "difficulty": "insane"}, "INVALID_DIFFICULTY"),
    ({"document_id": DOC_A, "question_type": "poem"}, "INVALID_QUESTION_TYPE"),
    ({"document_id": ""}, "EMPTY_DOCUMENT_ID"),
    ({}, "EMPTY_DOCUMENT_ID"),
    ({"document_id": "not-a-uuid"}, "INVALID_DOCUMENT_ID"),
])
async def test_api_validation_errors(client, payload: dict, code: str) -> None:
    resp = await client.post(URL, json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == code
    assert "request_id" in resp.json()["error"]


@pytest.mark.asyncio
async def test_api_insufficient_context(client) -> None:
    llm = make_llm()
    svc, _, _ = make_stack([hit("c1", 0.1, document_id=DOC_A)], llm)
    with patch("app.api.questions._get_question_service", return_value=svc):
        resp = await client.post(URL, json={"document_id": DOC_A})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INSUFFICIENT_CONTEXT"
    llm.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_api_document_not_found(client) -> None:
    svc, _, _ = make_stack(default_hits(), make_llm())
    with patch("app.api.questions._get_question_service", return_value=svc):
        resp = await client.post(URL, json={"document_id": UNKNOWN})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_api_ollama_down_clean_error(client) -> None:
    llm = make_llm(side_effect=AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable."))
    svc, _, _ = make_stack(default_hits(), llm)
    with patch("app.api.questions._get_question_service", return_value=svc):
        resp = await client.post(URL, json={"document_id": DOC_A})
    assert resp.status_code == 503
    assert "Traceback" not in str(resp.json())
