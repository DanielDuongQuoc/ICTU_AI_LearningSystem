"""Phase 4 — RAG retrieval tests.

EmbeddingService and ChromaRepository are mocked: these tests verify the
orchestration logic only.  No Ollama / ChromaDB required.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService, build_context, deduplicate


# ── Helpers ─────────────────────────────────────────────────────────


def hit(
    chunk_id: str,
    score: float,
    document_id: str = "A",
    page: Optional[int] = 1,
    text: Optional[str] = None,
    source: str = "ai.pdf",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text if text is not None else f"text of {chunk_id}",
        "page": page,
        "source": source,
        "score": score,
    }


class FakeRepo:
    """Stand-in for ChromaRepository: stores hits, filters by document_id, honours top_k.

    Deliberately does NOT apply the threshold, so RAGService's own filter is exercised.
    """

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.query_calls: list[dict[str, Any]] = []
        self.exists_calls: list[str] = []

    def document_exists(self, document_id: str) -> bool:
        self.exists_calls.append(document_id)
        return any(h["document_id"] == document_id for h in self.hits)

    def query(self, query_embedding, document_id, top_k=None, similarity_threshold=None):
        self.query_calls.append(
            {"document_id": document_id, "top_k": top_k, "threshold": similarity_threshold}
        )
        own = [h for h in self.hits if h["document_id"] == document_id]
        own.sort(key=lambda h: h["score"], reverse=True)
        return own[:top_k]


def make_embedding() -> MagicMock:
    emb = MagicMock(spec=EmbeddingService)
    emb.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return emb


def make_service(
    hits: list[dict[str, Any]],
    *,
    threshold: float = 0.3,
    max_chars: int = 12000,
    max_top_k: int = 20,
):
    repo = FakeRepo(hits)
    emb = make_embedding()
    svc = RAGService(
        embedding_service=emb,
        chroma_repo=repo,  # type: ignore[arg-type]
        similarity_threshold=threshold,
        max_context_chars=max_chars,
        max_top_k=max_top_k,
    )
    return svc, emb, repo


# ── Test 1: valid query ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_query() -> None:
    svc, emb, repo = make_service([hit("c1", 0.9, page=5), hit("c2", 0.8, page=6)])
    result = await svc.retrieve("A", "What is machine learning?", top_k=5)

    assert result["document_id"] == "A"
    assert result["question"] == "What is machine learning?"
    assert result["insufficient_context"] is False
    assert [c["chunk_id"] for c in result["chunks"]] == ["c1", "c2"]
    assert result["context"]
    emb.embed_text.assert_awaited_once_with("What is machine learning?")
    assert repo.query_calls[0]["document_id"] == "A"


# ── Test 2: empty query ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("q", ["", "   ", "\n\t "])
async def test_empty_query(q: str) -> None:
    svc, emb, repo = make_service([hit("c1", 0.9)])
    with pytest.raises(AppError) as exc:
        await svc.retrieve("A", q)
    assert exc.value.code == ErrorCode.EMPTY_QUERY
    emb.embed_text.assert_not_awaited()
    assert repo.query_calls == []
    assert repo.exists_calls == []


# ── Test 3: document not found ──────────────────────────────────────


@pytest.mark.asyncio
async def test_document_not_found() -> None:
    svc, emb, repo = make_service([hit("c1", 0.9, document_id="A")])
    with pytest.raises(AppError) as exc:
        await svc.retrieve("missing", "question?")
    assert exc.value.code == ErrorCode.DOCUMENT_NOT_FOUND
    emb.embed_text.assert_not_awaited()  # no wasted embedding
    assert repo.query_calls == []  # no wasted Chroma query


# ── Test 4: top-k ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_top_k_limits_results() -> None:
    hits = [hit(f"c{i}", 0.9 - i * 0.05) for i in range(8)]
    svc, _, repo = make_service(hits)
    result = await svc.retrieve("A", "q", top_k=3)
    assert len(result["chunks"]) == 3
    assert repo.query_calls[0]["top_k"] == 3


@pytest.mark.asyncio
async def test_top_k_fewer_available_no_fake_chunks() -> None:
    svc, _, _ = make_service([hit("c1", 0.9), hit("c2", 0.8), hit("c3", 0.7)])
    result = await svc.retrieve("A", "q", top_k=5)
    assert len(result["chunks"]) == 3


@pytest.mark.asyncio
async def test_top_k_default_from_settings() -> None:
    hits = [hit(f"c{i}", 0.95 - i * 0.01) for i in range(10)]
    svc, _, repo = make_service(hits)
    result = await svc.retrieve("A", "q")
    assert repo.query_calls[0]["top_k"] == 5
    assert len(result["chunks"]) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1, 21, 1000])
async def test_invalid_top_k(bad: int) -> None:
    svc, emb, repo = make_service([hit("c1", 0.9)])
    with pytest.raises(AppError) as exc:
        await svc.retrieve("A", "q", top_k=bad)
    assert exc.value.code == ErrorCode.INVALID_TOP_K
    emb.embed_text.assert_not_awaited()
    assert repo.query_calls == []


# ── Test 5 / IMPORTANT: similarity threshold ────────────────────────


@pytest.mark.asyncio
async def test_similarity_threshold() -> None:
    hits = [hit("a", 0.95), hit("b", 0.82), hit("c", 0.29), hit("d", 0.10)]
    svc, _, _ = make_service(hits, threshold=0.30)
    result = await svc.retrieve("A", "q")
    assert [c["score"] for c in result["chunks"]] == [0.95, 0.82]
    ids = {c["chunk_id"] for c in result["chunks"]}
    assert ids == {"a", "b"}
    assert "chunk: c]" not in result["context"]
    assert "chunk: d]" not in result["context"]


@pytest.mark.asyncio
async def test_threshold_boundary_inclusive() -> None:
    svc, _, _ = make_service([hit("edge", 0.30)], threshold=0.30)
    result = await svc.retrieve("A", "q")
    assert result["insufficient_context"] is False


# ── Test 6: ranking ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ranking_best_first() -> None:
    svc, _, repo = make_service([])
    # Repo returns an unsorted list; service must sort by score desc.
    repo_hits = [hit("low", 0.5), hit("high", 0.95), hit("mid", 0.7)]
    with patch.object(repo, "query", return_value=repo_hits), patch.object(
        repo, "document_exists", return_value=True
    ):
        result = await svc.retrieve("A", "q")
    assert [c["chunk_id"] for c in result["chunks"]] == ["high", "mid", "low"]
    assert [s["chunk_id"] for s in result["sources"]] == ["high", "mid", "low"]
    # Context follows relevance order, not page order.
    assert result["context"].index("chunk: high") < result["context"].index("chunk: low")


# ── Test 7: duplicate removal ───────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_chunks_removed() -> None:
    svc, _, repo = make_service([])
    dup = [hit("c1", 0.9), hit("c1", 0.9), hit("c2", 0.8), hit("c1", 0.85)]
    with patch.object(repo, "query", return_value=dup), patch.object(
        repo, "document_exists", return_value=True
    ):
        result = await svc.retrieve("A", "q")
    assert [c["chunk_id"] for c in result["chunks"]] == ["c1", "c2"]
    assert result["context"].count("chunk: c1]") == 1


def test_deduplicate_keeps_highest_score() -> None:
    out = deduplicate([hit("x", 0.5), hit("x", 0.9)])
    assert len(out) == 1 and out[0]["score"] == 0.9


# ── Test 8 / IMPORTANT: document isolation ──────────────────────────


@pytest.mark.asyncio
async def test_document_isolation() -> None:
    hits = [
        hit("chunk_a1", 0.95, document_id="A"),
        hit("chunk_a2", 0.80, document_id="A"),
        hit("chunk_b1", 0.99, document_id="B"),
    ]
    svc, _, _ = make_service(hits)
    result = await svc.retrieve("A", "q")
    ids = [c["chunk_id"] for c in result["chunks"]]
    assert ids == ["chunk_a1", "chunk_a2"]
    assert "chunk_b1" not in result["context"]
    assert all(s["document_id"] == "A" for s in result["sources"])


@pytest.mark.asyncio
async def test_document_isolation_defends_against_leaky_repo() -> None:
    """Even if the repository wrongly returned another document's chunk."""
    svc, _, repo = make_service([])
    leaky = [hit("chunk_b1", 0.99, document_id="B"), hit("chunk_a1", 0.9, document_id="A")]
    with patch.object(repo, "query", return_value=leaky), patch.object(
        repo, "document_exists", return_value=True
    ):
        result = await svc.retrieve("A", "q")
    assert [c["chunk_id"] for c in result["chunks"]] == ["chunk_a1"]


# ── Test 9: context construction ────────────────────────────────────


@pytest.mark.asyncio
async def test_context_construction_format() -> None:
    svc, _, _ = make_service(
        [
            hit("c1", 0.92, page=5, text="Machine learning is a branch of AI."),
            hit("c2", 0.81, page=6, text="Systems learn patterns from data."),
        ]
    )
    result = await svc.retrieve("A", "q")
    expected = (
        "[Source: page 5 | chunk: c1]\nMachine learning is a branch of AI."
        "\n\n"
        "[Source: page 6 | chunk: c2]\nSystems learn patterns from data."
    )
    assert result["context"] == expected


# ── Test 10 / IMPORTANT: context max size ───────────────────────────


@pytest.mark.asyncio
async def test_context_max_size() -> None:
    hits = [hit(f"c{i}", 0.95 - i * 0.05, text="x" * 400, page=i + 1) for i in range(5)]
    svc, _, _ = make_service(hits, max_chars=1000)
    result = await svc.retrieve("A", "q")

    assert len(result["context"]) <= 1000
    ids = [c["chunk_id"] for c in result["chunks"]]
    # Highest scores kept, in order; lower ones dropped.
    assert ids == ["c0", "c1"]
    assert len(set(ids)) == len(ids)
    # Metadata still correct.
    assert result["sources"][0] == {
        "chunk_id": "c0", "document_id": "A", "page": 1, "source": "ai.pdf", "score": 0.95,
    }
    # Chunks were not cut mid-way.
    assert all(len(c["text"]) == 400 for c in result["chunks"])


def test_build_context_truncates_only_when_first_chunk_too_big() -> None:
    chunk = hit("big", 0.9, text="y" * 5000)
    context, included = build_context([chunk], max_chars=500)
    assert 0 < len(context) <= 500
    assert len(included) == 1 and len(included[0]["text"]) < 5000


def test_build_context_does_not_skip_to_lower_ranked() -> None:
    ranked = [hit("big", 0.9, text="a" * 600), hit("small", 0.5, text="b")]
    context, included = build_context(ranked, max_chars=300)
    # First chunk truncated; lower-scored small chunk must not displace/join it.
    assert [c["chunk_id"] for c in included] == ["big"]


# ── Test 11 / IMPORTANT: insufficient context ───────────────────────


@pytest.mark.asyncio
async def test_insufficient_context() -> None:
    svc, _, _ = make_service([hit("a", 0.20), hit("b", 0.19), hit("c", 0.15)], threshold=0.30)
    result = await svc.retrieve("A", "q")
    assert result["insufficient_context"] is True
    assert result["chunks"] == []
    assert result["context"] == ""
    assert result["sources"] == []


# ── Test 12: source metadata preservation ───────────────────────────


@pytest.mark.asyncio
async def test_source_metadata_preserved() -> None:
    svc, _, _ = make_service([hit("c1", 0.91, page=5, source="ai.pdf")])
    result = await svc.retrieve("A", "q")
    assert result["sources"] == [
        {"chunk_id": "c1", "document_id": "A", "page": 5, "source": "ai.pdf", "score": 0.91}
    ]
    chunk = result["chunks"][0]
    assert chunk["page"] == 5 and chunk["source"] == "ai.pdf" and chunk["score"] == 0.91
    assert chunk["text"] == "text of c1"


# ── Test 13 / IMPORTANT: no LLM ─────────────────────────────────────


@pytest.mark.asyncio
async def test_no_llm_called() -> None:
    emb = MagicMock(spec=EmbeddingService)
    emb.embed_text = AsyncMock(return_value=[0.1])
    repo = MagicMock(spec=ChromaRepository)
    repo.document_exists.return_value = True
    repo.query.return_value = [hit("c1", 0.9)]

    svc = RAGService(
        embedding_service=emb,
        chroma_repo=repo,
        similarity_threshold=0.3,
        max_context_chars=1000,
        max_top_k=20,
    )
    await svc.retrieve("A", "q")

    # Only the embedding model (embed_text) and the repository are touched.
    assert {c[0] for c in emb.method_calls} <= {"embed_text"}
    emb.embed_text.assert_awaited_once()
    assert {c[0] for c in repo.method_calls} <= {"document_exists", "query"}


def test_rag_service_has_no_llm_dependency() -> None:
    import app.services.rag_service as mod

    src = open(mod.__file__, encoding="utf-8").read().lower()
    for forbidden in ("ollama_model", "/api/generate", "/api/chat", "qwen", "openai", "gemini"):
        assert forbidden not in src.replace("no generative llm", ""), forbidden


# ── Error wrapping ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embedding_unexpected_error_wrapped() -> None:
    svc, emb, _ = make_service([hit("c1", 0.9)])
    emb.embed_text = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(AppError) as exc:
        await svc.retrieve("A", "q")
    assert exc.value.code == ErrorCode.EMBEDDING_ERROR


@pytest.mark.asyncio
async def test_embedding_app_error_propagates() -> None:
    svc, emb, _ = make_service([hit("c1", 0.9)])
    emb.embed_text = AsyncMock(
        side_effect=AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "down")
    )
    with pytest.raises(AppError) as exc:
        await svc.retrieve("A", "q")
    assert exc.value.code == ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_chroma_error_wrapped() -> None:
    svc, _, repo = make_service([hit("c1", 0.9)])
    with patch.object(repo, "query", side_effect=RuntimeError("db down")):
        with pytest.raises(AppError) as exc:
            await svc.retrieve("A", "q")
    assert exc.value.code == ErrorCode.CHROMA_ERROR


# ── API endpoint ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_retrieve_ok(client) -> None:
    svc, _, _ = make_service([hit("c1", 0.9, page=5)])
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post(
            "/api/rag/retrieve",
            json={"document_id": "A", "question": "What is ML?", "top_k": 5},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_context"] is False
    assert body["sources"][0]["page"] == 5
    assert body["chunks"][0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_api_empty_question(client) -> None:
    svc, _, _ = make_service([hit("c1", 0.9)])
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post(
            "/api/rag/retrieve", json={"document_id": "A", "question": "   "}
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == ErrorCode.EMPTY_QUERY


@pytest.mark.asyncio
async def test_api_document_not_found(client) -> None:
    svc, _, _ = make_service([hit("c1", 0.9)])
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post(
            "/api/rag/retrieve", json={"document_id": "nope", "question": "hi"}
        )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == ErrorCode.DOCUMENT_NOT_FOUND


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — RAG + LLM (answer_question).  LLM is always mocked.
# ═══════════════════════════════════════════════════════════════════

from app.core.prompts import INSUFFICIENT_CONTEXT_ANSWER, RAG_SYSTEM_PROMPT  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402


def make_llm(response: Any = None, side_effect: Any = None) -> MagicMock:
    llm = MagicMock(spec=LLMService)
    llm.generate_json = AsyncMock(
        return_value=response if response is not None else {"answer": "ML learns from data."},
        side_effect=side_effect,
    )
    llm.generate = AsyncMock(return_value="should not be used")
    return llm


def make_rag(hits, llm, **kw):
    repo = FakeRepo(hits)
    emb = make_embedding()
    svc = RAGService(
        embedding_service=emb,
        chroma_repo=repo,  # type: ignore[arg-type]
        llm_service=llm,
        similarity_threshold=kw.get("threshold", 0.3),
        max_context_chars=kw.get("max_chars", 12000),
        max_top_k=20,
    )
    return svc, emb, repo


@pytest.mark.asyncio
async def test_answer_relevant_context_returns_llm_answer() -> None:
    llm = make_llm({"answer": "Machine learning allows systems to learn from data."})
    svc, _, _ = make_rag([hit("c1", 0.91, page=5, text="ML lets systems learn from data.")], llm)
    result = await svc.answer_question("A", "What is machine learning?")

    assert result["answer"] == "Machine learning allows systems to learn from data."
    assert result["insufficient_context"] is False
    assert result["document_id"] == "A"
    assert result["question"] == "What is machine learning?"
    llm.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_passed_to_llm_has_context_question_and_rules() -> None:
    llm = make_llm()
    svc, _, _ = make_rag([hit("c1", 0.9, page=5, text='ML "learns"\nfrom {data}.')], llm)
    await svc.answer_question("A", "What is ML?")

    args, kwargs = llm.generate_json.await_args
    system_prompt, user_prompt = args
    assert system_prompt == RAG_SYSTEM_PROMPT
    assert 'ML "learns"\nfrom {data}.' in user_prompt  # context preserved verbatim
    assert "[Source: page 5 | chunk: c1]" in user_prompt
    assert "What is ML?" in user_prompt
    assert kwargs["required_keys"] == ["answer"]


# ── Mandatory: no LLM call on insufficient context ──────────────────


@pytest.mark.asyncio
async def test_insufficient_context_does_not_call_llm() -> None:
    llm = make_llm()
    svc, _, _ = make_rag([hit("a", 0.2), hit("b", 0.1)], llm, threshold=0.3)
    result = await svc.answer_question("A", "unrelated?")

    assert result == {
        "document_id": "A",
        "question": "unrelated?",
        "answer": INSUFFICIENT_CONTEXT_ANSWER,
        "insufficient_context": True,
        "sources": [],
    }
    llm.generate.assert_not_called()
    llm.generate_json.assert_not_called()
    assert llm.method_calls == []


# ── Source integrity ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sources_come_from_retrieval_and_llm_cannot_alter_them() -> None:
    llm = make_llm(
        {
            "answer": "An answer.",
            "page": 999,
            "chunk_id": "FAKE",
            "score": 1.0,
            "sources": [{"page": 999, "chunk_id": "FAKE", "score": 1.0}],
        }
    )
    svc, _, _ = make_rag([hit("c1", 0.91, page=5, source="ai.pdf")], llm)
    result = await svc.answer_question("A", "q")

    assert result["answer"] == "An answer."
    assert result["sources"] == [
        {"chunk_id": "c1", "document_id": "A", "page": 5, "source": "ai.pdf", "score": 0.91}
    ]
    assert "FAKE" not in str(result) and 999 not in [s["page"] for s in result["sources"]]
    assert set(result) == {"document_id", "question", "answer", "insufficient_context", "sources"}


@pytest.mark.asyncio
async def test_answer_document_isolation_remains() -> None:
    llm = make_llm()
    hits = [
        hit("chunk_a1", 0.95, document_id="A"),
        hit("chunk_b1", 0.99, document_id="B", text="SECRET B CONTENT"),
    ]
    svc, _, _ = make_rag(hits, llm)
    result = await svc.answer_question("A", "q")

    assert [s["chunk_id"] for s in result["sources"]] == ["chunk_a1"]
    user_prompt = llm.generate_json.await_args.args[1]
    assert "SECRET B CONTENT" not in user_prompt


# ── Error handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_answer_rag_errors_propagate_without_llm() -> None:
    llm = make_llm()
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)

    with pytest.raises(AppError) as exc:
        await svc.answer_question("A", "   ")
    assert exc.value.code == ErrorCode.EMPTY_QUERY

    with pytest.raises(AppError) as exc:
        await svc.answer_question("missing", "q")
    assert exc.value.code == ErrorCode.DOCUMENT_NOT_FOUND
    llm.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_answer_llm_app_error_propagates() -> None:
    llm = make_llm(side_effect=AppError(ErrorCode.OLLAMA_UNAVAILABLE, "down"))
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)
    with pytest.raises(AppError) as exc:
        await svc.answer_question("A", "q")
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE


@pytest.mark.asyncio
async def test_answer_llm_unexpected_error_wrapped() -> None:
    llm = make_llm(side_effect=RuntimeError("boom internals"))
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)
    with pytest.raises(AppError) as exc:
        await svc.answer_question("A", "q")
    assert exc.value.code == ErrorCode.OLLAMA_GENERATION_ERROR
    assert "boom" not in exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [{"answer": ""}, {"answer": "   "}, {"answer": 123}])
async def test_answer_empty_or_wrong_type_rejected(bad: dict) -> None:
    llm = make_llm(bad)
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)
    with pytest.raises(AppError) as exc:
        await svc.answer_question("A", "q")
    assert exc.value.code == ErrorCode.LLM_EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_retrieve_alone_still_never_calls_llm() -> None:
    llm = make_llm()
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)
    await svc.retrieve("A", "q")
    assert llm.method_calls == []


# ── API ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_answer_ok(client) -> None:
    llm = make_llm({"answer": "From the doc."})
    svc, _, _ = make_rag([hit("c1", 0.9, page=5)], llm)
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post(
            "/api/rag/answer", json={"document_id": "A", "question": "What is ML?"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "From the doc."
    assert body["insufficient_context"] is False
    assert body["sources"][0]["page"] == 5 and body["sources"][0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_api_answer_insufficient(client) -> None:
    llm = make_llm()
    svc, _, _ = make_rag([hit("c1", 0.1)], llm)
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post("/api/rag/answer", json={"document_id": "A", "question": "q"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_context"] is True and body["sources"] == []
    assert body["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    llm.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_api_answer_llm_down_returns_clean_error(client) -> None:
    llm = make_llm(side_effect=AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable."))
    svc, _, _ = make_rag([hit("c1", 0.9)], llm)
    with patch("app.api.rag._get_rag_service", return_value=svc):
        resp = await client.post("/api/rag/answer", json={"document_id": "A", "question": "q"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == ErrorCode.OLLAMA_UNAVAILABLE
    assert "Traceback" not in str(body)
