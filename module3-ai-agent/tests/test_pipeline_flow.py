"""Phase 10 - API-level integration flow with offline dependencies.

Real: FastAPI app, PDF extraction, chunking, ChromaDB (in-memory), RAG, question/evaluation services,
SSE.  Fake: only the embedding model (lexical) and the LLM (scripted, schema-valid output).

PDF -> upload -> get -> retrieve -> answer -> questions -> evaluate -> stream -> delete
"""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio

from app.services.evaluation_service import EvaluationService
from app.services.question_service import QuestionGenerationService
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from tests.fakes import LexicalEmbedding, ScriptedLLM, make_repo, sample_pdf_bytes

ML_PAGES = [
    "Machine learning is a field of artificial intelligence in which computers learn patterns "
    "from data instead of following hand-written rules. A trained model can make predictions "
    "about new, unseen examples. Supervised learning uses labeled examples.",
]
DB_PAGES = [
    "A relational database stores data in tables made of rows and columns. SQL is the standard "
    "language used to query relational data. A primary key uniquely identifies each row.",
]


@pytest_asyncio.fixture()
async def flow(client, monkeypatch):
    embedding, llm = LexicalEmbedding(), ScriptedLLM()
    repo = make_repo(f"flow_{uuid.uuid4().hex[:8]}")
    vector = VectorService(embedding_service=embedding, chroma_repo=repo)
    rag = RAGService(embedding_service=embedding, chroma_repo=repo, llm_service=llm,
                     similarity_threshold=0.0, max_context_chars=12000, max_top_k=20)
    questions = QuestionGenerationService(rag_service=rag, llm_service=llm)
    evaluator = EvaluationService(rag_service=rag, llm_service=llm)
    monkeypatch.setattr("app.api.documents._get_vector_service", lambda: vector)
    monkeypatch.setattr("app.api.rag._get_rag_service", lambda: rag)
    monkeypatch.setattr("app.api.questions._get_question_service", lambda: questions)
    monkeypatch.setattr("app.api.evaluations._get_evaluation_service", lambda: evaluator)
    return client, llm


async def _upload(client, pages, name):
    resp = await client.post(
        "/api/documents/upload",
        files={"file": (name, sample_pdf_bytes(pages), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return resp


def _parse_sse(text: str) -> list[dict]:
    return [json.loads(b[len("data: "):]) for b in text.split("\n\n") if b]


def _assert_request_id(resp) -> None:
    rid = resp.headers["X-Request-ID"]
    assert str(uuid.UUID(rid)) == rid


@pytest.mark.asyncio
async def test_full_pipeline_flow(flow):
    client, llm = flow

    # upload two documents
    up_a = await _upload(client, ML_PAGES, "ml.pdf")
    up_b = await _upload(client, DB_PAGES, "db.pdf")
    _assert_request_id(up_a)
    doc_a, doc_b = up_a.json()["document_id"], up_b.json()["document_id"]
    assert up_a.json()["status"] == "indexed" and up_a.json()["chunks"] >= 1

    # document info
    info = await client.get(f"/api/documents/{doc_a}")
    assert info.status_code == 200 and info.json()["chunks"] >= 1

    # RAG retrieval: only the requested document's chunks (isolation)
    retr = await client.post("/api/rag/retrieve",
                             json={"document_id": doc_a, "question": "What is machine learning?"})
    assert retr.status_code == 200
    body = retr.json()
    assert not body["insufficient_context"] and body["chunks"]
    assert {c["document_id"] for c in body["chunks"]} == {doc_a}
    assert "relational" not in body["context"].lower()
    chunk_ids_a = {c["chunk_id"] for c in body["chunks"]}
    assert {s["chunk_id"] for s in body["sources"]} <= chunk_ids_a
    _assert_request_id(retr)

    # same question against document B never returns A's content
    retr_b = await client.post("/api/rag/retrieve",
                               json={"document_id": doc_b, "question": "What is machine learning?"})
    assert {c["document_id"] for c in retr_b.json()["chunks"]} <= {doc_b}
    assert "machine learning is a field" not in retr_b.json()["context"].lower()

    # RAG answer
    ans = await client.post("/api/rag/answer",
                            json={"document_id": doc_a, "question": "What is machine learning?"})
    assert ans.status_code == 200 and ans.json()["answer"]
    assert {s["document_id"] for s in ans.json()["sources"]} == {doc_a}

    # question generation: ids and sources are backend-derived
    qs = await client.post("/api/questions/generate", json={"document_id": doc_a, "number_of_questions": 2})
    assert qs.status_code == 200, qs.text
    questions = qs.json()["questions"]
    assert questions
    for q in questions:
        uuid.UUID(q["question_id"])
        assert q["source_chunks"] and set(q["source_chunks"]) <= {
            c["chunk_id"] for c in (await client.post(
                "/api/rag/retrieve", json={"document_id": doc_a, "question": q["question"]})).json()["chunks"]
        } | chunk_ids_a

    # evaluation
    payload = {"document_id": doc_a, "question": "What is machine learning?",
               "student_answer": "Computers learn patterns from data.",
               "expected_answer": "Computers learn patterns from data to make predictions.", "max_score": 10}
    ev = await client.post("/api/evaluations/evaluate", json=payload)
    assert ev.status_code == 200, ev.text
    result = ev.json()
    assert result["score"] == 8 and result["percentage"] == 80
    assert result["sources"] and {s["chunk_id"] for s in result["sources"]} <= chunk_ids_a
    _assert_request_id(ev)

    # streaming evaluation: status -> token... -> result -> done
    st = await client.post("/api/evaluations/evaluate/stream", json=payload)
    assert st.status_code == 200 and st.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(st.text)
    types = [e["type"] for e in events]
    assert types[0] == "status" and types[-1] == "done"
    assert types.count("done") == 1 and types.count("result") == 1
    assert types.index("result") < types.index("done")
    first_token = types.index("token")
    assert types.index("result") > first_token
    assert all(t in ("status", "token") for t in types[:types.index("result")])
    assert events[types.index("result")]["result"]["score"] == 8
    assert llm.stream_calls == 1

    # delete A: gone, B untouched
    deleted = await client.delete(f"/api/documents/{doc_a}")
    assert deleted.status_code == 200
    gone = await client.get(f"/api/documents/{doc_a}")
    assert gone.status_code == 404
    err = gone.json()["error"]
    assert err["code"] == "DOCUMENT_NOT_FOUND" and err["request_id"] == gone.headers["X-Request-ID"]
    assert (await client.get(f"/api/documents/{doc_b}")).status_code == 200


@pytest.mark.asyncio
async def test_unknown_document_errors_use_standard_format(flow):
    client, _ = flow
    unknown = str(uuid.uuid4())
    for path, body in (
        ("/api/rag/retrieve", {"document_id": unknown, "question": "anything"}),
        ("/api/questions/generate", {"document_id": unknown, "number_of_questions": 1}),
        ("/api/evaluations/evaluate", {"document_id": unknown, "question": "q?", "student_answer": "a",
                                       "expected_answer": "b", "max_score": 10}),
    ):
        resp = await client.post(path, json=body)
        assert resp.status_code == 404, path
        err = resp.json()["error"]
        assert err["code"] == "DOCUMENT_NOT_FOUND" and err["request_id"] == resp.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_stream_for_unknown_document_ends_with_error_then_done(flow):
    client, _ = flow
    st = await client.post("/api/evaluations/evaluate/stream", json={
        "document_id": str(uuid.uuid4()), "question": "q?", "student_answer": "a",
        "expected_answer": "b", "max_score": 10})
    types = [e["type"] for e in _parse_sse(st.text)]
    assert "result" not in types and types[-2:] == ["error", "done"] and types.count("done") == 1


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_with_standard_error(flow):
    client, _ = flow
    resp = await client.post("/api/documents/upload",
                             files={"file": ("evil.pdf", b"not a pdf at all", "application/pdf")})
    assert resp.status_code >= 400
    err = resp.json()["error"]
    assert set(err) == {"code", "message", "request_id"}
    assert "Traceback" not in resp.text


@pytest.mark.asyncio
async def test_upload_path_traversal_filename_is_sanitised(flow):
    client, _ = flow
    resp = await client.post(
        "/api/documents/upload",
        files={"file": ("../../etc/passwd.pdf", sample_pdf_bytes(ML_PAGES), "application/pdf")},
    )
    assert resp.status_code == 200
    name = resp.json()["filename"]
    assert "/" not in name and "\\" not in name and ".." not in name
