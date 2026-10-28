"""Phase 13: contract-level integration tests (CONTRACT E2E - Module 1 / Module 2 are not in this repo).

Module 1 and Module 2 sources/APIs were not available, so nothing here calls them. The tests drive
Module 3's own (frozen) HTTP contract exactly as a Module 1 uploader / Module 2 consumer would,
using a real in-memory ChromaDB with fake embedding + recording LLM. Fixtures live in tests/fixtures.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import chromadb
import httpx
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.core.exceptions import AppError, ErrorCode
from app.repositories.chroma_repository import ChromaRepository
from tests.fakes import Stack, sample_pdf_bytes

FIX = Path(__file__).parent / "fixtures"
TEXT_A = "Zebracorn models learn patterns from data. Zebracorn systems explain machine learning."
TEXT_B = "Quokkaplex indexes speed up database queries. Quokkaplex lookups use database data."


def fixture(name: str, **fill) -> dict:
    data = json.loads((FIX / name).read_text(encoding="utf-8"))
    data.pop("_meta", None)
    data.update(fill)
    return data


async def upload(client, text, name: str = "lecture.pdf") -> dict:
    pages = text if isinstance(text, list) else [text]
    r = await client.post("/api/documents/upload", files={"file": (name, sample_pdf_bytes(pages), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()


def sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


@pytest.fixture()
def stack(monkeypatch):
    return Stack().install(monkeypatch)


# ------------------------------------------------------------ fixtures describe the contract

def test_fixtures_are_valid_json_and_flagged():
    for p in FIX.glob("*.json"):
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "_meta" in raw, p.name
    assert "TBD" in json.loads((FIX / "module1_document.json").read_text(encoding="utf-8"))["_meta"]


# ------------------------------------------------------------ Module 1 -> Module 3 (document id)

@pytest.mark.asyncio
async def test_document_id_is_kept_end_to_end(client, stack):
    up = await upload(client, TEXT_A)
    doc = up["document_id"]
    assert (await client.get(f"/api/documents/{doc}")).json()["document_id"] == doc
    assert up["pages"] >= 1 and up["chunks"] >= 1

    r = (await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})).json()
    assert r["document_id"] == doc and {c["document_id"] for c in r["chunks"]} == {doc}
    assert all(s["page"] is None or s["page"] >= 1 for s in r["sources"])

    q = (await client.post("/api/questions/generate", json=fixture("module2_question_request.json", document_id=doc))).json()
    assert q["document_id"] == doc

    ev = (await client.post("/api/evaluations/evaluate",
                            json=fixture("module2_student_answer.json", document_id=doc))).json()
    chunk_ids = {c["chunk_id"] for c in r["chunks"]}
    assert ev["sources"] and {s["chunk_id"] for s in ev["sources"]} <= chunk_ids


@pytest.mark.asyncio
async def test_two_documents_never_mix_across_all_flows(client, stack):
    a, b = (await upload(client, TEXT_A))["document_id"], (await upload(client, TEXT_B))["document_id"]
    assert a != b
    for doc, own, other in ((a, "zebracorn", "quokkaplex"), (b, "quokkaplex", "zebracorn")):
        stack.llm.calls.clear()
        r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": f"{own} data"})
        assert other not in json.dumps(r.json()).lower()
        await client.post("/api/rag/answer", json={"document_id": doc, "question": f"{own} data"})
        await client.post("/api/questions/generate", json={"document_id": doc, "number_of_questions": 1})
        body = fixture("module2_student_answer.json", document_id=doc, question=f"What about {own} data?",
                       student_answer="Patterns from examples.", expected_answer="It learns patterns.")
        await client.post("/api/evaluations/evaluate", json=body)
        await client.post("/api/evaluations/evaluate/stream", json=body)
        assert other not in stack.llm.prompts.lower()


# ------------------------------------------------------------ Module 2 -> Module 3 -> Module 2

@pytest.mark.asyncio
async def test_evaluation_response_matches_contract_fixture(client, stack):
    doc = (await upload(client, TEXT_A))["document_id"]
    r = await client.post("/api/evaluations/evaluate", json=fixture("module2_student_answer.json", document_id=doc))
    assert r.status_code == 200 and "application/json" in r.headers["content-type"]
    body, spec = r.json(), fixture("module3_evaluation_response.json")
    assert set(spec["required"]) <= set(body)
    for src in body["sources"]:
        assert set(spec["source_required"]) <= set(src)
    assert 0 <= body["score"] <= body["max_score"] == 10
    assert body["percentage"] == pytest.approx(body["score"] / body["max_score"] * 100)


@pytest.mark.asyncio
async def test_client_cannot_supply_percentage_or_sources(client, stack):
    doc = (await upload(client, TEXT_A))["document_id"]
    body = fixture("module2_student_answer.json", document_id=doc)
    body.update(percentage=100, score=10, sources=[{"chunk_id": "forged", "page": 999, "score": 1}])
    r = await client.post("/api/evaluations/evaluate", json=body)
    if r.status_code == 200:
        out = r.json()
        assert out["percentage"] == pytest.approx(out["score"] / out["max_score"] * 100)
        assert out["score"] == 8 and "forged" not in r.text
    else:
        assert r.json()["error"]["code"]


@pytest.mark.parametrize("max_score", [0, -5])
@pytest.mark.asyncio
async def test_non_positive_max_score_rejected(client, stack, max_score):
    doc = (await upload(client, TEXT_A))["document_id"]
    r = await client.post("/api/evaluations/evaluate",
                          json=fixture("module2_student_answer.json", document_id=doc, max_score=max_score))
    assert r.status_code in (400, 422) and r.json()["error"]["code"]


@pytest.mark.asyncio
async def test_paraphrase_is_not_exact_string_compared(client, stack):
    """The backend never short-circuits on string equality: the LLM is always consulted."""
    doc = (await upload(client, TEXT_A))["document_id"]
    body = fixture("module2_student_answer.json", document_id=doc)
    assert body["student_answer"] != body["expected_answer"]
    stack.llm.calls.clear()
    assert (await client.post("/api/evaluations/evaluate", json=body)).status_code == 200
    assert len(stack.llm.calls) == 1
    stack.llm.calls.clear()
    same = {**body, "student_answer": body["expected_answer"]}
    assert (await client.post("/api/evaluations/evaluate", json=same)).status_code == 200
    assert len(stack.llm.calls) == 1  # identical text still goes through evaluation


@pytest.mark.asyncio
async def test_out_of_context_question_is_insufficient_and_skips_llm(client, monkeypatch):
    s = Stack(threshold=0.5).install(monkeypatch)
    doc = (await upload(client, TEXT_A))["document_id"]
    body = fixture("module2_student_answer.json", document_id=doc,
                   question="Describe volcanic eruptions in Iceland", expected_answer="Magma erupts.")
    r = await client.post("/api/evaluations/evaluate", json=body)
    assert r.json()["error"]["code"] == "INSUFFICIENT_CONTEXT"
    assert s.llm.calls == []


# ------------------------------------------------------------ error propagation

@pytest.mark.parametrize("code,status", [
    ("OLLAMA_UNAVAILABLE", 503), ("OLLAMA_TIMEOUT", 504), ("LLM_INVALID_JSON", 502),
    ("OLLAMA_MODEL_NOT_FOUND", 503),
])
@pytest.mark.asyncio
async def test_llm_failures_propagate_cleanly(client, stack, code, status):
    doc = (await upload(client, TEXT_A))["document_id"]

    async def boom(*a, **k):
        raise AppError(code=getattr(ErrorCode, code), message="dependency failed")

    stack.llm.generate_json = boom
    for path, body in (("/api/evaluations/evaluate", fixture("module2_student_answer.json", document_id=doc)),
                       ("/api/rag/answer", {"document_id": doc, "question": "zebracorn data"}),
                       ("/api/questions/generate", fixture("module2_question_request.json", document_id=doc))):
        r = await client.post(path, json=body, headers={"X-Request-ID": "cross-module-123"})
        assert r.status_code == status, (path, r.text)
        err = r.json()["error"]
        assert err["code"] == code and err["request_id"] == "cross-module-123" == r.headers["X-Request-ID"]
        assert "Traceback" not in r.text and "site-packages" not in r.text


@pytest.mark.asyncio
async def test_embedding_and_chroma_failures_propagate(client, stack):
    doc = (await upload(client, TEXT_A))["document_id"]

    async def bad_embed(*a, **k):
        raise AppError(code=ErrorCode.EMBEDDING_ERROR, message="embedding failed")

    stack.embedding.embed_text = bad_embed
    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "EMBEDDING_ERROR"

    def bad_query(*a, **k):
        raise AppError(code=ErrorCode.CHROMA_ERROR, message="chroma failed")

    stack.embedding.embed_text = type(stack.embedding).embed_text.__get__(stack.embedding)
    stack.repo.query = bad_query
    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "CHROMA_ERROR"


@pytest.mark.asyncio
async def test_stream_failure_is_error_then_done_with_safe_payload(client, stack):
    doc = (await upload(client, TEXT_A))["document_id"]

    async def failing_stream(*a, **k):
        raise AppError(code=ErrorCode.OLLAMA_UNAVAILABLE, message="down")
        yield ""  # pragma: no cover

    stack.llm.generate_stream = failing_stream
    r = await client.post("/api/evaluations/evaluate/stream",
                          json=fixture("module2_student_answer.json", document_id=doc))
    events = sse(r.text)
    types = [e["type"] for e in events]
    assert "error" in types and "result" not in types
    assert types[-1] == "done" and types.count("done") == 1
    assert "Traceback" not in r.text


# ------------------------------------------------------------ encoding

TEXTS = ["Nguy\u1ec5n V\u0103n A h\u1ecdc tr\u00ed tu\u1ec7 nh\u00e2n t\u1ea1o.", "\u4eba\u5de5\u667a\u80fd mixed with English", "caf\u00e9 \u00fc\u00f1\u00ee \U0001f600"]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.asyncio
async def test_unicode_round_trip(client, stack, text):
    doc = (await upload(client, TEXT_A))["document_id"]
    stack.llm.eval_payload = {"score": 7, "correct": True, "feedback": f"Ph\u1ea3n h\u1ed3i: {text}",
                              "strengths": [text], "missing_points": [], "errors": []}
    body = fixture("module2_student_answer.json", document_id=doc, student_answer=text, question=f"zebracorn {text}")
    r = await client.post("/api/evaluations/evaluate", json=body)
    assert r.status_code == 200
    assert text in r.json()["feedback"] and text in r.json()["strengths"]
    assert text in stack.llm.calls[-1][2]  # reached the model un-corrupted
    r = await client.post("/api/evaluations/evaluate/stream", json=body)
    result = next(e for e in sse(r.text) if e["type"] == "result")["result"]
    assert text in result["feedback"]


# ------------------------------------------------------------ large input / concurrency / idempotency

@pytest.mark.asyncio
async def test_long_inputs_are_handled_without_500(client, stack):
    long_text = [" ".join(["Zebracorn models learn patterns from data."] * 40)] * 40
    up = await upload(client, long_text)
    assert up["chunks"] > 10
    doc = up["document_id"]
    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})
    assert r.status_code == 200 and len(r.json()["context"]) <= 12000
    r = await client.post("/api/evaluations/evaluate",
                          json=fixture("module2_student_answer.json", document_id=doc, student_answer="learn " * 20000))
    assert r.status_code != 500 and ("error" in r.json() or "score" in r.json())


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_cross_or_corrupt(client, stack):
    a, b = (await upload(client, TEXT_A))["document_id"], (await upload(client, TEXT_B))["document_id"]
    jobs = []
    for i in range(10):
        doc, own = (a, "zebracorn") if i % 2 == 0 else (b, "quokkaplex")
        jobs.append(client.post("/api/rag/retrieve", json={"document_id": doc, "question": f"{own} data"}))
        jobs.append(client.post("/api/evaluations/evaluate",
                                json=fixture("module2_student_answer.json", document_id=doc,
                                             question=f"What about {own} data?")))
        if i < 4:
            jobs.append(client.post("/api/evaluations/evaluate/stream",
                                    json=fixture("module2_student_answer.json", document_id=doc,
                                                 question=f"What about {own} data?")))
    results = await asyncio.gather(*jobs)
    assert all(r.status_code == 200 for r in results)
    for r in results:
        if r.headers["content-type"].startswith("text/event-stream"):
            types = [e["type"] for e in sse(r.text)]
            assert types.count("result") == 1 and types[-1] == "done"
        elif "chunks" in r.json():
            body = r.json()
            assert {c["document_id"] for c in body["chunks"]} == {body["document_id"]}
        else:
            assert 0 <= r.json()["score"] <= 10
    ids = [r.headers["X-Request-ID"] for r in results]
    assert len(set(ids)) == len(ids)


@pytest.mark.asyncio
async def test_reupload_and_evaluation_are_stateless(client, stack):
    first, second = await upload(client, TEXT_A), await upload(client, TEXT_A)
    assert first["document_id"] != second["document_id"] and first["chunks"] == second["chunks"]
    doc = first["document_id"]
    before = stack.repo.count()
    for _ in range(3):
        assert (await client.post("/api/evaluations/evaluate",
                                  json=fixture("module2_student_answer.json", document_id=doc))).status_code == 200
    assert stack.repo.count() == before  # evaluation stores nothing
    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    assert (await client.get(f"/api/documents/{second['document_id']}")).status_code == 200


# ------------------------------------------------------------ lifecycle

@pytest.mark.asyncio
async def test_chroma_data_survives_restart(client, monkeypatch, tmp_path):
    path = str(tmp_path / "chroma")

    def persistent(name: str) -> ChromaRepository:
        return ChromaRepository(client=chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)), collection_name="persist")

    monkeypatch.setattr("tests.fakes.make_repo", persistent)
    Stack().install(monkeypatch)
    up = await upload(client, TEXT_A)
    reopened = persistent("persist")  # simulates a process restart on the same path
    assert reopened.count_by_document(up["document_id"]) == up["chunks"]


# ------------------------------------------------------------ CORS

def _app_with_origins(monkeypatch, origins: str):
    import app.main as main

    settings = Settings(cors_allowed_origins=origins)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return main.create_app()


@pytest.mark.asyncio
async def test_cors_disabled_by_default(monkeypatch):
    application = _app_with_origins(monkeypatch, "")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        r = await c.get("/health/live", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.asyncio
async def test_cors_only_configured_origins(monkeypatch):
    application = _app_with_origins(monkeypatch, "http://app.example, http://other.example")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        ok = await c.options("/api/rag/retrieve", headers={
            "Origin": "http://app.example", "Access-Control-Request-Method": "POST"})
        bad = await c.options("/api/rag/retrieve", headers={
            "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        live = await c.get("/health/live", headers={"Origin": "http://app.example"})
    assert ok.headers["access-control-allow-origin"] == "http://app.example"
    assert "access-control-allow-origin" not in bad.headers
    assert "x-request-id" in live.headers.get("access-control-expose-headers", "").lower()


def test_cors_origins_parsing():
    assert Settings(cors_allowed_origins=" a , ,b ").cors_origins == ["a", "b"]
    assert Settings().cors_origins == []
