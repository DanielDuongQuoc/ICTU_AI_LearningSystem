"""Phase 8 — Streaming (SSE) answer evaluation tests.

Ollama is never called: the LLM stream is a mocked async generator, and the Ollama
NDJSON parsing is tested through ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import AppError, ErrorCode
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import OllamaLLMService
from app.services.rag_service import RAGService
from app.utils.sse import done_event, error_event, format_sse, status_event, token_event
from tests.test_evaluation import (
    DOC_A,
    DOC_B,
    EXPECTED,
    QUESTION,
    STUDENT,
    UNKNOWN,
    default_hits,
    llm_result,
    make_llm,
    make_stack,
)
from tests.test_rag import hit

URL = "/api/evaluations/evaluate/stream"
ALLOWED = {"status", "token", "result", "error", "done"}


# ── Helpers ─────────────────────────────────────────────────────────


def chunked(text: str, size: int = 5) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def set_stream(llm: MagicMock, *outputs: Any) -> None:
    """Each call of generate_stream consumes the next output.

    An output is a list of text pieces, or an Exception raised after the pieces yielded so far.
    """
    queue = list(outputs)

    def factory(*_a: Any, **_k: Any) -> AsyncIterator[str]:
        item = queue.pop(0) if len(queue) > 1 else queue[0]

        async def gen() -> AsyncIterator[str]:
            pieces, exc = item if isinstance(item, tuple) else (item, None)
            for p in pieces:
                yield p
            if exc is not None:
                raise exc

        return gen()

    llm.generate_stream = MagicMock(side_effect=factory)


def good_stream(**over: Any) -> list[str]:
    return chunked(json.dumps(llm_result(**over)))


def payload(**over: Any) -> dict[str, Any]:
    base = {"document_id": DOC_A, "question": QUESTION, "student_answer": STUDENT,
            "expected_answer": EXPECTED, "max_score": 10}
    base.update(over)
    return base


def parse_sse(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: "), block
        events.append(json.loads(block[len("data: "):]))
    return events


def assert_contract(events: list[dict[str, Any]]) -> None:
    assert events, "no events"
    assert all(e["type"] in ALLOWED for e in events)
    assert events[-1] == {"type": "done"}
    assert sum(e["type"] == "done" for e in events) == 1
    assert sum(e["type"] == "result" for e in events) <= 1
    for e in events:
        if e["type"] == "token":
            assert set(e) == {"type", "content"} and isinstance(e["content"], str)
        if e["type"] == "status":
            assert set(e) == {"type", "message"} and isinstance(e["message"], str)
        if e["type"] == "error":
            assert set(e) == {"type", "code", "message"}
    if any(e["type"] == "error" for e in events):
        assert not any(e["type"] == "result" for e in events)
    if any(e["type"] == "result" for e in events):
        assert [e["type"] for e in events].index("result") < len(events) - 1


async def run(client, llm_stream_setup=None, hits=None, llm=None, **over: Any):
    llm = llm or make_llm()
    if llm_stream_setup is not None:
        set_stream(llm, *llm_stream_setup)
    svc, _ = make_stack(hits if hits is not None else default_hits(), llm)
    with patch("app.api.evaluations._get_evaluation_service", return_value=svc):
        resp = await client.post(URL, json=payload(**over), headers={"Accept": "text/event-stream"})
    return resp, llm, svc


def errors_of(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == "error"]


# ── Happy path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_returns_sse(client) -> None:
    resp, _, _ = await run(client, [good_stream()])
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["connection"] == "keep-alive"


@pytest.mark.asyncio
async def test_stream_status_events(client) -> None:
    resp, _, _ = await run(client, [good_stream()])
    messages = [e["message"] for e in parse_sse(resp.text) if e["type"] == "status"]
    assert messages[:3] == ["Validating request", "Retrieving relevant context", "Evaluating answer"]


@pytest.mark.asyncio
async def test_stream_token_events(client) -> None:
    pieces = good_stream()
    resp, _, _ = await run(client, [pieces])
    tokens = [e["content"] for e in parse_sse(resp.text) if e["type"] == "token"]
    assert tokens == pieces  # forwarded unchanged, in order


@pytest.mark.asyncio
async def test_stream_final_result(client) -> None:
    resp, _, _ = await run(client, [good_stream(score=8)])
    events = parse_sse(resp.text)
    result = [e for e in events if e["type"] == "result"]
    assert len(result) == 1
    r = result[0]["result"]
    assert r["score"] == 8 and r["max_score"] == 10 and r["percentage"] == 80
    assert r["correct"] is True and r["feedback"]
    assert set(r) == {"score", "max_score", "percentage", "correct", "feedback",
                      "strengths", "missing_points", "errors", "sources"}
    assert r["sources"][0] == {"page": 5, "chunk_id": "chunk_A1", "score": 0.87}


@pytest.mark.asyncio
async def test_stream_done_event(client) -> None:
    resp, _, _ = await run(client, [good_stream()])
    events = parse_sse(resp.text)
    assert events[-1] == {"type": "done"}
    assert [e["type"] for e in events].index("result") == len(events) - 2


@pytest.mark.asyncio
async def test_stream_event_format(client) -> None:
    resp, _, _ = await run(client, [good_stream()])
    assert resp.text.endswith("\n\n")
    assert "```" not in resp.text and "'type'" not in resp.text
    for block in resp.text.split("\n\n")[:-1]:
        assert block.startswith("data: ") and "\n" not in block
    assert_contract(parse_sse(resp.text))
    # no raw Ollama objects leak to the client
    assert "created_at" not in resp.text and '"model"' not in resp.text


@pytest.mark.asyncio
async def test_stream_json_is_valid(client) -> None:
    resp, _, _ = await run(client, [good_stream()])
    for block in resp.text.strip().split("\n\n"):
        obj = json.loads(block.removeprefix("data: "))
        assert isinstance(obj, dict) and obj["type"] in ALLOWED


@pytest.mark.asyncio
async def test_stream_unicode_not_escaped(client) -> None:
    resp, _, _ = await run(client, [good_stream(feedback="Câu trả lời đúng.")])
    assert "Câu trả lời đúng." in resp.text


@pytest.mark.asyncio
async def test_stream_markdown_fenced_output_is_parsed(client) -> None:
    fenced = ["```json\n", json.dumps(llm_result()), "\n```"]
    resp, _, _ = await run(client, [fenced])
    events = parse_sse(resp.text)
    assert_contract(events)
    assert sum(e["type"] == "result" for e in events) == 1


@pytest.mark.asyncio
async def test_phase7_endpoint_still_works(client) -> None:
    llm = make_llm()
    svc, _ = make_stack(default_hits(), llm)
    with patch("app.api.evaluations._get_evaluation_service", return_value=svc):
        resp = await client.post("/api/evaluations/evaluate", json=payload())
    assert resp.status_code == 200 and resp.json()["percentage"] == 80
    llm.generate_json.assert_awaited_once()
    assert "application/json" in resp.headers["content-type"]


# ── Early errors: LLM never called ──────────────────────────────────


@pytest.mark.asyncio
async def test_stream_insufficient_context_does_not_call_llm(client) -> None:
    llm = make_llm()
    set_stream(llm, good_stream())
    resp, _, _ = await run(client, hits=[hit("c1", 0.1, document_id=DOC_A)], llm=llm)
    events = parse_sse(resp.text)
    assert_contract(events)
    assert [e["type"] for e in events][-3:] == ["status", "error", "done"]
    assert events[-3]["message"] == "Insufficient context"
    assert events[-2] == {
        "type": "error", "code": "INSUFFICIENT_CONTEXT",
        "message": "The document does not contain enough information to evaluate this answer."}
    llm.generate_stream.assert_not_called()
    llm.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_stream_document_not_found(client) -> None:
    llm = make_llm()
    set_stream(llm, good_stream())
    resp, _, _ = await run(client, llm=llm, document_id=UNKNOWN)
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "DOCUMENT_NOT_FOUND"
    llm.generate_stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("student", ["", "   \n"])
async def test_stream_empty_student_answer(client, student: str) -> None:
    llm = make_llm()
    set_stream(llm, good_stream())
    resp, _, svc = await run(client, llm=llm, student_answer=student)
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "EMPTY_STUDENT_ANSWER"
    assert [e["type"] for e in events] == ["status", "error", "done"]
    llm.generate_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_empty_expected_answer_and_bad_inputs(client) -> None:
    for over, code in [
        ({"expected_answer": ""}, "EMPTY_EXPECTED_ANSWER"),
        ({"question": " "}, "EMPTY_QUESTION"),
        ({"max_score": 0}, "INVALID_SCORE"),
        ({"document_id": ""}, "EMPTY_DOCUMENT_ID"),
        ({"document_id": "not-a-uuid"}, "INVALID_DOCUMENT_ID"),
    ]:
        llm = make_llm()
        set_stream(llm, good_stream())
        resp, _, _ = await run(client, llm=llm, **over)
        events = parse_sse(resp.text)
        assert_contract(events)
        assert errors_of(events)[0]["code"] == code
        llm.generate_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_empty_body_yields_error_event(client) -> None:
    svc, _ = make_stack(default_hits(), make_llm())
    with patch("app.api.evaluations._get_evaluation_service", return_value=svc):
        resp = await client.post(URL, json={})
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "EMPTY_DOCUMENT_ID"


# ── LLM / Ollama failures ───────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [
    ErrorCode.OLLAMA_TIMEOUT, ErrorCode.OLLAMA_UNAVAILABLE,
    ErrorCode.OLLAMA_MODEL_NOT_FOUND, ErrorCode.LLM_EMPTY_RESPONSE,
])
async def test_stream_ollama_errors(client, code: str) -> None:
    resp, _, _ = await run(client, [([], AppError(code, "safe message"))])
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events) == [{"type": "error", "code": code, "message": "safe message"}]
    assert events[-2]["type"] == "error"


@pytest.mark.asyncio
async def test_stream_ollama_timeout(client) -> None:
    resp, _, _ = await run(client, [(["{\"sco"], AppError(ErrorCode.OLLAMA_TIMEOUT, "timed out"))])
    events = parse_sse(resp.text)
    assert_contract(events)
    assert [e["type"] for e in events if e["type"] in ("token", "error", "done")] == [
        "token", "error", "done"]
    assert errors_of(events)[0]["code"] == "OLLAMA_TIMEOUT"


@pytest.mark.asyncio
async def test_stream_ollama_unavailable(client) -> None:
    resp, _, _ = await run(client, [([], AppError(ErrorCode.OLLAMA_UNAVAILABLE, "not reachable"))])
    assert errors_of(parse_sse(resp.text))[0]["code"] == "OLLAMA_UNAVAILABLE"


@pytest.mark.asyncio
async def test_stream_unexpected_exception_is_hidden(client, caplog) -> None:
    resp, _, _ = await run(client, [([], RuntimeError("secret internals"))])
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events) == [{"type": "error", "code": "EVALUATION_ERROR",
                                  "message": "Answer evaluation failed."}]
    assert "secret" not in resp.text and "Traceback" not in resp.text


# ── Final JSON validation & retry ───────────────────────────────────


@pytest.mark.asyncio
async def test_stream_invalid_final_json(client) -> None:
    llm = make_llm(llm_result(score=8))
    llm.generate_json = AsyncMock(side_effect=AppError(ErrorCode.LLM_INVALID_JSON, "bad json"))
    resp, _, _ = await run(client, [["not json at all"]], llm=llm)
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "LLM_INVALID_JSON"
    llm.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_invalid_structure_after_retry_is_error(client) -> None:
    llm = make_llm({"nope": 1})
    resp, _, _ = await run(client, [chunked('{"score": "high"}')], llm=llm)
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "EVALUATION_INVALID_RESPONSE"
    assert llm.generate_json.await_count == 1  # exactly one retry
    assert llm.generate_stream.call_count == 1


@pytest.mark.asyncio
async def test_stream_json_retry_once(client) -> None:
    llm = make_llm(llm_result(score=7))
    resp, _, _ = await run(client, [["{ broken json"]], llm=llm)
    events = parse_sse(resp.text)
    assert_contract(events)
    messages = [e["message"] for e in events if e["type"] == "status"]
    assert "Retrying structured evaluation" in messages
    result = [e for e in events if e["type"] == "result"][0]["result"]
    assert result["score"] == 7 and result["percentage"] == 70
    llm.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_invalid_score(client) -> None:
    llm = make_llm(llm_result(score=15))
    resp, _, _ = await run(client, [good_stream(score=15)], llm=llm)
    events = parse_sse(resp.text)
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "INVALID_SCORE"
    assert not any(e["type"] == "result" for e in events)
    assert llm.generate_json.await_count == 1

    resp, _, _ = await run(client, [good_stream(score=-1)], llm=make_llm(llm_result(score=-1)))
    assert errors_of(parse_sse(resp.text))[0]["code"] == "INVALID_SCORE"


# ── Source integrity / backend-owned fields ─────────────────────────


@pytest.mark.asyncio
async def test_stream_source_integrity() -> None:
    llm = make_llm()
    set_stream(llm, good_stream())
    rag = MagicMock(spec=RAGService)
    rag.retrieve = AsyncMock(return_value={
        "document_id": DOC_A, "question": "q", "insufficient_context": False, "context": "ctx",
        "chunks": [{"chunk_id": "chunk_B1", "document_id": DOC_B, "page": 1,
                    "source": "b.pdf", "score": 0.9, "text": "x"}],
        "sources": [],
    })
    svc = EvaluationService(rag_service=rag, llm_service=llm)
    frames = [f async for f in svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)]
    events = parse_sse("".join(frames))
    assert_contract(events)
    assert errors_of(events)[0]["code"] == "SOURCE_INTEGRITY_ERROR"
    assert "chunk_B1" not in "".join(frames)
    llm.generate_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_fake_llm_sources_ignored(client) -> None:
    fake = [{"page": 999, "chunk_id": "fake", "score": 1.0}]
    resp, _, _ = await run(client, [good_stream(sources=fake)])
    result = [e for e in parse_sse(resp.text) if e["type"] == "result"][0]["result"]
    assert [s["chunk_id"] for s in result["sources"]] == ["chunk_A1", "chunk_A2"]
    assert "fake" not in json.dumps(result["sources"])


@pytest.mark.asyncio
async def test_stream_percentage_backend_calculated(client) -> None:
    resp, _, _ = await run(client, [good_stream(score=7, percentage=99, max_score=1000)],
                           max_score=20)
    result = [e for e in parse_sse(resp.text) if e["type"] == "result"][0]["result"]
    assert result["max_score"] == 20 and result["percentage"] == 35.0


@pytest.mark.asyncio
async def test_stream_document_isolation(client) -> None:
    resp, llm, _ = await run(client, [good_stream()])
    result = [e for e in parse_sse(resp.text) if e["type"] == "result"][0]["result"]
    assert "chunk_B1" not in [s["chunk_id"] for s in result["sources"]]
    system_prompt, user_prompt = llm.generate_stream.call_args.args
    assert "SECRET B CONTENT" not in user_prompt
    assert llm.generate_stream.call_args.kwargs == {"json_mode": True}


# ── Client disconnect ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_client_disconnect(caplog) -> None:
    state = {"closed": False}
    llm = make_llm()

    def factory(*_a: Any, **_k: Any) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            try:
                for p in good_stream():
                    yield p
                    await asyncio.sleep(0)
            finally:
                state["closed"] = True
        return gen()

    llm.generate_stream = MagicMock(side_effect=factory)
    svc, _ = make_stack(default_hits(), llm)

    with caplog.at_level(logging.WARNING):
        agen = svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)
        frames = []
        async for frame in agen:
            frames.append(frame)
            if len(frames) == 5:  # client goes away mid-stream
                break
        await agen.aclose()

    assert state["closed"] is True  # upstream LLM stream was closed
    assert not any('"type":"done"' in f or '"type":"result"' in f for f in frames)
    assert any("cancelled" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stream_task_cancellation_is_not_swallowed(caplog) -> None:
    llm = make_llm()

    def factory(*_a: Any, **_k: Any) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield "{"
            await asyncio.sleep(30)  # stalled model
        return gen()

    llm.generate_stream = MagicMock(side_effect=factory)
    svc, _ = make_stack(default_hits(), llm)
    seen: list[str] = []

    async def consume() -> None:
        async for frame in svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10):
            seen.append(frame)

    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not any('"type":"done"' in f for f in seen)
    assert any("cancelled" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stream_logs_first_token_latency(caplog) -> None:
    llm = make_llm()
    set_stream(llm, good_stream())
    svc, _ = make_stack(default_hits(), llm)
    with caplog.at_level(logging.INFO):
        _ = [f async for f in svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)]
    line = next(r.message for r in caplog.records if r.message.startswith("Evaluation stream |"))
    assert "first_token_latency_ms=" in line and "first_token_latency_ms=null" not in line
    assert "total_latency_ms=" in line and "prompt_version=v1" in line and "score=8.0" in line
    assert STUDENT not in line

    llm2 = make_llm()
    set_stream(llm2, [([], AppError(ErrorCode.OLLAMA_UNAVAILABLE, "down"))])
    svc2, _ = make_stack(default_hits(), llm2)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _ = [f async for f in svc2.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)]
    line = next(r.message for r in caplog.records if r.message.startswith("Evaluation stream |"))
    assert "first_token_latency_ms=null" in line


# ── SSE utility ─────────────────────────────────────────────────────


def test_sse_formatting() -> None:
    assert format_sse({"type": "token", "content": "Xin chào"}) == (
        'data: {"type":"token","content":"Xin chào"}\n\n')
    assert status_event("a") == 'data: {"type":"status","message":"a"}\n\n'
    assert token_event(" x\n") == 'data: {"type":"token","content":" x\\n"}\n\n'
    assert done_event() == 'data: {"type":"done"}\n\n'
    assert json.loads(error_event("C", "m").removeprefix("data: ")) == {
        "type": "error", "code": "C", "message": "m"}


# ── Ollama streaming implementation (httpx.MockTransport) ───────────


def ollama_service(handler) -> OllamaLLMService:
    svc = OllamaLLMService(base_url="http://ollama.test", model="qwen-test", timeout=5)
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return svc


def ndjson(*objs: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(o) for o in objs) + "\n").encode()


@pytest.mark.asyncio
async def test_ollama_generate_stream_yields_pieces_unchanged() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson(
            {"model": "m", "created_at": "t", "message": {"role": "assistant", "content": "The"}, "done": False},
            {"model": "m", "message": {"role": "assistant", "content": " answer"}, "done": False},
            {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True,
             "total_duration": 5},
        ))

    svc = ollama_service(handler)
    pieces = [p async for p in svc.generate_stream("sys", "usr", json_mode=True)]
    assert pieces == ["The", " answer"]
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["stream"] is True and seen["body"]["format"] == "json"
    assert seen["body"]["model"] == "qwen-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.asyncio
async def test_ollama_non_stream_payload_unchanged() -> None:
    svc = OllamaLLMService(base_url="http://x", model="m", timeout=5)
    payload_ = svc._build_payload("s", "u", json_mode=False, temperature=None, max_tokens=None)
    assert payload_["stream"] is False and "format" not in payload_


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,code", [
    (httpx.ReadTimeout("slow"), ErrorCode.OLLAMA_TIMEOUT),
    (httpx.ConnectError("refused"), ErrorCode.OLLAMA_UNAVAILABLE),
    (httpx.RemoteProtocolError("dropped"), ErrorCode.OLLAMA_UNAVAILABLE),
])
async def test_ollama_generate_stream_transport_errors(exc: Exception, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    svc = ollama_service(handler)
    with pytest.raises(AppError) as err:
        _ = [p async for p in svc.generate_stream("s", "u")]
    assert err.value.code == code


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [
    (404, ErrorCode.OLLAMA_MODEL_NOT_FOUND), (500, ErrorCode.OLLAMA_GENERATION_ERROR)])
async def test_ollama_generate_stream_http_status(status: int, code: str) -> None:
    svc = ollama_service(lambda r: httpx.Response(status, text="Traceback secret"))
    with pytest.raises(AppError) as err:
        _ = [p async for p in svc.generate_stream("s", "u")]
    assert err.value.code == code and "secret" not in err.value.message


@pytest.mark.asyncio
async def test_ollama_generate_stream_bad_payloads() -> None:
    cases = [
        (b"not json\n", ErrorCode.OLLAMA_INVALID_RESPONSE),
        (ndjson({"error": "model crashed"}), ErrorCode.OLLAMA_GENERATION_ERROR),
        (ndjson({"message": {"content": ""}, "done": True}), ErrorCode.LLM_EMPTY_RESPONSE),
        (ndjson({"message": {"content": 5}, "done": False}), ErrorCode.OLLAMA_INVALID_RESPONSE),
    ]
    for body, code in cases:
        svc = ollama_service(lambda r, b=body: httpx.Response(200, content=b))
        with pytest.raises(AppError) as err:
            _ = [p async for p in svc.generate_stream("s", "u")]
        assert err.value.code == code
        assert "model crashed" not in err.value.message


@pytest.mark.asyncio
async def test_default_llm_interface_stream_not_supported() -> None:
    from app.services.llm_service import LLMService

    class Minimal(LLMService):
        model_name = "x"  # type: ignore[assignment]

        async def generate(self, *a, **k): return ""
        async def generate_json(self, *a, **k): return {}
        async def check_health(self): return {}

    with pytest.raises(NotImplementedError):
        _ = [p async for p in Minimal().generate_stream("s", "u")]
