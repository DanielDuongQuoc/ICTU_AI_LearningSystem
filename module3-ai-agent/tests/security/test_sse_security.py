"""SSE stream security: contract on failure, cancellation cleanup, wall-clock and size caps (Phase 19)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from app.core.prompts import EVALUATION_SYSTEM_PROMPT
from tests.security.conftest import TEXT_A, eval_body, upload_text


def parse(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


def types(events) -> list[str]:
    return [e["type"] for e in events]


async def _stream(client, doc, **over):
    body = eval_body(doc)
    body.update(over)
    return await client.post("/api/evaluations/evaluate/stream", json=body)


async def test_happy_path_contract(client, stack):
    doc = await upload_text(client, TEXT_A)
    resp = await _stream(client, doc)
    ev = types(parse(resp.text))
    assert ev[0] == "status" and ev[-1] == "done" and "result" in ev and "error" not in ev
    assert EVALUATION_SYSTEM_PROMPT[:60] not in resp.text


async def test_unlimited_generation_is_cut_off_by_wall_clock(client, stack, set_limit):
    doc = await upload_text(client, TEXT_A)
    set_limit(stream_max_duration_s=0.2)
    state = {"closed": False}

    async def endless(system_prompt, user_prompt, **_) -> AsyncIterator[str]:
        try:
            while True:
                yield "x"
                await asyncio.sleep(0.02)
        finally:
            state["closed"] = True

    stack.llm.generate_stream = endless
    resp = await asyncio.wait_for(_stream(client, doc), timeout=10)
    ev = types(parse(resp.text))
    assert ev[-2:] == ["error", "done"] and "result" not in ev
    assert state["closed"] is True                      # upstream generation was closed, not leaked


async def test_oversized_model_output_is_cut_off(client, stack, set_limit):
    doc = await upload_text(client, TEXT_A)
    set_limit(max_llm_output_chars=100)

    async def flood(system_prompt, user_prompt, **_) -> AsyncIterator[str]:
        for _ in range(1000):
            yield "y" * 50

    stack.llm.generate_stream = flood
    resp = await _stream(client, doc)
    ev = types(parse(resp.text))
    assert ev[-2:] == ["error", "done"] and "result" not in ev
    assert len(resp.text) < 20_000


async def test_streamed_prompt_leak_never_becomes_a_result(client, stack):
    doc = await upload_text(client, TEXT_A)
    stack.llm.eval_payload = {**stack.llm.eval_payload, "feedback": EVALUATION_SYSTEM_PROMPT}
    resp = await _stream(client, doc)
    ev = types(parse(resp.text))
    assert "result" not in ev and ev[-2:] == ["error", "done"]


async def test_mid_stream_exception_is_error_then_done_without_internals(client, stack):
    doc = await upload_text(client, TEXT_A)

    async def broken(system_prompt, user_prompt, **_) -> AsyncIterator[str]:
        yield '{"score": '
        raise RuntimeError("SECRET-INTERNAL C:\\Users\\bob\\x.py Traceback")

    stack.llm.generate_stream = broken
    resp = await _stream(client, doc)
    assert resp.status_code == 200
    assert types(parse(resp.text))[-2:] == ["error", "done"]
    assert "SECRET-INTERNAL" not in resp.text and "Traceback" not in resp.text and "C:\\Users" not in resp.text


async def test_client_disconnect_closes_upstream_generation(client, stack):
    doc = await upload_text(client, TEXT_A)
    state = {"closed": False}

    async def slow(system_prompt, user_prompt, **_) -> AsyncIterator[str]:
        try:
            for _ in range(10_000):
                yield "z"
                await asyncio.sleep(0)
        finally:
            state["closed"] = True

    stack.llm.generate_stream = slow
    agen = stack.evaluator.evaluate_answer_stream(doc, "What do zebracorn models learn from?",
                                                  "Zebracorn models learn.", "They learn patterns.", 10)
    frames = []
    async for frame in agen:
        frames.append(frame)
        if len(frames) >= 4:
            break
    await agen.aclose()
    assert state["closed"] is True
    assert not any('"type":"done"' in f for f in frames)


async def test_task_cancellation_propagates_and_cleans_up(client, stack):
    doc = await upload_text(client, TEXT_A)
    state = {"closed": False}

    async def forever(system_prompt, user_prompt, **_) -> AsyncIterator[str]:
        try:
            while True:
                yield "z"
                await asyncio.sleep(0.01)
        finally:
            state["closed"] = True

    stack.llm.generate_stream = forever

    async def consume():
        async for _ in stack.evaluator.evaluate_answer_stream(doc, "What do zebracorn models learn from?",
                                                              "Zebracorn models learn.", "They learn patterns.", 10):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["closed"] is True


async def test_rate_limit_rejects_stream_before_it_starts(client, stack, set_limit):
    doc = await upload_text(client, TEXT_A)
    from app.core.rate_limit import limiter
    limiter.reset()                                      # the upload above used the same budget
    set_limit(rate_limit_per_minute=1)
    first = await _stream(client, doc)
    assert first.status_code == 200
    second = await _stream(client, doc)
    assert second.status_code == 429 and second.headers.get("retry-after")
    assert second.json()["error"]["code"] == "RATE_LIMITED"
