"""Phase 16 - LLM concurrency limit / backpressure (no real Ollama; httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import ERROR_HTTP_STATUS, AppError, ErrorCode
from app.core.metrics import metrics
from app.services.llm_service import OllamaLLMService

CHAT_OK = {"message": {"content": "hello"}, "done": True}


@pytest.fixture()
def limits(monkeypatch):
    def apply(max_concurrency: int, queue_timeout: float = 0.2) -> None:
        monkeypatch.setenv("LLM_MAX_CONCURRENCY", str(max_concurrency))
        monkeypatch.setenv("LLM_QUEUE_TIMEOUT_S", str(queue_timeout))
        get_settings.cache_clear()

    metrics.reset()
    yield apply
    get_settings.cache_clear()
    metrics.reset()


def _service(handler) -> OllamaLLMService:
    svc = OllamaLLMService(base_url="http://ollama.test", model="m")
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return svc


def test_overload_error_is_503_and_documented():
    assert ERROR_HTTP_STATUS[ErrorCode.SERVICE_OVERLOADED] == 503
    assert AppError(ErrorCode.SERVICE_OVERLOADED, "x").status_code == 503


@pytest.mark.parametrize("env,value", [("LLM_MAX_CONCURRENCY", "-1"), ("LLM_MAX_CONCURRENCY", "x"),
                                       ("LLM_QUEUE_TIMEOUT_S", "0"), ("LLM_QUEUE_TIMEOUT_S", "-3")])
def test_invalid_limits_are_rejected(monkeypatch, env, value):
    from app.core.config import Settings

    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.asyncio
async def test_excess_call_is_rejected_then_slot_is_reusable(limits):
    limits(1, 0.2)
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    first = asyncio.create_task(svc.generate("s", "u"))
    await asyncio.sleep(0.05)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.SERVICE_OVERLOADED and exc.value.status_code == 503
    assert "module3_llm_rejected_total 1" in metrics.render()
    assert "module3_llm_waiting 0" in metrics.render()

    release.set()
    assert await first == "hello"
    assert await svc.generate("s", "u") == "hello"          # slot released after completion
    assert "module3_llm_active 0" in metrics.render()
    await svc.close()


@pytest.mark.asyncio
async def test_cancellation_releases_slot_and_does_not_affect_others(limits):
    limits(1, 1.0)
    block = asyncio.Event()
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            await block.wait()
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    victim = asyncio.create_task(svc.generate("s", "u"))
    await asyncio.sleep(0.05)
    waiter = asyncio.create_task(svc.generate("s", "u"))     # queued behind the victim
    await asyncio.sleep(0.05)
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim
    assert await asyncio.wait_for(waiter, 2) == "hello"       # queued request proceeds
    assert "module3_llm_active 0" in metrics.render()
    await svc.close()


@pytest.mark.asyncio
async def test_failure_inside_slot_releases_it(limits):
    limits(1, 0.2)
    state = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE
    state["fail"] = False
    assert await svc.generate("s", "u") == "hello"
    await svc.close()


@pytest.mark.asyncio
async def test_limit_zero_means_unlimited(limits):
    limits(0, 0.2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    results = await asyncio.gather(*(svc.generate("s", "u") for _ in range(8)))
    assert results == ["hello"] * 8
    await svc.close()


@pytest.mark.asyncio
async def test_concurrent_burst_within_limit_all_succeed(limits):
    limits(2, 5.0)
    active = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    results = await asyncio.gather(*(svc.generate("s", "u") for _ in range(10)))
    assert results == ["hello"] * 10 and active["peak"] <= 2
    await svc.close()


@pytest.mark.asyncio
async def test_stream_holds_slot_until_closed(limits):
    limits(1, 0.2)
    release = asyncio.Event()

    async def body():
        yield (json.dumps({"message": {"content": "hi"}, "done": False}) + "\n").encode()
        await release.wait()
        yield (json.dumps({"message": {"content": "!"}, "done": True}) + "\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=body())
        return httpx.Response(200, json=CHAT_OK)

    svc = _service(handler)
    stream = svc.generate_stream("s", "u")
    assert await stream.__anext__() == "hi"                 # slot now held by the open stream
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.SERVICE_OVERLOADED

    await stream.aclose()                                    # client disconnect / cancel
    assert "module3_llm_active 0" in metrics.render()
    assert await svc.generate("s", "u") == "hello"           # slot is free again
    release.set()
    await svc.close()
