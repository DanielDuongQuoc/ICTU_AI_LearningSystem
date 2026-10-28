"""Phase 20 - Ollama failure injection, retry policy and request time budget (offline, httpx MockTransport)."""

from __future__ import annotations

import time

import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.request_context import deadline_var, remaining_s
from app.core.timeouts import can_retry, call_timeout, timeout_chain
from app.services.llm_service import OllamaLLMService


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


def _svc(handler) -> OllamaLLMService:
    svc = OllamaLLMService(base_url="http://ollama.test", model="m")
    svc._retry_backoff_s = 0.0
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return svc


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})


def _count(name: str, outcome: str) -> int:
    return metrics.snapshot()["events"].get((name, outcome), 0)


# ---------------------------------------------------------------- failure -> detection -> response
@pytest.mark.asyncio
@pytest.mark.parametrize("exc,code", [
    (httpx.ConnectError("refused"), ErrorCode.OLLAMA_UNAVAILABLE),
    (httpx.ReadTimeout("slow"), ErrorCode.OLLAMA_TIMEOUT),
    (httpx.ConnectTimeout("slow"), ErrorCode.OLLAMA_TIMEOUT),
])
async def test_transport_failures_map_to_codes_without_retry(exc, code):
    calls = []

    def handler(request):
        calls.append(1)
        raise exc

    svc = _svc(handler)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == code
    assert len(calls) == 1, "refused / timed-out calls must NOT be retried (no retry storm)"
    assert "Traceback" not in e.value.message
    await svc.close()


@pytest.mark.asyncio
async def test_model_not_found_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": "model 'm' not found"})

    svc = _svc(handler)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == ErrorCode.OLLAMA_MODEL_NOT_FOUND
    assert len(calls) == 1
    await svc.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"<html>not json</html>"),
    httpx.Response(200, json={"unexpected": "shape"}),
])
async def test_malformed_or_unexpected_response_is_controlled(response):
    svc = _svc(lambda request: response)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == ErrorCode.OLLAMA_INVALID_RESPONSE
    await svc.close()


# ---------------------------------------------------------------- retry policy
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [502, 503, 504])
async def test_transient_status_retried_exactly_once_then_succeeds(status):
    seq = [httpx.Response(status), _ok()]
    svc = _svc(lambda request: seq.pop(0))
    assert await svc.generate("s", "u") == "ok"
    assert _count("llm_retry", "transient") == 1
    await svc.close()


@pytest.mark.asyncio
async def test_persistent_503_is_bounded_to_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    svc = _svc(handler)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code in (ErrorCode.OLLAMA_UNAVAILABLE, ErrorCode.OLLAMA_GENERATION_ERROR)
    assert len(calls) == 1 + get_settings().ollama_max_retries
    await svc.close()


@pytest.mark.asyncio
async def test_connection_reset_retried_once():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.RemoteProtocolError("server disconnected")
        return _ok()

    svc = _svc(handler)
    assert await svc.generate("s", "u") == "ok"
    assert _count("llm_retry", "connection_reset") == 1
    await svc.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 422, 500])
async def test_non_transient_status_never_retried(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"error": "x"})

    svc = _svc(handler)
    with pytest.raises(AppError):
        await svc.generate("s", "u")
    assert len(calls) == 1
    await svc.close()


# ---------------------------------------------------------------- timeout chain / request budget
def test_timeout_chain_is_consistent():
    s = get_settings()
    chain = timeout_chain()
    assert chain["request_timeout_s"] >= chain["ollama_timeout_s"] >= chain["ollama_connect_timeout_s"]
    assert chain["worst_case_single_llm_call_s"] >= chain["ollama_timeout_s"]
    assert s.request_timeout_s >= s.ollama_timeout


def test_no_budget_outside_a_request():
    assert remaining_s() is None
    assert call_timeout(120.0, 5.0) is None
    assert can_retry(0.5) is True


def test_budget_tightens_then_refuses_then_blocks_retry():
    token = deadline_var.set(time.monotonic() + 10.0)
    try:
        t = call_timeout(120.0, 5.0)
        assert isinstance(t, httpx.Timeout) and t.read <= 10.0       # budget binds
        assert call_timeout(5.0, 1.0) is None                          # own timeout is tighter
    finally:
        deadline_var.reset(token)
    token = deadline_var.set(time.monotonic() + 0.2)
    try:
        before = _count("request_budget", "exhausted")
        with pytest.raises(AppError) as e:
            call_timeout(120.0, 5.0)
        assert e.value.code == ErrorCode.OLLAMA_TIMEOUT
        assert _count("request_budget", "exhausted") == before + 1
        assert can_retry(0.5) is False
        assert _count("request_budget", "retry_skipped") >= 1
    finally:
        deadline_var.reset(token)


@pytest.mark.asyncio
async def test_exhausted_budget_makes_no_ollama_call():
    calls = []

    def handler(request):
        calls.append(1)
        return _ok()

    svc = _svc(handler)
    token = deadline_var.set(time.monotonic() - 1.0)
    try:
        with pytest.raises(AppError) as e:
            await svc.generate("s", "u")
    finally:
        deadline_var.reset(token)
    assert e.value.code == ErrorCode.OLLAMA_TIMEOUT and calls == []
    await svc.close()


@pytest.mark.asyncio
async def test_budget_blocks_retry_when_too_little_time_is_left():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    svc = _svc(handler)
    token = deadline_var.set(time.monotonic() + 1.5)       # enough for one call, not for backoff + retry
    svc._retry_backoff_s = 1.0
    try:
        with pytest.raises(AppError):
            await svc.generate("s", "u")
    finally:
        deadline_var.reset(token)
    assert len(calls) == 1
    await svc.close()


@pytest.mark.asyncio
async def test_llm_events_recorded_for_ok_and_failure():
    state = {"fail": True}

    def handler(request):
        if state["fail"]:
            raise httpx.ReadTimeout("slow")
        return _ok()

    svc = _svc(handler)
    with pytest.raises(AppError):
        await svc.generate("s", "u")
    state["fail"] = False
    await svc.generate("s", "u")
    assert _count("llm_request", "timeout") == 1 and _count("llm_request", "ok") == 1
    await svc.close()
