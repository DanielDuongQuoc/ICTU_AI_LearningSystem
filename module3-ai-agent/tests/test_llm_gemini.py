"""Gemini LLM + embedding services and provider health probes.

Everything runs against ``httpx.MockTransport`` (see ``tests/fakes.py``): request shape, response parsing,
JSON retry, SSE stream parsing, retry policy (429/5xx), timeouts / request budget, error mapping and the
guarantee that neither the API key nor prompt/model text ends up in logs or error messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.request_context import deadline_var, remaining_s
from app.core.timeouts import call_timeout, can_retry
from app.services.embedding_service import GeminiEmbeddingService
from app.services.llm_service import GeminiLLMService, LLMService
from tests.fakes import (
    GEMINI_TEST_KEY,
    GEMINI_TEST_URL,
    embed_request_texts,
    gemini_body,
    gemini_embedding,
    gemini_embeddings_body,
    gemini_llm,
    gemini_ok,
    gemini_sse,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "0")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    get_settings.cache_clear()
    metrics.reset()
    yield
    get_settings.cache_clear()


def _count(name: str, outcome: str) -> int:
    return metrics.snapshot()["events"].get((name, outcome), 0)


class Recorder:
    """MockTransport handler that records requests and replays a scripted sequence of responses/errors."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


# ============================================================================ generate (non-streaming)

async def test_generate_request_shape_and_key_header():
    rec = Recorder(gemini_ok("Hello there"))
    svc = gemini_llm(rec, model="models/gemini-test", temperature=0.1, max_output_tokens=64)
    assert await svc.generate("SYS", "USER") == "Hello there"
    req = rec.requests[0]
    assert req.url.path == "/v1beta/models/gemini-test:generateContent"        # "models/" prefix normalised
    assert req.headers["x-goog-api-key"] == GEMINI_TEST_KEY
    assert GEMINI_TEST_KEY not in str(req.url)                                  # key never in the URL
    body = rec.bodies[0]
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    assert body["generationConfig"] == {"temperature": 0.1, "maxOutputTokens": 64}
    await svc.close()


async def test_per_call_overrides_and_json_mode():
    rec = Recorder(gemini_ok("x"), gemini_ok('{"answer": "42"}'))
    svc = gemini_llm(rec)
    await svc.generate("s", "u", temperature=0.9, max_tokens=5)
    assert rec.bodies[0]["generationConfig"] == {"temperature": 0.9, "maxOutputTokens": 5}
    assert await svc.generate_json("s", "u", required_keys=["answer"]) == {"answer": "42"}
    assert rec.bodies[1]["generationConfig"]["responseMimeType"] == "application/json"
    await svc.close()


async def test_empty_system_prompt_is_omitted():
    rec = Recorder(gemini_ok("x"))
    svc = gemini_llm(rec)
    await svc.generate("", "u")
    assert "systemInstruction" not in rec.bodies[0]
    await svc.close()


def test_configuration_comes_from_settings():
    s = get_settings()
    svc = GeminiLLMService()
    assert svc.model_name == s.gemini_model
    assert svc._temperature == s.llm_temperature
    assert svc._max_tokens == s.llm_max_output_tokens
    assert svc._timeout == s.llm_timeout
    assert svc._max_retries == s.llm_max_retries


def test_interface_and_no_hardcoded_model_names():
    import app.services.llm_service as llm_mod
    import app.services.rag_service as rag_mod

    assert issubclass(GeminiLLMService, LLMService)
    with pytest.raises(TypeError):
        LLMService()  # type: ignore[abstract]
    for mod in (llm_mod, rag_mod):
        assert "gemini-3" not in open(mod.__file__, encoding="utf-8").read()


async def test_thought_parts_are_skipped_and_text_parts_joined():
    body = {"candidates": [{"content": {"parts": [
        {"text": "internal reasoning", "thought": True}, {"text": "Hel"}, {"text": "lo"}]}}]}
    svc = gemini_llm(Recorder(httpx.Response(200, json=body)))
    assert await svc.generate("s", "u") == "Hello"
    await svc.close()


@pytest.mark.parametrize("text", ["", "   \n"])
async def test_empty_response(text):
    svc = gemini_llm(Recorder(gemini_ok(text)))
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.LLM_EMPTY_RESPONSE
    await svc.close()


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"<html>not json</html>"),
    httpx.Response(200, json={"unexpected": "shape"}),
    httpx.Response(200, json={"candidates": []}),            # e.g. prompt blocked by safety filters
    httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY"}]}),
])
async def test_malformed_or_unexpected_response_is_controlled(response):
    svc = gemini_llm(Recorder(response))
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.LLM_INVALID_RESPONSE
    await svc.close()


@pytest.mark.parametrize("exc,code", [
    (httpx.ConnectError("refused"), ErrorCode.LLM_UNAVAILABLE),
    (httpx.ReadTimeout("slow"), ErrorCode.LLM_TIMEOUT),
    (httpx.ConnectTimeout("slow"), ErrorCode.LLM_TIMEOUT),
])
async def test_transport_failures_map_to_codes_without_retry(exc, code):
    rec = Recorder(exc)
    svc = gemini_llm(rec)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == code
    assert len(rec.requests) == 1, "refused / timed-out calls must NOT be retried (no retry storm)"
    assert "Traceback" not in e.value.message
    await svc.close()


async def test_model_not_found_is_not_retried():
    rec = Recorder(httpx.Response(404, json={"error": {"message": "models/x is not found"}}))
    svc = gemini_llm(rec)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == ErrorCode.LLM_MODEL_NOT_FOUND and len(rec.requests) == 1
    await svc.close()


async def test_error_status_never_leaks_upstream_body(caplog):
    svc = gemini_llm(Recorder(httpx.Response(400, text="Traceback secret stack USER-PROMPT-ECHO")))
    with caplog.at_level(logging.DEBUG), pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == ErrorCode.LLM_GENERATION_ERROR
    assert "Traceback" not in e.value.message and "USER-PROMPT-ECHO" not in caplog.text
    await svc.close()


async def test_missing_api_key_fails_fast_without_network_call():
    rec = Recorder(gemini_ok())
    svc = gemini_llm(rec, api_key="")
    for call in (svc.generate("s", "u"), svc.check_health()):
        with pytest.raises(AppError) as e:
            await call
        assert e.value.code == ErrorCode.LLM_UNAVAILABLE and e.value.status_code == 503
    with pytest.raises(AppError) as e:
        _ = [p async for p in svc.generate_stream("s", "u")]
    assert e.value.code == ErrorCode.LLM_UNAVAILABLE
    assert rec.requests == []
    await svc.close()


# ============================================================================ generate_json

async def test_json_markdown_fence_is_stripped():
    rec = Recorder(gemini_ok('```json\n{"answer": "fenced"}\n```'))
    svc = gemini_llm(rec)
    assert await svc.generate_json("s", "u") == {"answer": "fenced"}
    assert len(rec.requests) == 1
    await svc.close()


async def test_invalid_json_fails_after_exactly_one_retry():
    rec = Recorder(gemini_ok("this is not json"))
    svc = gemini_llm(rec)
    with pytest.raises(AppError) as exc:
        await svc.generate_json("s", "u")
    assert exc.value.code == ErrorCode.LLM_INVALID_JSON
    assert len(rec.requests) == 2  # initial + ONE retry, never more
    await svc.close()


async def test_json_retry_succeeds_and_prompt_is_strengthened():
    rec = Recorder(gemini_ok("garbage"), gemini_ok('{"answer": "ok"}'))
    svc = gemini_llm(rec)
    assert await svc.generate_json("s", "u", required_keys=["answer"]) == {"answer": "ok"}
    first, retry = (b["contents"][0]["parts"][0]["text"] for b in rec.bodies)
    assert first == "u"
    assert "Return ONLY valid JSON." in retry and "Do not include Markdown." in retry
    await svc.close()


async def test_missing_required_key_triggers_retry():
    rec = Recorder(gemini_ok('{"foo": 1}'), gemini_ok('{"answer": "fixed"}'))
    svc = gemini_llm(rec)
    assert (await svc.generate_json("s", "u", required_keys=["answer"]))["answer"] == "fixed"
    assert len(rec.requests) == 2
    await svc.close()


async def test_json_does_not_retry_on_transport_errors():
    rec = Recorder(httpx.ConnectError("refused"))
    svc = gemini_llm(rec)
    with pytest.raises(AppError) as exc:
        await svc.generate_json("s", "u")
    assert exc.value.code == ErrorCode.LLM_UNAVAILABLE and len(rec.requests) == 1
    await svc.close()


async def test_oversized_model_output_is_rejected_not_parsed(monkeypatch):
    monkeypatch.setenv("MAX_LLM_OUTPUT_CHARS", "50")
    get_settings.cache_clear()
    svc = gemini_llm(Recorder(gemini_ok(json.dumps({"answer": "x" * 200}))))
    with pytest.raises(AppError) as exc:
        await svc.generate_json("s", "u")
    assert exc.value.code == ErrorCode.LLM_OUTPUT_REJECTED
    await svc.close()


async def test_invalid_json_log_never_contains_model_output(caplog):
    svc = gemini_llm(Recorder(gemini_ok("MODEL-OUTPUT-CANARY not json")))
    with caplog.at_level(logging.DEBUG), pytest.raises(AppError):
        await svc.generate_json("s", "u")
    assert "MODEL-OUTPUT-CANARY" not in caplog.text and "invalid JSON" in caplog.text
    await svc.close()


# ============================================================================ retry policy

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_transient_status_retried_exactly_once_then_succeeds(status):
    rec = Recorder(httpx.Response(status), gemini_ok("ok"))
    svc = gemini_llm(rec)
    assert await svc.generate("s", "u") == "ok"
    assert len(rec.requests) == 2 and _count("llm_retry", "transient") == 1
    await svc.close()


@pytest.mark.parametrize("status", [429, 503])
async def test_persistent_transient_status_is_bounded_to_max_retries(status):
    rec = Recorder(httpx.Response(status, json={"error": {"status": "RESOURCE_EXHAUSTED"}}))
    svc = gemini_llm(rec)
    with pytest.raises(AppError) as e:
        await svc.generate("s", "u")
    assert e.value.code == ErrorCode.LLM_GENERATION_ERROR and e.value.status_code == 502
    assert len(rec.requests) == 1 + get_settings().llm_max_retries
    await svc.close()


async def test_retry_waits_for_the_configured_backoff():
    rec = Recorder(httpx.Response(429), gemini_ok("ok"))
    svc = gemini_llm(rec, backoff=0.05)
    started = time.perf_counter()
    assert await svc.generate("s", "u") == "ok"
    assert time.perf_counter() - started >= 0.05
    await svc.close()


async def test_connection_reset_retried_once():
    rec = Recorder(httpx.RemoteProtocolError("server disconnected"), gemini_ok("ok"))
    svc = gemini_llm(rec)
    assert await svc.generate("s", "u") == "ok"
    assert _count("llm_retry", "connection_reset") == 1
    await svc.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_non_transient_status_never_retried(status):
    rec = Recorder(httpx.Response(status, json={"error": "x"}))
    svc = gemini_llm(rec)
    with pytest.raises(AppError):
        await svc.generate("s", "u")
    assert len(rec.requests) == 1
    await svc.close()


async def test_provider_down_then_recovers_on_the_same_instance():
    state = {"down": True}

    def handler(request):
        if state["down"]:
            raise httpx.ConnectError("refused")
        return gemini_ok("ok")

    svc = gemini_llm(handler)
    with pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.LLM_UNAVAILABLE
    state["down"] = False
    assert await svc.generate("s", "u") == "ok"          # no restart needed
    await svc.close()


async def test_llm_events_recorded_for_ok_and_failure():
    rec = Recorder(httpx.ReadTimeout("slow"), gemini_ok("ok"))
    svc = gemini_llm(rec)
    with pytest.raises(AppError):
        await svc.generate("s", "u")
    await svc.generate("s", "u")
    assert _count("llm_request", "timeout") == 1 and _count("llm_request", "ok") == 1
    await svc.close()


# ============================================================================ timeouts / request budget

def test_timeout_chain_is_consistent():
    s = get_settings()
    assert s.request_timeout_s >= s.llm_timeout >= s.llm_connect_timeout
    assert s.llm_health_timeout <= s.llm_timeout


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
        assert e.value.code == ErrorCode.LLM_TIMEOUT
        assert _count("request_budget", "exhausted") == before + 1
        assert can_retry(0.5) is False
        assert _count("request_budget", "retry_skipped") >= 1
    finally:
        deadline_var.reset(token)


async def test_exhausted_budget_makes_no_provider_call():
    rec = Recorder(gemini_ok())
    svc = gemini_llm(rec)
    token = deadline_var.set(time.monotonic() - 1.0)
    try:
        with pytest.raises(AppError) as e:
            await svc.generate("s", "u")
    finally:
        deadline_var.reset(token)
    assert e.value.code == ErrorCode.LLM_TIMEOUT and rec.requests == []
    await svc.close()


async def test_budget_blocks_retry_when_too_little_time_is_left():
    rec = Recorder(httpx.Response(503))
    svc = gemini_llm(rec, backoff=1.0)
    token = deadline_var.set(time.monotonic() + 1.5)       # enough for one call, not for backoff + retry
    try:
        with pytest.raises(AppError):
            await svc.generate("s", "u")
    finally:
        deadline_var.reset(token)
    assert len(rec.requests) == 1
    await svc.close()


# ============================================================================ streaming (SSE)

async def test_stream_yields_pieces_in_order_and_uses_sse_endpoint():
    rec = Recorder(httpx.Response(200, content=gemini_sse("The", " answer", "", done=True)))
    svc = gemini_llm(rec)
    pieces = [p async for p in svc.generate_stream("sys", "usr", json_mode=True)]
    assert pieces == ["The", " answer"]
    req = rec.requests[0]
    assert req.url.path == "/v1beta/models/gemini-test:streamGenerateContent" and req.url.params["alt"] == "sse"
    assert rec.bodies[0]["generationConfig"]["responseMimeType"] == "application/json"
    assert rec.bodies[0]["systemInstruction"] == {"parts": [{"text": "sys"}]}
    await svc.close()


async def test_stream_skips_thoughts_and_frames_without_candidates():
    frames = (f"data: {json.dumps(gemini_body('Hi', thought='plan'))}\n\n"
              f"data: {json.dumps({'usageMetadata': {'totalTokenCount': 3}})}\n\n"
              f"data: {json.dumps(gemini_body('!'))}\n\n").encode()
    svc = gemini_llm(Recorder(httpx.Response(200, content=frames)))
    assert [p async for p in svc.generate_stream("s", "u")] == ["Hi", "!"]
    await svc.close()


@pytest.mark.parametrize("exc,code", [
    (httpx.ReadTimeout("slow"), ErrorCode.LLM_TIMEOUT),
    (httpx.ConnectError("refused"), ErrorCode.LLM_UNAVAILABLE),
    (httpx.RemoteProtocolError("dropped"), ErrorCode.LLM_UNAVAILABLE),
])
async def test_stream_transport_errors(exc, code):
    svc = gemini_llm(Recorder(exc))
    with pytest.raises(AppError) as err:
        _ = [p async for p in svc.generate_stream("s", "u")]
    assert err.value.code == code
    await svc.close()


@pytest.mark.parametrize("status,code", [(404, ErrorCode.LLM_MODEL_NOT_FOUND), (400, ErrorCode.LLM_GENERATION_ERROR)])
async def test_stream_http_status(status, code):
    svc = gemini_llm(Recorder(httpx.Response(status, text="Traceback secret")))
    with pytest.raises(AppError) as err:
        _ = [p async for p in svc.generate_stream("s", "u")]
    assert err.value.code == code and "secret" not in err.value.message
    await svc.close()


@pytest.mark.parametrize("status", [429, 503])
async def test_stream_transient_status_retried_once(status):
    rec = Recorder(httpx.Response(status, text="busy"), httpx.Response(200, content=gemini_sse("hi")))
    svc = gemini_llm(rec)
    assert [p async for p in svc.generate_stream("s", "u")] == ["hi"]
    assert len(rec.requests) == 2
    await svc.close()


async def test_stream_bad_payloads():
    cases = [
        (b"data: not json\n\n", ErrorCode.LLM_INVALID_RESPONSE),
        (b"data: [1, 2]\n\n", ErrorCode.LLM_INVALID_RESPONSE),
        (gemini_sse("", "  "), ErrorCode.LLM_EMPTY_RESPONSE),
        (b"", ErrorCode.LLM_EMPTY_RESPONSE),
    ]
    for body, code in cases:
        svc = gemini_llm(Recorder(httpx.Response(200, content=body)))
        with pytest.raises(AppError) as err:
            _ = [p async for p in svc.generate_stream("s", "u")]
        assert err.value.code == code, body
        await svc.close()


async def test_stream_interruption_and_invalid_json_then_recovery():
    mode = {"m": "timeout"}

    async def body():
        yield gemini_sse("par")
        if mode["m"] == "timeout":
            raise httpx.ReadTimeout("stream stalled")
        if mode["m"] == "garbage":
            yield b"data: {not json}\n\n"
            return
        yield gemini_sse("tial")

    svc = gemini_llm(lambda request: httpx.Response(200, content=body()))

    async def drain() -> str:
        return "".join([p async for p in svc.generate_stream("s", "u")])

    with pytest.raises(AppError) as exc:
        await drain()
    assert exc.value.code == ErrorCode.LLM_TIMEOUT
    mode["m"] = "garbage"
    with pytest.raises(AppError) as exc:
        await drain()
    assert exc.value.code == ErrorCode.LLM_INVALID_RESPONSE
    mode["m"] = "ok"
    assert await drain() == "partial"
    await svc.close()


async def test_default_llm_interface_stream_not_supported():
    class Minimal(LLMService):
        model_name = "x"  # type: ignore[assignment]

        async def generate(self, *a, **k): return ""
        async def generate_json(self, *a, **k): return {}
        async def check_health(self): return {}

    with pytest.raises(NotImplementedError):
        _ = [p async for p in Minimal().generate_stream("s", "u")]


# ============================================================================ service-level health

async def test_check_health_ok_and_uses_model_endpoint():
    rec = Recorder(httpx.Response(200, json={"name": "models/gemini-test"}))
    svc = gemini_llm(rec)
    assert await svc.check_health() == {"status": "ok", "model": "gemini-test"}
    assert rec.requests[0].url.path == "/v1beta/models/gemini-test"
    await svc.close()


@pytest.mark.parametrize("response,code", [
    (httpx.Response(404), ErrorCode.LLM_MODEL_NOT_FOUND),
    (httpx.Response(403), ErrorCode.LLM_UNAVAILABLE),
    (httpx.ConnectError("refused"), ErrorCode.LLM_UNAVAILABLE),
    (httpx.ReadTimeout("slow"), ErrorCode.LLM_UNAVAILABLE),
])
async def test_check_health_failures(response, code):
    svc = gemini_llm(Recorder(response))
    with pytest.raises(AppError) as exc:
        await svc.check_health()
    assert exc.value.code == code
    await svc.close()


# ============================================================================ provider_checker (/health probes)

class _ProbeClient:
    """Stand-in for ``httpx.AsyncClient`` inside provider_checker (records the probed URL + headers)."""

    calls: list[tuple[str, dict]] = []
    result: Any = None

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        _ProbeClient.calls.append((url, headers or {}))
        if isinstance(_ProbeClient.result, Exception):
            raise _ProbeClient.result
        return _ProbeClient.result


@pytest.fixture()
def probe(monkeypatch):
    import app.services.provider_checker as pc

    _ProbeClient.calls = []
    monkeypatch.setattr(pc.httpx, "AsyncClient", _ProbeClient)
    return pc


@pytest.mark.parametrize("result,expected", [
    (httpx.Response(200, json={"models": []}), True),
    (httpx.Response(503), False),
    (httpx.Response(400, json={"error": {"message": "API key not valid"}}), False),
    (httpx.ConnectError("refused"), False),
    (httpx.TimeoutException("timeout"), False),
    (RuntimeError("unexpected"), False),
])
async def test_provider_health_probe(probe, result, expected):
    _ProbeClient.result = result
    assert await probe.check_provider_health() is expected
    url, headers = _ProbeClient.calls[0]
    assert url.startswith(get_settings().gemini_base_url) and "/models" in url
    assert headers["x-goog-api-key"] and headers["x-goog-api-key"] not in url


async def test_model_health_probe_targets_the_configured_model(probe, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "models/gemini-probe")
    get_settings.cache_clear()
    _ProbeClient.result = httpx.Response(200, json={"name": "models/gemini-probe"})
    assert await probe.check_model_health() is True
    assert _ProbeClient.calls[0][0].endswith("/models/gemini-probe")
    _ProbeClient.result = httpx.Response(404)
    assert await probe.check_model_health() is False


async def test_probes_without_api_key_report_down_without_network(probe, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    assert await probe.check_provider_health() is False
    assert await probe.check_model_health() is False
    assert _ProbeClient.calls == []


# ============================================================================ embeddings

async def test_embedding_request_shape_and_single_vector():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=gemini_embeddings_body([[0.1, -0.2, 0.3]]))

    svc = gemini_embedding(handler, model="models/embed-test")
    assert await svc.embed_text("Hello world") == [0.1, -0.2, 0.3]
    assert seen[0].url.path == "/v1beta/models/embed-test:batchEmbedContents"
    body = json.loads(seen[0].content)
    assert body == {"requests": [{"model": "models/embed-test", "content": {"parts": [{"text": "Hello world"}]}}]}
    assert svc.model_name == "models/embed-test"
    await svc.close()


async def test_embedding_configuration_comes_from_settings():
    s = get_settings()
    svc = GeminiEmbeddingService()
    assert svc.model_name == s.gemini_embedding_model and svc._timeout == s.llm_timeout


@pytest.mark.parametrize("call", [
    lambda s: s.embed_text(""), lambda s: s.embed_text("   \n\t  "),
    lambda s: s.embed_documents([]), lambda s: s.embed_documents(["valid", ""]),
])
async def test_embedding_empty_input_rejected_without_call(call):
    rec = Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0]])))
    svc = gemini_embedding(rec)
    with pytest.raises(AppError) as exc:
        await call(svc)
    assert exc.value.code == ErrorCode.EMBEDDING_EMPTY_TEXT and rec.requests == []


async def test_embeddings_are_sent_in_bounded_batches_and_keep_order(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "10")
    get_settings.cache_clear()
    sizes: list[int] = []

    def handler(request):
        texts = embed_request_texts(request)
        sizes.append(len(texts))
        return httpx.Response(200, json=gemini_embeddings_body([[float(t)] for t in texts]))

    svc = gemini_embedding(handler)
    texts = [str(i) for i in range(1, 36)]
    vectors = await svc.embed_documents(texts)
    assert sizes == [10, 10, 10, 5]
    assert vectors == [[float(t)] for t in texts], "order and count must be preserved across batches"
    await svc.close()


async def test_small_input_is_still_a_single_request():
    calls: list[int] = []

    def handler(request):
        n = len(embed_request_texts(request))
        calls.append(n)
        return httpx.Response(200, json=gemini_embeddings_body([[1.0, 2.0]] * n))

    svc = gemini_embedding(handler)
    assert len(await svc.embed_documents(["a", "b", "c"])) == 3
    assert calls == [3]
    await svc.close()


@pytest.mark.parametrize("body", [
    {"result": "ok"},                                        # no embeddings key
    {"embeddings": []},
    {"embeddings": [{"values": []}]},                        # empty vector
    {"embeddings": [{"values": ["a", "b"]}]},                # non-numeric
    {"embeddings": [{"values": [1.0, float("nan")]}]},       # non-finite
    {"embeddings": [{"other": 1}]},
])
async def test_embedding_invalid_vectors_rejected(body):
    content = json.dumps(body, allow_nan=True).encode()
    svc = gemini_embedding(Recorder(httpx.Response(200, content=content)))
    with pytest.raises(AppError) as exc:
        await svc.embed_text("hello")
    assert exc.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE


async def test_embedding_count_and_dimension_validation(monkeypatch):
    svc = gemini_embedding(Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0, 2.0]]))))
    with pytest.raises(AppError) as exc:                      # asked for 2, got 1
        await svc.embed_documents(["a", "b"])
    assert exc.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE
    svc = gemini_embedding(Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0, 2.0], [1.0]]))))
    with pytest.raises(AppError) as exc:                      # mixed dimensions inside one batch
        await svc.embed_documents(["a", "b"])
    assert "dimension" in exc.value.message
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1")          # mixed dimensions across batches
    get_settings.cache_clear()
    svc = gemini_embedding(Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0, 2.0]])),
                                    httpx.Response(200, json=gemini_embeddings_body([[1.0]]))))
    with pytest.raises(AppError) as exc:
        await svc.embed_documents(["a", "b"])
    assert exc.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE


@pytest.mark.parametrize("response,code", [
    (httpx.ConnectError("refused"), ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE),
    (httpx.ReadTimeout("timed out"), ErrorCode.EMBEDDING_TIMEOUT),
    (httpx.Response(404, text="model not found"), ErrorCode.EMBEDDING_MODEL_NOT_FOUND),
    (httpx.Response(200, text="not json at all"), ErrorCode.EMBEDDING_INVALID_RESPONSE),
    (httpx.Response(400, text="bad request"), ErrorCode.EMBEDDING_INVALID_RESPONSE),
])
async def test_embedding_error_mapping(response, code):
    svc = gemini_embedding(Recorder(response))
    with pytest.raises(AppError) as exc:
        await svc.embed_text("hello")
    assert exc.value.code == code


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_embedding_transient_status_retried_once(status):
    rec = Recorder(httpx.Response(status), httpx.Response(200, json=gemini_embeddings_body([[1.0]])))
    svc = gemini_embedding(rec)
    assert await svc.embed_text("x") == [1.0]
    assert len(rec.requests) == 2 and _count("embedding_retry", "transient") == 1
    rec = Recorder(httpx.Response(status))
    svc = gemini_embedding(rec)
    with pytest.raises(AppError) as exc:
        await svc.embed_text("x")
    assert exc.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE and len(rec.requests) == 2


async def test_embedding_connection_reset_retried_once():
    rec = Recorder(httpx.ReadError("reset"), httpx.Response(200, json=gemini_embeddings_body([[1.0]])))
    svc = gemini_embedding(rec)
    assert await svc.embed_text("x") == [1.0]
    assert _count("embedding_retry", "connection_reset") == 1


async def test_embedding_without_api_key_is_unavailable_without_call():
    rec = Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0]])))
    svc = gemini_embedding(rec, api_key="")
    with pytest.raises(AppError) as exc:
        await svc.embed_text("x")
    assert exc.value.code == ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE and exc.value.status_code == 503
    assert rec.requests == []


async def test_embedding_concurrency_never_exceeds_the_configured_limit(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "2")
    monkeypatch.setenv("EMBEDDING_QUEUE_TIMEOUT_S", "10")
    get_settings.cache_clear()
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(200, json=gemini_embeddings_body([[1.0]]))

    svc = gemini_embedding(handler)
    await asyncio.gather(*[svc.embed_text(f"q{i}") for i in range(12)])
    assert peak == 2
    await svc.close()


async def test_embedding_overload_is_rejected_with_503_not_queued_forever(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "1")
    monkeypatch.setenv("EMBEDDING_QUEUE_TIMEOUT_S", "0.05")
    get_settings.cache_clear()
    release = asyncio.Event()

    async def handler(request):
        await release.wait()
        return httpx.Response(200, json=gemini_embeddings_body([[1.0]]))

    svc = gemini_embedding(handler)
    first = asyncio.create_task(svc.embed_text("slow"))
    await asyncio.sleep(0.01)
    with pytest.raises(AppError) as err:
        await svc.embed_text("second")
    assert err.value.code == ErrorCode.SERVICE_OVERLOADED
    release.set()
    assert await first == [1.0]
    assert await svc.embed_text("after") == [1.0], "service recovers once the slot is free"
    await svc.close()


# ============================================================================ log redaction

async def test_api_key_and_prompts_never_logged(caplog):
    key = "AIza-REDACTION-CANARY-KEY-0123456789"
    rec = Recorder(httpx.Response(503), httpx.ConnectError(f"refused {GEMINI_TEST_URL}"), gemini_ok("ANSWER-CANARY"))
    svc = gemini_llm(rec, api_key=key)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AppError):
            await svc.generate("SYSTEM-CANARY", "PROMPT-CANARY")
        await svc.generate("SYSTEM-CANARY", "PROMPT-CANARY")
        emb = gemini_embedding(Recorder(httpx.Response(500, text="EMBED-BODY-CANARY")), api_key=key)
        with pytest.raises(AppError):
            await emb.embed_text("DOC-TEXT-CANARY")
    for canary in (key, "SYSTEM-CANARY", "PROMPT-CANARY", "ANSWER-CANARY", "EMBED-BODY-CANARY", "DOC-TEXT-CANARY"):
        assert canary not in caplog.text, canary
    await svc.close()


# ============================================================================ multi-key rotation & failover

async def test_multi_key_round_robin_across_generate_and_embeddings():
    rec = Recorder(gemini_ok("1"), gemini_ok("2"), gemini_ok("3"), gemini_ok("4"))
    svc = gemini_llm(rec, api_key="key-a, key-b; key-c, key-a")  # duplicate key-a is deduplicated
    for expected in ("1", "2", "3", "4"):
        assert await svc.generate("s", "u") == expected
    used_keys = [r.headers["x-goog-api-key"] for r in rec.requests]
    assert used_keys == ["key-a", "key-b", "key-c", "key-a"]
    await svc.close()

    emb_rec = Recorder(httpx.Response(200, json=gemini_embeddings_body([[1.0]])))
    emb = gemini_embedding(emb_rec, api_key="emb-1,emb-2")
    await emb.embed_text("x")
    await emb.embed_text("y")
    await emb.embed_text("z")
    assert [r.headers["x-goog-api-key"] for r in emb_rec.requests] == ["emb-1", "emb-2", "emb-1"]
    await emb.close()


async def test_multi_key_failover_on_429_skips_cooled_down_key_in_subsequent_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["x-goog-api-key"]
        if key == "key-exhausted":
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})
        return gemini_ok(f"ok-from-{key}")

    svc = gemini_llm(handler, api_key="key-exhausted,key-healthy-1,key-healthy-2", backoff=5.0)
    svc._max_retries = 0
    # Call 1: hits key-exhausted (429), immediately fails over to key-healthy-1 without waiting backoff
    started = time.perf_counter()
    assert await svc.generate("s", "u") == "ok-from-key-healthy-1"
    assert time.perf_counter() - started < 1.0
    # Call 2 & Call 3: key-exhausted is in 60s cooldown, so round-robin rotates only healthy keys
    assert await svc.generate("s", "u") == "ok-from-key-healthy-2"
    assert await svc.generate("s", "u") == "ok-from-key-healthy-1"
    await svc.close()


async def test_multi_key_stream_and_health_failover_on_429():
    rec = Recorder(
        httpx.Response(429, text="quota exceeded"),
        httpx.Response(200, content=gemini_sse("stream", "-ok")),
    )
    svc = gemini_llm(rec, api_key="k1,k2")
    svc._max_retries = 0
    assert [p async for p in svc.generate_stream("s", "u")] == ["stream", "-ok"]
    assert [r.headers["x-goog-api-key"] for r in rec.requests] == ["k1", "k2"]
    await svc.close()

    health_rec = Recorder(httpx.Response(429), httpx.Response(200, json={"name": "models/gemini-test"}))
    svc2 = gemini_llm(health_rec, api_key="k1,k2")
    assert await svc2.check_health() == {"status": "ok", "model": "gemini-test"}
    assert [r.headers["x-goog-api-key"] for r in health_rec.requests] == ["k1", "k2"]
    await svc2.close()


