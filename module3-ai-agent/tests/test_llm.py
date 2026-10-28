"""Phase 5 — LLM service tests.  Ollama is always mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.prompts import (
    PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    build_rag_messages,
)
from app.services.llm_service import LLMService, OllamaLLMService
from app.utils.json_utils import safe_parse_json


# ── Helpers ─────────────────────────────────────────────────────────


def chat_response(content, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json={"message": {"role": "assistant", "content": content}},
        request=httpx.Request("POST", "http://fake/api/chat"),
    )


def raw_response(status: int, text: str = "") -> httpx.Response:
    return httpx.Response(
        status_code=status, text=text, request=httpx.Request("POST", "http://fake/api/chat")
    )


def make_service() -> OllamaLLMService:
    return OllamaLLMService(
        base_url="http://fake-ollama:11434",
        model="test-model:1b",
        timeout=10,
        temperature=0.1,
        num_predict=64,
    )


def mock_client(service: OllamaLLMService, *, post=None, get=None):
    client = AsyncMock()
    if post is not None:
        client.post = post if isinstance(post, AsyncMock) else AsyncMock(**post)
    if get is not None:
        client.get = get if isinstance(get, AsyncMock) else AsyncMock(**get)
    return patch.object(service, "_get_client", AsyncMock(return_value=client)), client


# ── generate ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_generation_and_request_shape() -> None:
    svc = make_service()
    ctx, client = mock_client(svc, post={"return_value": chat_response("Hello there")})
    with ctx:
        out = await svc.generate("SYS", "USER")
    assert out == "Hello there"

    args, kwargs = client.post.call_args
    assert args[0] == "/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "test-model:1b"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert payload["options"] == {"temperature": 0.1, "num_predict": 64}
    assert "format" not in payload


@pytest.mark.asyncio
async def test_per_call_overrides() -> None:
    svc = make_service()
    ctx, client = mock_client(svc, post={"return_value": chat_response("x")})
    with ctx:
        await svc.generate("s", "u", temperature=0.9, max_tokens=5)
    assert client.post.call_args.kwargs["json"]["options"] == {"temperature": 0.9, "num_predict": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   \n"])
async def test_empty_response(content: str) -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"return_value": chat_response(content)})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.LLM_EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_invalid_response_format() -> None:
    svc = make_service()
    bad = httpx.Response(
        200, json={"unexpected": 1}, request=httpx.Request("POST", "http://fake/api/chat")
    )
    ctx, _ = mock_client(svc, post={"return_value": bad})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_INVALID_RESPONSE


@pytest.mark.asyncio
async def test_non_json_body() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"return_value": raw_response(200, "not json")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_INVALID_RESPONSE


@pytest.mark.asyncio
async def test_ollama_unavailable() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"side_effect": httpx.ConnectError("refused")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE


@pytest.mark.asyncio
async def test_timeout() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"side_effect": httpx.ReadTimeout("slow")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_TIMEOUT


@pytest.mark.asyncio
async def test_model_not_found() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"return_value": raw_response(404, "model not found")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_MODEL_NOT_FOUND


@pytest.mark.asyncio
async def test_server_error_is_generation_error_without_leaking_body() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, post={"return_value": raw_response(500, "Traceback secret stack")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_GENERATION_ERROR
    assert "Traceback" not in exc.value.message


# ── JSON generation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_json_generation() -> None:
    svc = make_service()
    ctx, client = mock_client(svc, post={"return_value": chat_response('{"answer": "42"}')})
    with ctx:
        out = await svc.generate_json("s", "u", required_keys=["answer"])
    assert out == {"answer": "42"}
    assert client.post.call_args.kwargs["json"]["format"] == "json"
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_json_markdown_fence_is_stripped() -> None:
    svc = make_service()
    fenced = '```json\n{"answer": "fenced"}\n```'
    ctx, client = mock_client(svc, post={"return_value": chat_response(fenced)})
    with ctx:
        out = await svc.generate_json("s", "u")
    assert out == {"answer": "fenced"}
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_invalid_json_fails_after_exactly_one_retry() -> None:
    svc = make_service()
    ctx, client = mock_client(svc, post={"return_value": chat_response("this is not json")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate_json("s", "u")
    assert exc.value.code == ErrorCode.LLM_INVALID_JSON
    assert client.post.await_count == 2  # initial + ONE retry, never more


@pytest.mark.asyncio
async def test_json_retry_succeeds_and_prompt_is_strengthened() -> None:
    svc = make_service()
    post = AsyncMock(side_effect=[chat_response("garbage"), chat_response('{"answer": "ok"}')])
    ctx, client = mock_client(svc, post=post)
    with ctx:
        out = await svc.generate_json("s", "u", required_keys=["answer"])
    assert out == {"answer": "ok"}
    assert post.await_count == 2
    first_user = post.await_args_list[0].kwargs["json"]["messages"][1]["content"]
    retry_user = post.await_args_list[1].kwargs["json"]["messages"][1]["content"]
    assert first_user == "u"
    assert "Return ONLY valid JSON." in retry_user
    assert "Do not include Markdown." in retry_user


@pytest.mark.asyncio
async def test_missing_required_key_triggers_retry() -> None:
    svc = make_service()
    post = AsyncMock(
        side_effect=[chat_response('{"foo": 1}'), chat_response('{"answer": "fixed"}')]
    )
    ctx, _ = mock_client(svc, post=post)
    with ctx:
        out = await svc.generate_json("s", "u", required_keys=["answer"])
    assert out["answer"] == "fixed"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_json_does_not_retry_on_transport_errors() -> None:
    svc = make_service()
    post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    ctx, _ = mock_client(svc, post=post)
    with ctx, pytest.raises(AppError) as exc:
        await svc.generate_json("s", "u")
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE
    assert post.await_count == 1


# ── Health ──────────────────────────────────────────────────────────


def tags_response(names: list[str], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"models": [{"name": n} for n in names]},
        request=httpx.Request("GET", "http://fake/api/tags"),
    )


@pytest.mark.asyncio
async def test_health_ok() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, get={"return_value": tags_response(["test-model:1b", "other"])})
    with ctx:
        info = await svc.check_health()
    assert info["status"] == "ok" and info["model"] == "test-model:1b"


@pytest.mark.asyncio
async def test_health_model_missing() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, get={"return_value": tags_response(["something-else:7b"])})
    with ctx, pytest.raises(AppError) as exc:
        await svc.check_health()
    assert exc.value.code == ErrorCode.OLLAMA_MODEL_NOT_FOUND


@pytest.mark.asyncio
async def test_health_server_down() -> None:
    svc = make_service()
    ctx, _ = mock_client(svc, get={"side_effect": httpx.ConnectError("refused")})
    with ctx, pytest.raises(AppError) as exc:
        await svc.check_health()
    assert exc.value.code == ErrorCode.OLLAMA_UNAVAILABLE


@pytest.mark.asyncio
async def test_health_untagged_model_matches_latest() -> None:
    svc = OllamaLLMService(base_url="http://x", model="plainmodel", timeout=5)
    ctx, _ = mock_client(svc, get={"return_value": tags_response(["plainmodel:latest"])})
    with ctx:
        assert (await svc.check_health())["status"] == "ok"


# ── Configuration / abstraction ─────────────────────────────────────


def test_configuration_comes_from_settings() -> None:
    s = get_settings()
    svc = OllamaLLMService()
    assert svc.model_name == s.ollama_model
    assert svc._temperature == s.ollama_temperature
    assert svc._num_predict == s.ollama_num_predict
    assert svc._timeout == s.ollama_timeout


def test_model_name_not_hardcoded_in_source() -> None:
    import app.services.llm_service as mod
    import app.services.rag_service as rag_mod

    for m in (mod, rag_mod):
        assert "qwen2.5:7b" not in open(m.__file__, encoding="utf-8").read()


def test_ollama_service_implements_interface() -> None:
    assert issubclass(OllamaLLMService, LLMService)
    assert not hasattr(OllamaLLMService, "stream")  # streaming is a later phase
    with pytest.raises(TypeError):
        LLMService()  # type: ignore[abstract]


# ── Prompts ─────────────────────────────────────────────────────────


def test_prompt_version_and_rules() -> None:
    assert PROMPT_VERSION == "v1"
    for rule in (
        "ONLY the provided reference context",
        "Do not use external knowledge",
        "Do not invent facts",
        "Do not invent source pages",
        "Do not invent chunk IDs",
        "Do not invent similarity scores",
        "conflict",
    ):
        assert rule in RAG_SYSTEM_PROMPT


def test_prompt_contains_question_and_context() -> None:
    system, user = build_rag_messages("CTX BODY", "THE QUESTION?")
    assert system == RAG_SYSTEM_PROMPT
    assert "CTX BODY" in user and "THE QUESTION?" in user
    assert "REFERENCE CONTEXT" in user and "USER QUESTION" in user


def test_prompt_survives_special_characters() -> None:
    nasty = 'He said "hi" {x} {0} {{y}} %s \\n\nline2 `code` ```json'
    _, user = build_rag_messages(nasty, nasty)
    assert user.count(nasty) == 2


def test_prompts_contain_no_secrets() -> None:
    _, user = build_rag_messages("c", "q")
    blob = (RAG_SYSTEM_PROMPT + user).lower()
    for word in ("api_key", "api key", "secret", "password", "token"):
        assert word not in blob


# ── json_utils ──────────────────────────────────────────────────────


def test_json_utils_variants() -> None:
    assert safe_parse_json('{"a": 1}') == {"a": 1}
    assert safe_parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert safe_parse_json('```\n{"a": 1}\n```') == {"a": 1}
    assert safe_parse_json('Sure! {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(ValueError):
        safe_parse_json("no json here")
    with pytest.raises(ValueError):
        safe_parse_json("{broken: json}")
