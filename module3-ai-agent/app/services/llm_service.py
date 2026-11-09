"""LLM service abstraction + the Google Gemini (AI Studio) implementation.

Architecture
~~~~~~~~~~~~
RAGService / QuestionGenerationService / EvaluationService → LLMService (abstract) → GeminiLLMService

Services depend only on :class:`LLMService`, so tests can inject fakes without touching business logic.
Model name, base URL, API key, timeouts, temperature and max output tokens all come from ``Settings``.

Error codes keep their historical ``OLLAMA_*`` string values (see ``ErrorCode.LLM_*``) for API compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import aclosing
from typing import Any, AsyncIterator, Optional

import httpx

from app.core import runtime_config
from app.core.concurrency import llm_gate
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.key_pool import GeminiKeyPool
from app.core.metrics import metrics
from app.core.output_guard import enforce_output_size
from app.core.prompts import JSON_RETRY_PROMPT
from app.core.timeouts import call_timeout, can_retry
from app.utils.json_utils import safe_parse_json

logger = logging.getLogger(__name__)

# Upstream statuses considered transient and worth a retry (429 = Gemini quota / rate limit).
# Never retried: 404 (model missing), other 4xx, transport errors, timeouts, invalid JSON from the model.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Connection dropped mid-exchange (server restart / reset). Safe to repeat once for the non-streaming call;
# never for ConnectError (provider unreachable) or timeouts (already consumed the full timeout).
CONNECTION_RESET = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)

_KEY_MISSING = "Gemini API key is not configured."
_BAD_FORMAT = "Language model returned an unexpected response format."


def _llm_outcome(exc: AppError) -> str:
    if exc.code == ErrorCode.SERVICE_OVERLOADED:
        return "rejected"
    if exc.code == ErrorCode.LLM_TIMEOUT:
        return "timeout"
    return "error"


def validate_required_keys(parsed: Any, required_keys: Optional[list[str]]) -> None:
    """Raise ``ValueError`` when a JSON object lacks one of ``required_keys``."""
    if not required_keys:
        return
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


class LLMService(ABC):
    """Provider-agnostic generative-LLM interface."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the model's plain-text answer."""

    @abstractmethod
    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        required_keys: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Return parsed JSON from the model (one retry if invalid)."""

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield the model output incrementally as raw text pieces."""
        raise NotImplementedError("This LLM provider does not support streaming.")
        yield ""  # pragma: no cover - makes this an async generator

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Verify provider reachability and model availability (raises AppError)."""

    async def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources."""


class GeminiLLMService(LLMService):
    """LLMService backed by the Google AI Studio Gemini REST API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        s = get_settings()
        raw_key = api_key if api_key is not None else s.gemini_api_key
        self._key_pool = GeminiKeyPool(raw_key)
        self._api_key = self._key_pool.keys[0] if self._key_pool else ""
        self._base_url = (base_url or s.gemini_base_url).rstrip("/")
        self._model_explicit = model
        self._timeout = timeout or s.llm_timeout
        self._temperature_explicit = temperature
        self._max_tokens_explicit = max_output_tokens
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = max(0, s.llm_max_retries)
        self._retry_backoff_s = max(0.0, s.llm_retry_backoff_s)

    # ── Effective parameters (explicit constructor values win over settings) ──

    @property
    def _model(self) -> str:
        return self._model_explicit or runtime_config.current().llm_model

    @property
    def _model_id(self) -> str:
        return self._model.removeprefix("models/")

    @property
    def _temperature(self) -> float:
        explicit = self._temperature_explicit
        return runtime_config.current().temperature if explicit is None else explicit

    @property
    def _max_tokens(self) -> int:
        return self._max_tokens_explicit or runtime_config.current().num_predict

    @property
    def model_name(self) -> str:
        return self._model

    # ── Client lifecycle ────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # The key travels only in this header - never in a URL (URLs end up in logs).
            headers = {"x-goog-api-key": self._api_key} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().llm_connect_timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── Health ──────────────────────────────────────────────────────

    async def check_health(self) -> dict[str, Any]:
        if not self._key_pool:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, _KEY_MISSING)
        client = await self._get_client()
        last_status = 0
        for key in self._key_pool.keys:
            try:
                resp = await client.get(
                    f"/models/{self._model_id}",
                    headers={"x-goog-api-key": key},
                    timeout=get_settings().llm_health_timeout,
                )
            except httpx.TimeoutException as exc:
                raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API did not respond.") from exc
            except httpx.HTTPError as exc:
                raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API is not reachable.") from exc
            self._key_pool.report_status(key, resp.status_code)
            if resp.status_code == 200:
                return {"status": "ok", "model": self._model}
            if resp.status_code == 404:
                raise AppError(ErrorCode.LLM_MODEL_NOT_FOUND, f"Model '{self._model}' is not available in Gemini API.")
            last_status = resp.status_code
            if resp.status_code not in RETRYABLE_STATUS:
                break
        raise AppError(ErrorCode.LLM_UNAVAILABLE, f"Gemini API returned HTTP {last_status}.")

    # ── Generation ──────────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await self._chat(
            system_prompt, user_prompt, json_mode=False, temperature=temperature, max_tokens=max_tokens,
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        required_keys: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Ask for JSON; on invalid output retry exactly once, then LLM_INVALID_JSON."""
        prompt = user_prompt
        for attempt in (1, 2):
            raw = await self._chat(
                system_prompt, prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens,
            )
            enforce_output_size(raw)  # oversized output is rejected, not parsed
            parse_started = time.perf_counter()
            try:
                parsed = safe_parse_json(raw)
                validate_required_keys(parsed, required_keys)
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                return parsed
            except ValueError as exc:
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                # Log the exception type/length only - never raw model output.
                logger.warning(
                    "LLM returned invalid JSON (attempt %d/2, model=%s, error=%s, raw_chars=%d)",
                    attempt, self._model, type(exc).__name__, len(raw),
                )
                prompt = f"{user_prompt}\n\n{JSON_RETRY_PROMPT}"
        raise AppError(ErrorCode.LLM_INVALID_JSON, "The language model did not return valid structured output.")

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        gen_cfg: dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature,
            "maxOutputTokens": max_tokens or self._max_tokens,
        }
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    @staticmethod
    def _extract_text(obj: Any, *, allow_empty_candidates: bool = False) -> str:
        """Concatenate the text parts of the first candidate (``thought`` parts are skipped)."""
        if not isinstance(obj, dict):
            raise ValueError("Expected JSON object")
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            if allow_empty_candidates:
                return ""
            raise ValueError("Missing candidates")
        first = candidates[0]
        if not isinstance(first, dict):
            raise ValueError("Invalid candidate")
        content = first.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            if allow_empty_candidates:
                return ""
            raise ValueError("Missing content parts")
        return "".join(
            p["text"] for p in parts
            if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str)
        )

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Gated stream: holds one LLM concurrency slot for the whole stream (released on close)."""
        started = time.perf_counter()
        first = True
        try:
            async with llm_gate.slot():
                inner = self._generate_stream_unlocked(
                    system_prompt, user_prompt, json_mode=json_mode,
                    temperature=temperature, max_tokens=max_tokens,
                )
                async with aclosing(inner):
                    async for piece in inner:
                        if first:
                            first = False
                            metrics.observe_stage("llm_first_token", time.perf_counter() - started)
                        yield piece
        except AppError as exc:
            metrics.event("llm_request", _llm_outcome(exc))
            raise
        metrics.observe_stage("llm_stream_total", time.perf_counter() - started)
        metrics.event("llm_request", "ok")

    async def _generate_stream_unlocked(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream ``streamGenerateContent?alt=sse`` and yield each text piece.

        Raw provider objects never leave this method. The httpx read timeout applies to every read, so a
        stalled stream raises LLM_TIMEOUT instead of hanging. Retries only happen before any byte was read.
        """
        if not self._key_pool:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, _KEY_MISSING)
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens,
        )
        endpoint = f"/models/{self._model_id}:streamGenerateContent?alt=sse"
        client = await self._get_client()
        call_timeout(self._timeout, get_settings().llm_connect_timeout)  # refuse when the budget is spent
        started = time.perf_counter()
        pieces = 0
        has_text = False
        retry_count = 0
        tried_keys: set[str] = set()
        max_retries = max(self._max_retries, self._key_pool.size - 1)
        try:
            while True:
                retry = False
                key = self._key_pool.next_key(exclude=tried_keys)
                tried_keys.add(key)
                async with client.stream("POST", endpoint, json=payload, headers={"x-goog-api-key": key}) as resp:
                    self._key_pool.report_status(key, resp.status_code)
                    if resp.status_code == 404:
                        raise AppError(
                            ErrorCode.LLM_MODEL_NOT_FOUND, f"Model '{self._model}' is not available in Gemini API.",
                        )
                    if resp.status_code != 200:
                        logger.error(
                            "Gemini stream status=%d key_slot=%s", resp.status_code, self._key_pool.key_slot(key),
                        )
                        if resp.status_code in RETRYABLE_STATUS and retry_count < max_retries:
                            retry = True
                        else:
                            raise AppError(
                                ErrorCode.LLM_GENERATION_ERROR, f"Language model returned HTTP {resp.status_code}.",
                            )
                    if not retry:
                        async for line in resp.aiter_lines():
                            data = line.strip()
                            if data.startswith("data:"):
                                data = data[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                content = self._extract_text(json.loads(data), allow_empty_candidates=True)
                            except ValueError as exc:
                                raise AppError(ErrorCode.LLM_INVALID_RESPONSE, _BAD_FORMAT) from exc
                            if content:
                                pieces += 1
                                has_text = has_text or bool(content.strip())
                                yield content
                if not retry:
                    break
                retry_count += 1
                has_untried_key = self._key_pool.size > 1 and len(tried_keys) < self._key_pool.size
                logger.warning(
                    "Gemini stream transient failure - retrying retry_count=%d model=%s key_slot=%s",
                    retry_count, self._model, self._key_pool.key_slot(key),
                )
                if not has_untried_key:
                    await asyncio.sleep(self._retry_backoff_s)
        except httpx.ConnectError as exc:
            logger.error("Gemini unreachable: %s", exc)
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API is not reachable.") from exc
        except httpx.TimeoutException as exc:
            logger.error("Gemini stream timed out: %s", exc)
            raise AppError(ErrorCode.LLM_TIMEOUT, "Language model request timed out.") from exc
        except httpx.HTTPError as exc:
            logger.error("Gemini HTTP error during stream: %s", exc)
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API returned an HTTP error.") from exc

        if not has_text:
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty response.")
        logger.info(
            "LLM stream | model=%s json_mode=%s pieces=%d retry_count=%d time_ms=%.1f",
            self._model, json_mode, pieces, retry_count, (time.perf_counter() - started) * 1000,
        )

    async def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Gated, timed wrapper around :meth:`_chat_unlocked`."""
        outcome: Optional[str] = None
        try:
            async with llm_gate.slot():
                started = time.perf_counter()
                try:
                    result = await self._chat_unlocked(
                        system_prompt, user_prompt, json_mode=json_mode,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    outcome = "ok"
                    return result
                finally:
                    metrics.observe_stage("llm_generate", time.perf_counter() - started)
        except AppError as exc:
            outcome = _llm_outcome(exc)
            raise
        finally:
            if outcome is not None:  # cancellation records nothing
                metrics.event("llm_request", outcome)

    async def _chat_unlocked(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        if not self._key_pool:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, _KEY_MISSING)
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens,
        )
        endpoint = f"/models/{self._model_id}:generateContent"
        client = await self._get_client()
        started = time.perf_counter()
        retry_count = 0
        tried_keys: set[str] = set()
        max_retries = max(self._max_retries, self._key_pool.size - 1)
        while True:
            timeout = call_timeout(self._timeout, get_settings().llm_connect_timeout)
            kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
            key = self._key_pool.next_key(exclude=tried_keys)
            tried_keys.add(key)
            try:
                resp = await client.post(endpoint, json=payload, headers={"x-goog-api-key": key}, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Gemini unreachable: %s", exc)
                raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API is not reachable.") from exc
            except httpx.TimeoutException as exc:
                # Not retried: a timed-out call already used its whole budget; retrying would double the wait.
                logger.error("Gemini generation timed out: %s", exc)
                raise AppError(ErrorCode.LLM_TIMEOUT, "Language model request timed out.") from exc
            except CONNECTION_RESET as exc:
                if retry_count < self._max_retries and can_retry(self._retry_backoff_s, 5.0):
                    retry_count += 1
                    metrics.event("llm_retry", "connection_reset")
                    logger.warning("Gemini connection reset - retrying retry_count=%d model=%s", retry_count, self._model)
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                logger.error("Gemini connection reset: %s", exc)
                raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API dropped the connection.") from exc
            except httpx.HTTPError as exc:
                logger.error("Gemini HTTP error: %s", exc)
                raise AppError(ErrorCode.LLM_UNAVAILABLE, "Gemini API returned an HTTP error.") from exc

            self._key_pool.report_status(key, resp.status_code)
            has_untried_key = self._key_pool.size > 1 and len(tried_keys) < self._key_pool.size
            backoff = 0.0 if has_untried_key else self._retry_backoff_s
            if (resp.status_code in RETRYABLE_STATUS and retry_count < max_retries
                    and can_retry(backoff, 5.0)):
                retry_count += 1
                metrics.event("llm_retry", "transient")
                logger.warning(
                    "Gemini transient failure status=%d - retrying retry_count=%d model=%s key_slot=%s",
                    resp.status_code, retry_count, self._model, self._key_pool.key_slot(key),
                )
                if backoff > 0:
                    await asyncio.sleep(backoff)
                continue
            break

        elapsed_ms = (time.perf_counter() - started) * 1000
        if resp.status_code == 404:
            raise AppError(ErrorCode.LLM_MODEL_NOT_FOUND, f"Model '{self._model}' is not available in Gemini API.")
        if resp.status_code != 200:
            # The upstream body is not logged: it may echo prompt content.
            logger.error("Gemini generate status=%d", resp.status_code)
            raise AppError(ErrorCode.LLM_GENERATION_ERROR, f"Language model returned HTTP {resp.status_code}.")
        try:
            content = self._extract_text(resp.json())
        except Exception as exc:
            raise AppError(ErrorCode.LLM_INVALID_RESPONSE, _BAD_FORMAT) from exc
        if not content.strip():
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty response.")

        logger.info(
            "LLM generation | model=%s json_mode=%s prompt_chars=%d response_length=%d "
            "retry_count=%d llm_time_ms=%.1f",
            self._model, json_mode, len(system_prompt) + len(user_prompt), len(content), retry_count, elapsed_ms,
        )
        return content


# ── Module-level singleton ──────────────────────────────────────────

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return the shared Gemini LLMService (a missing API key surfaces as LLM_UNAVAILABLE per call)."""
    global _llm_service
    if _llm_service is None:
        _llm_service = GeminiLLMService()
    return _llm_service
