"""Phase 5 — LLM service abstraction + Gemini (Google AI Studio) & Ollama implementations.

Architecture
~~~~~~~~~~~~
RAGService → LLMService (abstract) → GeminiLLMService / OllamaLLMService

RAGService depends only on :class:`LLMService`, so providers can be swapped
without touching RAG logic.

Model name, base URL, API key, timeout, temperature and num_predict all come from
``Settings`` — nothing is hard-coded here.
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

from app.core.concurrency import llm_gate
from app.core import runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.prompts import JSON_RETRY_PROMPT
from app.core.output_guard import enforce_output_size
from app.core.timeouts import call_timeout, can_retry
from app.utils.json_utils import safe_parse_json

logger = logging.getLogger(__name__)

# Only these upstream statuses are considered transient and worth ONE retry.
# Never retried: 404 (model missing), 400/other 4xx, 500, transport errors,
# timeouts, invalid JSON from the model.
_RETRYABLE_STATUS = frozenset({502, 503, 504})
# Connection dropped mid-exchange (server restart / reset). Transient and safe to repeat once for the
# non-streaming call; never for ConnectError (refused = Ollama is down, a retry in 0.5 s cannot help) or
# timeouts (already consumed the full timeout).
_CONNECTION_RESET = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)


def _llm_outcome(exc: AppError) -> str:
    if exc.code == ErrorCode.SERVICE_OVERLOADED:
        return "rejected"
    if exc.code == ErrorCode.OLLAMA_TIMEOUT:
        return "timeout"
    return "error"


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
        """Yield the model output incrementally as raw text pieces (Phase 8).

        Not abstract so existing providers/subclasses keep working; providers that
        support streaming override it.
        """
        raise NotImplementedError("This LLM provider does not support streaming.")
        yield ""  # pragma: no cover - makes this an async generator

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Verify provider reachability and model availability (raises AppError)."""

    async def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources."""


class OllamaLLMService(LLMService):
    """LLMService backed by a local Ollama server (e.g. Qwen)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> None:
        s = get_settings()
        self._base_url = (base_url or s.ollama_base_url).rstrip("/")
        # Phase 18: unset values follow the effective (active AI version) configuration per call.
        self._model_explicit = model
        self._timeout = timeout or s.ollama_timeout
        self._temperature_explicit = temperature
        self._num_predict_explicit = num_predict
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = max(0, s.ollama_max_retries)
        self._retry_backoff_s = max(0.0, s.ollama_retry_backoff_s)

    @property
    def _model(self) -> str:
        return self._model_explicit or runtime_config.current().llm_model

    @property
    def _temperature(self) -> float:
        explicit = self._temperature_explicit
        return runtime_config.current().temperature if explicit is None else explicit

    @property
    def _num_predict(self) -> int:
        return self._num_predict_explicit or runtime_config.current().num_predict

    @property
    def model_name(self) -> str:
        return self._model

    # ── Client lifecycle ────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().ollama_connect_timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── Health ──────────────────────────────────────────────────────

    async def check_health(self) -> dict[str, Any]:
        client = await self._get_client()
        try:
            resp = await client.get("/api/tags", timeout=get_settings().ollama_health_timeout)
        except httpx.TimeoutException as exc:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server did not respond.") from exc
        except httpx.HTTPError as exc:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.") from exc

        if resp.status_code != 200:
            raise AppError(
                ErrorCode.OLLAMA_UNAVAILABLE,
                f"Ollama server returned HTTP {resp.status_code}.",
            )
        try:
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception as exc:
            raise AppError(
                ErrorCode.OLLAMA_INVALID_RESPONSE, "Ollama returned an invalid model list."
            ) from exc

        if not self._model_available(names):
            raise AppError(
                ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                f"Model '{self._model}' is not available. Pull it first with: ollama pull {self._model}",
            )
        return {"status": "ok", "model": self._model, "available_models": names}

    def _model_available(self, names: list[str]) -> bool:
        if self._model in names:
            return True
        # "name" without a tag resolves to "name:latest" in Ollama.
        return ":" not in self._model and f"{self._model}:latest" in names

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
            system_prompt, user_prompt, json_mode=False,
            temperature=temperature, max_tokens=max_tokens,
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
                system_prompt, prompt, json_mode=True,
                temperature=temperature, max_tokens=max_tokens,
            )
            enforce_output_size(raw)  # Phase 19: oversized output is rejected, not parsed
            parse_started = time.perf_counter()
            try:
                parsed = safe_parse_json(raw)
                self._validate_keys(parsed, required_keys)
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                return parsed
            except ValueError as exc:
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                # Log the exception type/length only — never raw model output.
                logger.warning(
                    "LLM returned invalid JSON (attempt %d/2, model=%s, error=%s, raw_chars=%d)",
                    attempt, self._model, type(exc).__name__, len(raw),
                )
                prompt = f"{user_prompt}\n\n{JSON_RETRY_PROMPT}"

        raise AppError(
            ErrorCode.LLM_INVALID_JSON,
            "The language model did not return valid structured output.",
        )

    @staticmethod
    def _validate_keys(parsed: Any, required_keys: Optional[list[str]]) -> None:
        if not required_keys:
            return
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": max_tokens or self._num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

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
        """Stream ``/api/chat`` (NDJSON) and yield each content piece unchanged.

        Raw Ollama objects never leave this method.  The httpx timeout (OLLAMA_TIMEOUT)
        applies to every read, so a stalled stream raises OLLAMA_TIMEOUT instead of hanging.
        """
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode,
            temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        client = await self._get_client()
        call_timeout(self._timeout, get_settings().ollama_connect_timeout)  # refuse when the budget is spent
        started = time.perf_counter()
        pieces = 0
        has_text = False
        retry_count = 0
        try:
            while True:
                retry = False
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code == 404:
                        raise AppError(
                            ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                            f"Model '{self._model}' is not available. Pull it first with: ollama pull {self._model}",
                        )
                    if resp.status_code != 200:
                        logger.error("Ollama stream status=%d", resp.status_code)
                        if resp.status_code in _RETRYABLE_STATUS and retry_count < self._max_retries:
                            retry = True
                        else:
                            raise AppError(
                                ErrorCode.OLLAMA_GENERATION_ERROR,
                                f"Language model returned HTTP {resp.status_code}.",
                            )
                    if not retry:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except ValueError as exc:
                                raise AppError(
                                    ErrorCode.OLLAMA_INVALID_RESPONSE,
                                    "Language model returned an unexpected response format.",
                                ) from exc
                            if not isinstance(obj, dict):
                                raise AppError(
                                    ErrorCode.OLLAMA_INVALID_RESPONSE,
                                    "Language model returned an unexpected response format.",
                                )
                            if obj.get("error"):
                                logger.error("Ollama stream error: %s", str(obj["error"])[:300])
                                raise AppError(
                                    ErrorCode.OLLAMA_GENERATION_ERROR, "Language model reported an error."
                                )
                            message = obj.get("message")
                            content = message.get("content", "") if isinstance(message, dict) else ""
                            if not isinstance(content, str):
                                raise AppError(
                                    ErrorCode.OLLAMA_INVALID_RESPONSE,
                                    "Language model returned an unexpected response format.",
                                )
                            if content:
                                pieces += 1
                                has_text = has_text or bool(content.strip())
                                yield content
                            if obj.get("done"):
                                break
                if not retry:
                    break
                retry_count += 1
                logger.warning(
                    "Ollama stream transient failure — retrying retry_count=%d model=%s",
                    retry_count, self._model,
                )
                await asyncio.sleep(self._retry_backoff_s)
        except httpx.ConnectError as exc:
            logger.error("Ollama unreachable: %s", exc)
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.") from exc
        except httpx.TimeoutException as exc:
            logger.error("Ollama stream timed out: %s", exc)
            raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.") from exc
        except httpx.HTTPError as exc:
            logger.error("Ollama HTTP error during stream: %s", exc)
            raise AppError(
                ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server returned an HTTP error."
            ) from exc

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
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode,
            temperature=temperature, max_tokens=max_tokens,
        )

        client = await self._get_client()
        started = time.perf_counter()
        retry_count = 0
        while True:
            timeout = call_timeout(self._timeout, get_settings().ollama_connect_timeout)
            kwargs = {} if timeout is None else {"timeout": timeout}
            try:
                resp = await client.post("/api/chat", json=payload, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Ollama unreachable: %s", exc)
                raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.") from exc
            except httpx.TimeoutException as exc:
                # Not retried: a timed-out call already used its whole budget; retrying would double the wait.
                logger.error("Ollama generation timed out: %s", exc)
                raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.") from exc
            except _CONNECTION_RESET as exc:
                if retry_count < self._max_retries and can_retry(self._retry_backoff_s, 5.0):
                    retry_count += 1
                    metrics.event("llm_retry", "connection_reset")
                    logger.warning("Ollama connection reset - retrying retry_count=%d model=%s", retry_count, self._model)
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                logger.error("Ollama connection reset: %s", exc)
                raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server dropped the connection.") from exc
            except httpx.HTTPError as exc:
                logger.error("Ollama HTTP error: %s", exc)
                raise AppError(
                    ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server returned an HTTP error."
                ) from exc

            if (resp.status_code in _RETRYABLE_STATUS and retry_count < self._max_retries
                    and can_retry(self._retry_backoff_s, 5.0)):
                retry_count += 1
                metrics.event("llm_retry", "transient")
                logger.warning(
                    "Ollama transient failure status=%d — retrying retry_count=%d model=%s",
                    resp.status_code, retry_count, self._model,
                )
                await asyncio.sleep(self._retry_backoff_s)
                continue
            break

        elapsed_ms = (time.perf_counter() - started) * 1000

        if resp.status_code == 404:
            raise AppError(
                ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                f"Model '{self._model}' is not available. Pull it first with: ollama pull {self._model}",
            )
        if resp.status_code != 200:
            logger.error("Ollama chat status=%d body=%s", resp.status_code, resp.text[:300])
            raise AppError(
                ErrorCode.OLLAMA_GENERATION_ERROR,
                f"Language model returned HTTP {resp.status_code}.",
            )

        try:
            content = resp.json()["message"]["content"]
        except Exception as exc:
            raise AppError(
                ErrorCode.OLLAMA_INVALID_RESPONSE,
                "Language model returned an unexpected response format.",
            ) from exc
        if not isinstance(content, str):
            raise AppError(
                ErrorCode.OLLAMA_INVALID_RESPONSE,
                "Language model returned an unexpected response format.",
            )

        if not content.strip():
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty response.")

        logger.info(
            "LLM generation | model=%s json_mode=%s prompt_chars=%d response_length=%d "
            "retry_count=%d llm_time_ms=%.1f",
            self._model, json_mode, len(system_prompt) + len(user_prompt), len(content),
            retry_count, elapsed_ms,
        )
        return content


class GeminiLLMService(LLMService):
    """LLMService backed by Google AI Studio (Gemini REST API)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> None:
        s = get_settings()
        self._api_key = (api_key if api_key is not None else s.gemini_api_key).strip()
        self._base_url = (base_url or s.gemini_base_url).rstrip("/")
        self._model_explicit = model
        self._timeout = timeout or s.ollama_timeout
        self._temperature_explicit = temperature
        self._num_predict_explicit = num_predict
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = max(0, s.ollama_max_retries)
        self._retry_backoff_s = max(0.0, s.ollama_retry_backoff_s)

    @property
    def _model(self) -> str:
        if self._model_explicit:
            return self._model_explicit
        s = get_settings()
        eff = runtime_config.current()
        if eff.source != "env" and eff.llm_model != s.ollama_model:
            return eff.llm_model
        return s.gemini_model

    @property
    def _temperature(self) -> float:
        explicit = self._temperature_explicit
        return runtime_config.current().temperature if explicit is None else explicit

    @property
    def _num_predict(self) -> int:
        return self._num_predict_explicit or runtime_config.current().num_predict

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"x-goog-api-key": self._api_key} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().ollama_connect_timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> dict[str, Any]:
        if not self._api_key:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API key is not configured.")
        client = await self._get_client()
        model_id = self._model.removeprefix("models/")
        try:
            resp = await client.get(f"/models/{model_id}", timeout=get_settings().ollama_health_timeout)
        except httpx.TimeoutException as exc:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API did not respond.") from exc
        except httpx.HTTPError as exc:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API is not reachable.") from exc

        if resp.status_code == 404:
            raise AppError(
                ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                f"Model '{self._model}' is not available in Gemini API.",
            )
        if resp.status_code != 200:
            raise AppError(
                ErrorCode.OLLAMA_UNAVAILABLE,
                f"Gemini API returned HTTP {resp.status_code}.",
            )
        return {"status": "ok", "model": self._model}

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await self._chat(
            system_prompt, user_prompt, json_mode=False,
            temperature=temperature, max_tokens=max_tokens,
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
        prompt = user_prompt
        for attempt in (1, 2):
            raw = await self._chat(
                system_prompt, prompt, json_mode=True,
                temperature=temperature, max_tokens=max_tokens,
            )
            enforce_output_size(raw)
            parse_started = time.perf_counter()
            try:
                parsed = safe_parse_json(raw)
                OllamaLLMService._validate_keys(parsed, required_keys)
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                return parsed
            except ValueError as exc:
                metrics.observe_stage("json_parse", time.perf_counter() - parse_started)
                logger.warning(
                    "LLM returned invalid JSON (attempt %d/2, model=%s, error=%s, raw_chars=%d)",
                    attempt, self._model, type(exc).__name__, len(raw),
                )
                prompt = f"{user_prompt}\n\n{JSON_RETRY_PROMPT}"

        raise AppError(
            ErrorCode.LLM_INVALID_JSON,
            "The language model did not return valid structured output.",
        )

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
            "maxOutputTokens": max_tokens or self._num_predict,
        }
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": gen_cfg,
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    @staticmethod
    def _extract_text(obj: Any, *, allow_empty_candidates: bool = False) -> str:
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
        if not isinstance(content, dict):
            if allow_empty_candidates:
                return ""
            raise ValueError("Missing content")
        parts = content.get("parts")
        if not isinstance(parts, list):
            if allow_empty_candidates:
                return ""
            raise ValueError("Missing parts")
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and not part.get("thought"):
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "".join(texts)

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
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
        if not self._api_key:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API key is not configured.")
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode,
            temperature=temperature, max_tokens=max_tokens,
        )
        model_id = self._model.removeprefix("models/")
        endpoint = f"/models/{model_id}:streamGenerateContent?alt=sse"
        client = await self._get_client()
        call_timeout(self._timeout, get_settings().ollama_connect_timeout)
        started = time.perf_counter()
        pieces = 0
        has_text = False
        retry_count = 0
        try:
            while True:
                retry = False
                async with client.stream("POST", endpoint, json=payload) as resp:
                    if resp.status_code == 404:
                        raise AppError(
                            ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                            f"Model '{self._model}' is not available in Gemini API.",
                        )
                    if resp.status_code != 200:
                        logger.error("Gemini stream status=%d", resp.status_code)
                        if resp.status_code in (_RETRYABLE_STATUS | {429}) and retry_count < self._max_retries:
                            retry = True
                        else:
                            raise AppError(
                                ErrorCode.OLLAMA_GENERATION_ERROR,
                                f"Language model returned HTTP {resp.status_code}.",
                            )
                    if not retry:
                        async for line in resp.aiter_lines():
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if stripped.startswith("data:"):
                                stripped = stripped[5:].strip()
                            if not stripped or stripped == "[DONE]":
                                continue
                            try:
                                obj = json.loads(stripped)
                                content = self._extract_text(obj, allow_empty_candidates=True)
                            except ValueError as exc:
                                raise AppError(
                                    ErrorCode.OLLAMA_INVALID_RESPONSE,
                                    "Language model returned an unexpected response format.",
                                ) from exc
                            if content:
                                pieces += 1
                                has_text = has_text or bool(content.strip())
                                yield content
                if not retry:
                    break
                retry_count += 1
                logger.warning(
                    "Gemini stream transient failure — retrying retry_count=%d model=%s",
                    retry_count, self._model,
                )
                await asyncio.sleep(self._retry_backoff_s)
        except httpx.ConnectError as exc:
            logger.error("Gemini unreachable: %s", exc)
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API is not reachable.") from exc
        except httpx.TimeoutException as exc:
            logger.error("Gemini stream timed out: %s", exc)
            raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.") from exc
        except httpx.HTTPError as exc:
            logger.error("Gemini HTTP error during stream: %s", exc)
            raise AppError(
                ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API returned an HTTP error."
            ) from exc

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
            if outcome is not None:
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
        if not self._api_key:
            raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API key is not configured.")
        payload = self._build_payload(
            system_prompt, user_prompt, json_mode=json_mode,
            temperature=temperature, max_tokens=max_tokens,
        )
        model_id = self._model.removeprefix("models/")
        endpoint = f"/models/{model_id}:generateContent"

        client = await self._get_client()
        started = time.perf_counter()
        retry_count = 0
        while True:
            timeout = call_timeout(self._timeout, get_settings().ollama_connect_timeout)
            kwargs = {} if timeout is None else {"timeout": timeout}
            try:
                resp = await client.post(endpoint, json=payload, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Gemini unreachable: %s", exc)
                raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API is not reachable.") from exc
            except httpx.TimeoutException as exc:
                logger.error("Gemini generation timed out: %s", exc)
                raise AppError(ErrorCode.OLLAMA_TIMEOUT, "Language model request timed out.") from exc
            except _CONNECTION_RESET as exc:
                if retry_count < self._max_retries and can_retry(self._retry_backoff_s, 5.0):
                    retry_count += 1
                    metrics.event("llm_retry", "connection_reset")
                    logger.warning("Gemini connection reset - retrying retry_count=%d model=%s", retry_count, self._model)
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                logger.error("Gemini connection reset: %s", exc)
                raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API dropped the connection.") from exc
            except httpx.HTTPError as exc:
                logger.error("Gemini HTTP error: %s", exc)
                raise AppError(
                    ErrorCode.OLLAMA_UNAVAILABLE, "Gemini API returned an HTTP error."
                ) from exc

            if (resp.status_code in (_RETRYABLE_STATUS | {429}) and retry_count < self._max_retries
                    and can_retry(self._retry_backoff_s, 5.0)):
                retry_count += 1
                metrics.event("llm_retry", "transient")
                logger.warning(
                    "Gemini transient failure status=%d — retrying retry_count=%d model=%s",
                    resp.status_code, retry_count, self._model,
                )
                await asyncio.sleep(self._retry_backoff_s)
                continue
            break

        elapsed_ms = (time.perf_counter() - started) * 1000

        if resp.status_code == 404:
            raise AppError(
                ErrorCode.OLLAMA_MODEL_NOT_FOUND,
                f"Model '{self._model}' is not available in Gemini API.",
            )
        if resp.status_code != 200:
            logger.error("Gemini chat status=%d", resp.status_code)
            raise AppError(
                ErrorCode.OLLAMA_GENERATION_ERROR,
                f"Language model returned HTTP {resp.status_code}.",
            )

        try:
            content = self._extract_text(resp.json())
        except Exception as exc:
            raise AppError(
                ErrorCode.OLLAMA_INVALID_RESPONSE,
                "Language model returned an unexpected response format.",
            ) from exc

        if not content.strip():
            raise AppError(ErrorCode.LLM_EMPTY_RESPONSE, "Language model returned an empty response.")

        logger.info(
            "LLM generation | model=%s json_mode=%s prompt_chars=%d response_length=%d "
            "retry_count=%d llm_time_ms=%.1f",
            self._model, json_mode, len(system_prompt) + len(user_prompt), len(content),
            retry_count, elapsed_ms,
        )
        return content


# ── Module-level singleton ──────────────────────────────────────────

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return a shared LLMService (Gemini when GEMINI_API_KEY is configured, else Ollama)."""
    global _llm_service
    if _llm_service is None:
        s = get_settings()
        if s.gemini_api_key.strip():
            _llm_service = GeminiLLMService()
        else:
            _llm_service = OllamaLLMService()
    return _llm_service
