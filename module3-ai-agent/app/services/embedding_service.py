"""Phase 3 — Embedding service backed by Google AI Studio (Gemini) or Ollama.

Architecture
~~~~~~~~~~~~
API → Service → EmbeddingService / GeminiEmbeddingService → Provider Embedding API

The service reuses a single ``httpx.AsyncClient`` for connection pooling.
Model name, base URL and API key are read from ``Settings`` (never hard-coded).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from app.core.concurrency import embedding_gate
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.timeouts import call_timeout, can_retry

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate vector embeddings via the Ollama embedding API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._base_url: str = (base_url or settings.ollama_base_url).rstrip("/")
        self._model: str = model or settings.ollama_embedding_model
        self._timeout: int = timeout or settings.ollama_timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── Client lifecycle ────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().ollama_connect_timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── Public API ──────────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns a flat list of floats."""
        if not text or not text.strip():
            raise AppError(
                ErrorCode.EMBEDDING_EMPTY_TEXT,
                "Cannot create embedding for empty text.",
            )

        embeddings = await self._timed_call([text])
        return embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch call.

        Returns a list of embedding vectors, one per input text.
        Empty strings in the batch are rejected upfront.
        """
        if not texts:
            raise AppError(
                ErrorCode.EMBEDDING_EMPTY_TEXT,
                "Cannot create embeddings for an empty list of texts.",
            )

        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise AppError(
                    ErrorCode.EMBEDDING_EMPTY_TEXT,
                    f"Text at index {idx} is empty.",
                )

        # Phase 22: bounded batches, sent one after another (bounded memory and Ollama load; order preserved).
        size = max(1, get_settings().embedding_batch_size)
        if len(texts) <= size:
            return await self._timed_call(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(await self._timed_call(texts[start:start + size]))
        return vectors

    async def _timed_call(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        outcome: Optional[str] = None
        try:
            async with embedding_gate.slot():
                result = await self._call_ollama(texts)
            outcome = "ok"
            return result
        except AppError as exc:
            outcome = "timeout" if exc.code == ErrorCode.EMBEDDING_TIMEOUT else "error"
            raise
        finally:
            metrics.observe_stage("embedding", time.perf_counter() - started)
            if outcome is not None:
                metrics.event("embedding_request", outcome)

    # ── Internal ────────────────────────────────────────────────────

    async def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        """Send a batch embedding request to Ollama ``/api/embed``.

        Ollama's ``/api/embed`` endpoint accepts ``{"model": "...", "input": [...]}``
        and returns ``{"embeddings": [[...], ...]}``.
        """
        client = await self._get_client()
        payload = {"model": self._model, "input": texts}
        started = time.perf_counter()

        settings = get_settings()
        retry_count = 0
        while True:  # same policy as the chat call: ONE retry, transient causes only, within the request budget
            timeout = call_timeout(self._timeout, settings.ollama_connect_timeout, code=ErrorCode.EMBEDDING_TIMEOUT)
            kwargs = {} if timeout is None else {"timeout": timeout}
            try:
                response = await client.post("/api/embed", json=payload, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Ollama embedding service unreachable: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service (Ollama) is not reachable.",
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error("Ollama embedding request timed out: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_TIMEOUT,
                    "Embedding request timed out.",
                ) from exc
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as exc:
                if retry_count < max(0, settings.ollama_max_retries) and can_retry(settings.ollama_retry_backoff_s, 5.0):
                    retry_count += 1
                    metrics.event("embedding_retry", "connection_reset")
                    logger.warning("Ollama embedding connection reset - retrying retry_count=%d", retry_count)
                    await asyncio.sleep(max(0.0, settings.ollama_retry_backoff_s))
                    continue
                logger.error("Ollama embedding connection reset: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service dropped the connection.",
                ) from exc
            except httpx.HTTPError as exc:
                logger.error("Ollama embedding HTTP error: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service returned an HTTP error.",
                ) from exc
            if (response.status_code in (502, 503, 504) and retry_count < max(0, settings.ollama_max_retries)
                    and can_retry(settings.ollama_retry_backoff_s, 5.0)):
                retry_count += 1
                metrics.event("embedding_retry", "transient")
                logger.warning("Ollama embedding transient status=%d - retrying retry_count=%d",
                               response.status_code, retry_count)
                await asyncio.sleep(max(0.0, settings.ollama_retry_backoff_s))
                continue
            break

        elapsed_ms = (time.perf_counter() - started) * 1000

        # ── Handle HTTP errors ──────────────────────────────────────
        if response.status_code == 404:
            raise AppError(
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                f"Embedding model '{self._model}' is not available.",
            )
        if response.status_code != 200:
            body = response.text[:300]
            logger.error(
                "Ollama embedding returned status=%d body=%s",
                response.status_code,
                body,
            )
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                f"Embedding service returned HTTP {response.status_code}.",
            )

        # ── Parse response ──────────────────────────────────────────
        try:
            data = response.json()
        except Exception as exc:
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                "Embedding service returned non-JSON response.",
            ) from exc

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                "Embedding response does not contain valid embeddings.",
            )

        if len(embeddings) != len(texts):
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                f"Expected {len(texts)} embeddings, got {len(embeddings)}.",
            )

        logger.info(
            "Embedding completed | model=%s texts=%d dimension=%d time_ms=%.1f",
            self._model,
            len(texts),
            len(embeddings[0]) if embeddings else 0,
            elapsed_ms,
        )
        return embeddings

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self._model


class GeminiEmbeddingService(EmbeddingService):
    """Generate vector embeddings via Google AI Studio (Gemini ``batchEmbedContents``)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            base_url=base_url or settings.gemini_base_url,
            model=model or settings.gemini_embedding_model,
            timeout=timeout or settings.ollama_timeout,
        )
        self._api_key: str = (api_key if api_key is not None else settings.gemini_api_key).strip()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"x-goog-api-key": self._api_key} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().ollama_connect_timeout),
            )
        return self._client

    async def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        """Send a batch embedding request to Gemini ``/models/{model}:batchEmbedContents``."""
        if not self._api_key:
            raise AppError(
                ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                "Gemini API key is not configured.",
            )
        client = await self._get_client()
        model_id = self._model.removeprefix("models/")
        full_model = f"models/{model_id}"
        endpoint = f"/models/{model_id}:batchEmbedContents"
        payload = {
            "requests": [
                {"model": full_model, "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        started = time.perf_counter()

        settings = get_settings()
        retry_count = 0
        while True:
            timeout = call_timeout(self._timeout, settings.ollama_connect_timeout, code=ErrorCode.EMBEDDING_TIMEOUT)
            kwargs = {} if timeout is None else {"timeout": timeout}
            try:
                response = await client.post(endpoint, json=payload, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Gemini embedding service unreachable: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service (Gemini) is not reachable.",
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error("Gemini embedding request timed out: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_TIMEOUT,
                    "Embedding request timed out.",
                ) from exc
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as exc:
                if retry_count < max(0, settings.ollama_max_retries) and can_retry(settings.ollama_retry_backoff_s, 5.0):
                    retry_count += 1
                    metrics.event("embedding_retry", "connection_reset")
                    logger.warning("Gemini embedding connection reset - retrying retry_count=%d", retry_count)
                    await asyncio.sleep(max(0.0, settings.ollama_retry_backoff_s))
                    continue
                logger.error("Gemini embedding connection reset: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service dropped the connection.",
                ) from exc
            except httpx.HTTPError as exc:
                logger.error("Gemini embedding HTTP error: %s", exc)
                raise AppError(
                    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                    "Embedding service returned an HTTP error.",
                ) from exc
            if (response.status_code in (429, 502, 503, 504) and retry_count < max(0, settings.ollama_max_retries)
                    and can_retry(settings.ollama_retry_backoff_s, 5.0)):
                retry_count += 1
                metrics.event("embedding_retry", "transient")
                logger.warning("Gemini embedding transient status=%d - retrying retry_count=%d",
                               response.status_code, retry_count)
                await asyncio.sleep(max(0.0, settings.ollama_retry_backoff_s))
                continue
            break

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            raise AppError(
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                f"Embedding model '{self._model}' is not available.",
            )
        if response.status_code != 200:
            logger.error("Gemini embedding returned status=%d", response.status_code)
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                f"Embedding service returned HTTP {response.status_code}.",
            )

        try:
            data = response.json()
        except Exception as exc:
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                "Embedding service returned non-JSON response.",
            ) from exc

        raw_embeddings = data.get("embeddings")
        if not raw_embeddings or not isinstance(raw_embeddings, list):
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                "Embedding response does not contain valid embeddings.",
            )

        embeddings: list[list[float]] = []
        for item in raw_embeddings:
            values = item.get("values") if isinstance(item, dict) else item
            if not isinstance(values, list) or not values:
                raise AppError(
                    ErrorCode.EMBEDDING_INVALID_RESPONSE,
                    "Embedding response does not contain valid vector values.",
                )
            embeddings.append(values)

        if len(embeddings) != len(texts):
            raise AppError(
                ErrorCode.EMBEDDING_INVALID_RESPONSE,
                f"Expected {len(texts)} embeddings, got {len(embeddings)}.",
            )

        logger.info(
            "Embedding completed | model=%s texts=%d dimension=%d time_ms=%.1f",
            self._model,
            len(texts),
            len(embeddings[0]) if embeddings else 0,
            elapsed_ms,
        )
        return embeddings


# ── Module-level singleton ──────────────────────────────────────────

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Return a module-level EmbeddingService singleton (Gemini when GEMINI_API_KEY is set)."""
    global _embedding_service
    if _embedding_service is None:
        s = get_settings()
        if s.gemini_api_key.strip():
            _embedding_service = GeminiEmbeddingService()
        else:
            _embedding_service = EmbeddingService()
    return _embedding_service
