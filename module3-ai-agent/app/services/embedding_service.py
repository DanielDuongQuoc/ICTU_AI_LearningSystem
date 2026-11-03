"""Embedding service backed by the Google Gemini API (``batchEmbedContents``).

Architecture
~~~~~~~~~~~~
VectorService / RAGService → GeminiEmbeddingService → Gemini embedding API

The service reuses a single ``httpx.AsyncClient`` for connection pooling.
Model name, base URL and API key are read from ``Settings`` (never hard-coded).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Optional

import httpx

from app.core.concurrency import embedding_gate
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.timeouts import call_timeout, can_retry

logger = logging.getLogger(__name__)

# Same transient policy as the generation call (429 = quota / rate limit).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GeminiEmbeddingService:
    """Generate vector embeddings via Gemini ``/models/{model}:batchEmbedContents``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._api_key: str = (api_key if api_key is not None else settings.gemini_api_key).strip()
        self._base_url: str = (base_url or settings.gemini_base_url).rstrip("/")
        self._model: str = model or settings.gemini_embedding_model
        self._timeout: float = timeout or settings.llm_timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self._model

    # ── Client lifecycle ────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            headers = {"x-goog-api-key": self._api_key} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout, connect=get_settings().llm_connect_timeout),
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
            raise AppError(ErrorCode.EMBEDDING_EMPTY_TEXT, "Cannot create embedding for empty text.")
        embeddings = await self._timed_call([text])
        return embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts; one vector per input text, order preserved.

        Texts are sent in sequential batches of at most ``EMBEDDING_BATCH_SIZE`` (bounded memory and
        provider load). Empty strings are rejected upfront. All vectors must share one dimension.
        """
        if not texts:
            raise AppError(ErrorCode.EMBEDDING_EMPTY_TEXT, "Cannot create embeddings for an empty list of texts.")
        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise AppError(ErrorCode.EMBEDDING_EMPTY_TEXT, f"Text at index {idx} is empty.")

        size = max(1, get_settings().embedding_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(await self._timed_call(texts[start:start + size]))
        if len({len(v) for v in vectors}) > 1:
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE, "Embedding vectors have inconsistent dimensions.")
        return vectors

    async def _timed_call(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        outcome: Optional[str] = None
        try:
            async with embedding_gate.slot():
                result = await self._call_api(texts)
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

    async def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with ONE retry for transient causes only, always within the request budget."""
        settings = get_settings()
        max_retries = max(0, settings.llm_max_retries)
        backoff = max(0.0, settings.llm_retry_backoff_s)
        client = await self._get_client()
        retry_count = 0
        while True:
            timeout = call_timeout(self._timeout, settings.llm_connect_timeout, code=ErrorCode.EMBEDDING_TIMEOUT)
            kwargs = {} if timeout is None else {"timeout": timeout}
            try:
                response = await client.post(endpoint, json=payload, **kwargs)
            except httpx.ConnectError as exc:
                logger.error("Gemini embedding service unreachable: %s", exc)
                raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                               "Embedding service (Gemini) is not reachable.") from exc
            except httpx.TimeoutException as exc:
                logger.error("Gemini embedding request timed out: %s", exc)
                raise AppError(ErrorCode.EMBEDDING_TIMEOUT, "Embedding request timed out.") from exc
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as exc:
                if retry_count < max_retries and can_retry(backoff, 5.0):
                    retry_count += 1
                    metrics.event("embedding_retry", "connection_reset")
                    logger.warning("Gemini embedding connection reset - retrying retry_count=%d", retry_count)
                    await asyncio.sleep(backoff)
                    continue
                logger.error("Gemini embedding connection reset: %s", exc)
                raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                               "Embedding service dropped the connection.") from exc
            except httpx.HTTPError as exc:
                logger.error("Gemini embedding HTTP error: %s", exc)
                raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
                               "Embedding service returned an HTTP error.") from exc
            if response.status_code in RETRYABLE_STATUS and retry_count < max_retries and can_retry(backoff, 5.0):
                retry_count += 1
                metrics.event("embedding_retry", "transient")
                logger.warning("Gemini embedding transient status=%d - retrying retry_count=%d",
                               response.status_code, retry_count)
                await asyncio.sleep(backoff)
                continue
            return response

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Send one ``batchEmbedContents`` request and validate the returned vectors."""
        if not self._api_key:
            raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "Gemini API key is not configured.")
        model_id = self._model.removeprefix("models/")
        payload = {
            "requests": [
                {"model": f"models/{model_id}", "content": {"parts": [{"text": t}]}} for t in texts
            ]
        }
        started = time.perf_counter()
        response = await self._post_with_retry(f"/models/{model_id}:batchEmbedContents", payload)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            raise AppError(ErrorCode.EMBEDDING_MODEL_NOT_FOUND, f"Embedding model '{self._model}' is not available.")
        if response.status_code != 200:
            logger.error("Gemini embedding returned status=%d", response.status_code)
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE,
                           f"Embedding service returned HTTP {response.status_code}.")
        try:
            data = response.json()
        except Exception as exc:
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE, "Embedding service returned non-JSON response.") from exc

        raw_embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not raw_embeddings or not isinstance(raw_embeddings, list):
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE, "Embedding response does not contain valid embeddings.")

        embeddings: list[list[float]] = []
        for item in raw_embeddings:
            values = item.get("values") if isinstance(item, dict) else item
            if (not isinstance(values, list) or not values
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                               for v in values)):
                raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE,
                               "Embedding response does not contain valid vector values.")
            embeddings.append(values)

        if len(embeddings) != len(texts):
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE,
                           f"Expected {len(texts)} embeddings, got {len(embeddings)}.")
        if len({len(v) for v in embeddings}) > 1:
            raise AppError(ErrorCode.EMBEDDING_INVALID_RESPONSE, "Embedding vectors have inconsistent dimensions.")

        logger.info(
            "Embedding completed | model=%s texts=%d dimension=%d time_ms=%.1f",
            self._model, len(texts), len(embeddings[0]), elapsed_ms,
        )
        return embeddings


# ── Module-level singleton ──────────────────────────────────────────

_embedding_service: Optional[GeminiEmbeddingService] = None


def get_embedding_service() -> GeminiEmbeddingService:
    """Return the module-level GeminiEmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = GeminiEmbeddingService()
    return _embedding_service
