"""Concurrency gate for LLM generation (Phase 16 backpressure).

Caps simultaneous Ollama generation calls (``LLM_MAX_CONCURRENCY``, 0 = unlimited). A call that
cannot get a slot within ``LLM_QUEUE_TIMEOUT_S`` is rejected with ``SERVICE_OVERLOADED`` (503)
instead of piling up behind the model and causing a timeout cascade. No queue system: the wait
is a plain in-process semaphore. The semaphore is created per running event loop.
"""

from __future__ import annotations

import asyncio
import weakref
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics


class LLMGate:
    def __init__(self) -> None:
        self._sems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]]" = (
            weakref.WeakKeyDictionary()
        )

    def _semaphore(self, limit: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        entry = self._sems.get(loop)
        if entry is None or entry[0] != limit:
            entry = (limit, asyncio.Semaphore(limit))
            self._sems[loop] = entry
        return entry[1]

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        settings = get_settings()
        limit = settings.llm_max_concurrency
        if limit <= 0:
            yield
            return
        sem = self._semaphore(limit)
        metrics.llm_waiting(+1)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=settings.llm_queue_timeout_s)
        except asyncio.TimeoutError:
            metrics.llm_rejected()
            raise AppError(
                ErrorCode.SERVICE_OVERLOADED,
                "The language model is busy; please retry shortly.",
            ) from None
        finally:
            metrics.llm_waiting(-1)
        metrics.llm_active(+1)
        try:
            yield
        finally:
            metrics.llm_active(-1)
            sem.release()


llm_gate = LLMGate()


class EmbeddingGate:
    """Same backpressure for Ollama embedding calls (Phase 22): ``MAX_CONCURRENT_EMBEDDINGS`` slots, 0 = unlimited.

    A call that cannot get a slot within ``EMBEDDING_QUEUE_TIMEOUT_S`` is rejected with ``SERVICE_OVERLOADED`` (503).
    """

    def __init__(self) -> None:
        self._sems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]]" = (
            weakref.WeakKeyDictionary()
        )

    def _semaphore(self, limit: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        entry = self._sems.get(loop)
        if entry is None or entry[0] != limit:
            entry = (limit, asyncio.Semaphore(limit))
            self._sems[loop] = entry
        return entry[1]

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        settings = get_settings()
        limit = settings.max_concurrent_embeddings
        if limit <= 0:
            yield
            return
        sem = self._semaphore(limit)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=settings.embedding_queue_timeout_s)
        except asyncio.TimeoutError:
            metrics.event("embedding_request", "rejected")
            raise AppError(
                ErrorCode.SERVICE_OVERLOADED,
                "The embedding service is busy; please retry shortly.",
            ) from None
        try:
            yield
        finally:
            sem.release()


embedding_gate = EmbeddingGate()
