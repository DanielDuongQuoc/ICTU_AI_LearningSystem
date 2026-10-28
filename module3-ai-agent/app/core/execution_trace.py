"""Structured, secret-free execution trace: which AI configuration produced a result.

One log line per pipeline execution on the ``app.ai_trace`` logger.  Contains identifiers and
configuration only - never document text, prompts, student answers, credentials or URLs.
Chunk ids/pages come from the backend (retrieval metadata), never from model output.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from app.core import runtime_config

logger = logging.getLogger("app.ai_trace")


def trace_execution(
    task: str,
    request_id: str,
    document_id: Optional[str],
    outcome: str,
    latency_ms: float,
    *,
    source_chunk_ids: Optional[Iterable[str]] = None,
    llm_called: Optional[bool] = None,
) -> None:
    cfg = runtime_config.current()
    ids = list(source_chunk_ids or [])
    logger.info(
        "AI execution | task=%s request_id=%s document_id=%s outcome=%s latency_ms=%.1f "
        "config_version=%s config_source=%s config_fingerprint=%s rag_fingerprint=%s "
        "prompt_version=%s llm_provider=%s llm_model=%s embedding_model=%s top_k=%d "
        "similarity_threshold=%s llm_called=%s source_chunks=%d source_chunk_ids=%s",
        task, request_id, document_id, outcome, latency_ms,
        cfg.version_id, cfg.source, cfg.fingerprint[:16], cfg.rag_fingerprint[:16],
        cfg.prompt_version(task), cfg.llm_provider, cfg.llm_model, cfg.embedding_model, cfg.top_k,
        cfg.similarity_threshold, llm_called, len(ids), ",".join(ids[:20]),
    )
