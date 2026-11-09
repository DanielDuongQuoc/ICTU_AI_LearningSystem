"""Retrieval over one or more documents for rubric generation and grading (optional reference material)."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional
from uuid import UUID

from app.core import runtime_config
from app.core.exceptions import AppError, ErrorCode
from app.services.rag_service import RAGService

MAX_QUERY_CHARS = 1000
MAX_SOURCES_PER_ITEM = 2
MIN_LEXICAL_OVERLAP = 0.2
_WORD = re.compile(r"\w+", re.UNICODE)


async def build_context(
    rag: RAGService,
    document_ids: Iterable[UUID | str],
    query: str,
    request_id: Optional[str] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """``(context, chunks)`` from every requested document. Unknown documents raise DOCUMENT_NOT_FOUND;
    documents without relevant chunks simply contribute nothing."""
    ids = [str(d) for d in dict.fromkeys(str(x) for x in document_ids)]
    if not ids:
        return "", []
    eff = runtime_config.current()
    query = query.strip()[:MAX_QUERY_CHARS]
    parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    for doc_id in ids:
        retrieval = await rag.retrieve(doc_id, query, top_k=eff.top_k, request_id=request_id)
        if retrieval["insufficient_context"] or not retrieval["chunks"]:
            continue
        for c in retrieval["chunks"]:
            if c.get("document_id") != doc_id:
                raise AppError(ErrorCode.SOURCE_INTEGRITY_ERROR, "Source verification failed.")
        chunks.extend(retrieval["chunks"])
        parts.append(retrieval["context"])
    context = "\n\n---\n\n".join(parts)[: eff.max_context_chars]
    return context, chunks


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def pick_sources(text: str, chunks: list[dict[str, Any]]) -> list[str]:
    """Chunk ids supporting ``text`` by word overlap; falls back to the best retrieved chunk."""
    if not chunks:
        return []
    tokens = _tokens(text)
    scored = sorted(
        ((len(tokens & _tokens(c.get("text", ""))) / len(tokens) if tokens else 0.0, c["chunk_id"]) for c in chunks),
        key=lambda t: t[0], reverse=True,
    )
    picked = [cid for ov, cid in scored if ov >= MIN_LEXICAL_OVERLAP][:MAX_SOURCES_PER_ITEM]
    if picked:
        return picked
    best = max(chunks, key=lambda c: c.get("score", 0.0))
    return [best["chunk_id"]]
