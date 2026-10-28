from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional

from app.core.config import get_settings
from app.schemas.document import Chunk, ChunkMetadata, ExtractedPage

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+\S.*|\d+(?:\.\d+)+\s+\S.{0,100}|[A-Z0-9][A-Z0-9 ,.\-]{3,100})$"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int
    kind: str


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^\d+(?:\.\d+)+\s+\S", stripped):
        return True
    if stripped.endswith((".", "?", "!", ",", ";", ":")):
        return False
    return bool(_HEADING_RE.match(stripped))


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _units_from_page(page: ExtractedPage, chunk_size: int) -> list[_Unit]:
    text = page.text.strip()
    if not text:
        return []

    units: list[_Unit] = []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    for paragraph in paragraphs:
        first_line = paragraph.split("\n", 1)[0]
        kind = "heading" if _is_heading(first_line) and "\n" not in paragraph else "paragraph"
        if len(paragraph) <= chunk_size:
            units.append(_Unit(paragraph, page.page, kind))
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(_Unit(sentence, page.page, "sentence"))
            else:
                for word in sentence.split(" "):
                    if word:
                        units.append(_Unit(word, page.page, "words"))
    return units


def _join_units(units: Iterable[_Unit]) -> str:
    parts: list[str] = []
    for unit in units:
        if not parts:
            parts.append(unit.text)
            continue
        separator = "\n\n" if unit.kind in {"heading", "paragraph"} else " "
        parts.append(separator + unit.text)
    return "".join(parts)


def _metadata_for_pages(pages: list[int], source: str) -> ChunkMetadata:
    unique = sorted(set(pages))
    if len(unique) == 1:
        return ChunkMetadata(source=source, page=unique[0], pages=None)
    return ChunkMetadata(source=source, page=None, pages=unique)


def _overlap_units(previous: list[_Unit], overlap: int) -> list[_Unit]:
    if overlap <= 0 or not previous:
        return []
    selected: list[_Unit] = []
    total = 0
    for unit in reversed(previous):
        selected.insert(0, unit)
        total += len(unit.text) + 1
        if total >= overlap:
            break
    return selected


def create_chunks(
    pages: list[ExtractedPage],
    document_id: str,
    source: str,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Chunk]:
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap >= size:
        overlap = max(size // 5, 0)

    units: list[_Unit] = []
    for page in pages:
        units.extend(_units_from_page(page, size))

    if not units:
        return []

    chunks: list[Chunk] = []
    index = 0
    total = len(units)
    previous_units: list[_Unit] = []

    while index < total:
        prefix_units = _overlap_units(previous_units, overlap) if chunks else []
        prefix_len = len(_join_units(prefix_units)) if prefix_units else 0
        new_budget = size if not prefix_units else max(size - prefix_len, 1)

        selected: list[_Unit] = []
        cursor = index
        current_len = 0  # == len(_join_units(selected)); tracked incrementally (was rebuilt per unit: O(k^2))
        while cursor < total:
            unit = units[cursor]
            if selected:
                separator = "\n\n" if unit.kind in {"heading", "paragraph"} else " "
                candidate_len = current_len + len(separator) + len(unit.text)
            else:
                candidate_len = len(unit.text)
            if selected and candidate_len > new_budget:
                break
            selected.append(unit)
            current_len = candidate_len
            cursor += 1
            if candidate_len >= new_budget:
                break

        if not selected:
            selected = [units[index]]
            cursor = index + 1

        combined = prefix_units + selected
        text = _join_units(combined)
        pages_used = [unit.page for unit in combined]
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=text,
                metadata=_metadata_for_pages(pages_used, source),
            )
        )
        previous_units = selected
        index = cursor

    logger.debug(
        "Created %s chunks for document_id=%s source=%s",
        len(chunks),
        document_id,
        source,
    )
    return chunks
