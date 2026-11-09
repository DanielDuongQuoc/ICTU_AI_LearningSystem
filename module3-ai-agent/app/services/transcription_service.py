"""Handwritten submissions: page images -> masked, normalised PNGs -> vision transcription -> text + review flags.

The transcript is then graded with ``/api/grading/grade`` (``submission_kind="handwritten_transcript"``), passing the
returned ``flags`` as ``extra_flags`` so homework with unreadable parts or diagrams goes to the lecturer.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Literal, Optional

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from app.core import prompt_registry, runtime_config
from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.output_guard import clamp_text, enforce_no_prompt_leak
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

MAX_SIDE_PX = 1600
MAX_LLM_ATTEMPTS = 2
MAX_PAGE_TEXT_CHARS = 20000
UNREADABLE_MARK = "[?]"
_WORD = re.compile(r"\w+", re.UNICODE)


class PageTranscript(BaseModel):
    page: int
    text: str
    confidence: Literal["high", "medium", "low"]
    unreadable_count: int
    diagram_present: bool


class TranscriptionTrace(BaseModel):
    model: str
    prompt_version: str
    request_id: str


class TranscriptionResponse(BaseModel):
    pages: list[PageTranscript]
    text: str = Field(..., description="All pages joined; send as `answer` to /api/grading/grade")
    flags: list[str] = Field(default_factory=list, description="Pass as `extra_flags` when grading")
    trace: TranscriptionTrace


# ── image preparation ──────────────────────────────────────────────


def _kind(data: bytes) -> Optional[str]:
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return None


def prepare_pages(files: list[bytes], mask_header_ratio: float = 0.12) -> list[tuple[str, bytes]]:
    """Render every uploaded file (JPG/PNG/PDF) to PNG pages, downscale, and blank the header of page 1
    (where the name / student id usually is) so it never reaches the model."""
    max_pages = get_settings().max_handwritten_pages
    pages: list[tuple[str, bytes]] = []
    for data in files:
        kind = _kind(data)
        if kind is None:
            raise AppError(ErrorCode.INVALID_FILE, "Only JPG, PNG or PDF pages are accepted.")
        try:
            doc = fitz.open(stream=data, filetype=kind)
        except Exception as exc:
            raise AppError(ErrorCode.INVALID_FILE, "A page file could not be opened.") from exc
        with doc:
            for page in doc:
                if len(pages) >= max_pages:
                    raise AppError(ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                                   f"At most {max_pages} pages per handwritten submission.")
                rect = page.rect
                zoom = min(2.0, MAX_SIDE_PX / max(rect.width, rect.height, 1))
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                if not pages and mask_header_ratio > 0:
                    pix.set_rect(fitz.IRect(0, 0, pix.width, int(pix.height * mask_header_ratio)), (255, 255, 255))
                pages.append(("image/png", pix.tobytes("png")))
    if not pages:
        raise AppError(ErrorCode.EMPTY_FILE, "No page images were uploaded.")
    return pages


# ── transcription ──────────────────────────────────────────────────


def parse_transcription(raw: Any, page_count: int) -> list[PageTranscript]:
    bad = AppError(ErrorCode.TRANSCRIPTION_INVALID_RESPONSE, "The model returned an unusable transcription.")
    items = raw.get("pages") if isinstance(raw, dict) else None
    if not isinstance(items, list) or len(items) != page_count:
        raise bad
    out: list[PageTranscript] = []
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict) or not isinstance(it.get("text"), str):
            raise bad
        conf = it.get("confidence") if it.get("confidence") in ("high", "medium", "low") else "low"
        text = clamp_text(it["text"].strip(), MAX_PAGE_TEXT_CHARS)
        out.append(PageTranscript(page=i, text=text, confidence=conf, unreadable_count=text.count(UNREADABLE_MARK),
                                  diagram_present=it.get("diagram_present") is True))
    enforce_no_prompt_leak(*(p.text for p in out))
    return out


def transcription_flags(pages: list[PageTranscript]) -> list[str]:
    flags: list[str] = []
    words = sum(len(_WORD.findall(p.text)) for p in pages)
    unreadable = sum(p.unreadable_count for p in pages)
    if any(p.confidence == "low" for p in pages) or unreadable > max(3, 0.05 * words) or words == 0:
        flags.append("unreadable")
    if any(p.diagram_present for p in pages):
        flags.append("diagram_present")
    return flags


class TranscriptionService:
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self._llm = llm_service or get_llm_service()

    async def transcribe(
        self, files: list[bytes], *, question: str = "", language: str = "vi", mask_header_ratio: float = 0.12,
        request_id: Optional[str] = None,
    ) -> TranscriptionResponse:
        request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        pages = prepare_pages(files, mask_header_ratio)
        version = runtime_config.current().prompt_version(prompt_registry.TASK_HANDWRITING_TRANSCRIPTION)
        system_prompt, user_prompt = prompt_registry.build(
            prompt_registry.TASK_HANDWRITING_TRANSCRIPTION, version, len(pages), question[:2000], language)
        last: Optional[AppError] = None
        result: Optional[list[PageTranscript]] = None
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            raw = await self._llm.generate_json(
                system_prompt, user_prompt, required_keys=["pages"], images=pages,
                max_tokens=get_settings().rubric_max_output_tokens)
            try:
                result = parse_transcription(raw, len(pages))
                break
            except AppError as exc:
                last = exc
                logger.warning("Invalid transcription | request_id=%s attempt=%d/%d", request_id, attempt,
                               MAX_LLM_ATTEMPTS)
        if result is None:
            assert last is not None
            raise last
        flags = transcription_flags(result)
        logger.info("Transcription finished | request_id=%s pages=%d flags=%s total_ms=%.1f", request_id,
                    len(result), ",".join(flags) or "-", (time.perf_counter() - started) * 1000)
        return TranscriptionResponse(
            pages=result, text="\n\n".join(p.text for p in result), flags=flags,
            trace=TranscriptionTrace(model=self._llm.model_name, request_id=request_id,
                                     prompt_version=f"{prompt_registry.TASK_HANDWRITING_TRANSCRIPTION}:{version}"),
        )


_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service

