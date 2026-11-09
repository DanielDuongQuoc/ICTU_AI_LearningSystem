"""Submissions API: handwritten answers -> text (the transcript is then graded via /api/grading/grade)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit
from app.core.request_context import get_request_id
from app.services.transcription_service import (
    TranscriptionResponse, TranscriptionService, get_transcription_service,
)

router = APIRouter(prefix="/api/submissions", tags=["Submissions"])
TRANSCRIBE_PATH = "/api/submissions/handwritten/transcribe"


def _get_transcription_service() -> TranscriptionService:
    return get_transcription_service()


@router.post(
    "/handwritten/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe a handwritten answer (page images) to text",
    description=(
        "Multipart upload: `files` (JPG/PNG/PDF pages, in page order, at most MAX_HANDWRITTEN_PAGES pages and "
        "MAX_UPLOAD_SIZE_MB in total), optional `question`, `language` (default vi) and `mask_header_ratio` "
        "(0-0.4, default 0.12: the top of page 1 is blanked so names / student ids never reach the model). "
        "Returns the transcript per page (`[?]` = unreadable word, diagrams described in brackets) and review "
        "`flags` (`unreadable`, `diagram_present`). Grade the `text` with `/api/grading/grade` using "
        "`submission_kind=handwritten_transcript` and `extra_flags=flags`; the lecturer can correct the transcript first."
    ),
    responses=error_responses(400, 413, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def transcribe_handwritten(
    request: Request,
    files: List[UploadFile] = File(...),
    question: str = Form(""),
    language: str = Form("vi"),
    mask_header_ratio: float = Form(0.12),
) -> TranscriptionResponse:
    settings = get_settings()
    if not (0.0 <= mask_header_ratio <= 0.4):
        raise AppError(ErrorCode.VALIDATION_ERROR, "mask_header_ratio must be between 0 and 0.4.", status_code=422)
    if len(language) != 2 or not language.isalpha():
        raise AppError(ErrorCode.VALIDATION_ERROR, "language must be a 2-letter code.", status_code=422)
    if len(question) > settings.max_question_chars:
        raise AppError(ErrorCode.INPUT_TOO_LONG, f"question must be at most {settings.max_question_chars} characters.")
    blobs: list[bytes] = []
    total = 0
    for f in files:
        data = bytearray()
        while chunk := await f.read(1024 * 1024):
            data += chunk
            total += len(chunk)
            if total > settings.max_upload_size_bytes:
                raise AppError(ErrorCode.FILE_TOO_LARGE,
                               f"Pages exceed the maximum upload size of {settings.max_upload_size_mb} MB.")
        if not data:
            raise AppError(ErrorCode.EMPTY_FILE, "An uploaded page is empty.")
        blobs.append(bytes(data))
    return await _get_transcription_service().transcribe(
        blobs, question=question, language=language.lower(), mask_header_ratio=mask_header_ratio,
        request_id=get_request_id(request))
