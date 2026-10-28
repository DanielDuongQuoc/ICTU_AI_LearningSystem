from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import fitz

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.schemas.document import ExtractedPage

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
PathLike = Union[str, Path]


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def looks_like_pdf_bytes(data: bytes) -> bool:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.startswith(PDF_MAGIC)


def validate_pdf_path(
    path: PathLike,
    *,
    max_bytes: Optional[int] = None,
    declared_filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Path:
    """Validate a PDF on disk. Does not trust client-supplied names."""
    pdf_path = _as_path(path)
    limit = max_bytes if max_bytes is not None else get_settings().max_upload_size_bytes

    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        allowed = {
            "application/pdf",
            "application/x-pdf",
            "application/octet-stream",
            "binary/octet-stream",
        }
        if mime and mime not in allowed:
            raise AppError(
                ErrorCode.INVALID_FILE,
                "File MIME type is not PDF.",
            )

    if not pdf_path.exists():
        raise AppError(ErrorCode.FILE_NOT_FOUND, "File was not found.")

    if not pdf_path.is_file():
        raise AppError(ErrorCode.INVALID_FILE, "Path is not a file.")

    size = pdf_path.stat().st_size
    if size == 0:
        raise AppError(ErrorCode.EMPTY_FILE, "File is empty.")
    if size > limit:
        raise AppError(
            ErrorCode.FILE_TOO_LARGE,
            f"File exceeds the maximum upload size of {limit} bytes.",
        )

    name_for_ext = declared_filename or pdf_path.name
    if Path(name_for_ext).suffix.lower() != ".pdf":
        raise AppError(ErrorCode.INVALID_FILE, "Only PDF files are accepted.")

    with pdf_path.open("rb") as handle:
        header = handle.read(8)
    if not looks_like_pdf_bytes(header):
        raise AppError(
            ErrorCode.INVALID_FILE,
            "File content is not a PDF.",
        )

    return pdf_path


def extract_pages(path: PathLike) -> list[ExtractedPage]:
    """Open the PDF once, extract text per page, then close the document."""
    pdf_path = _as_path(path)
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF path=%s error=%s", pdf_path.name, exc)
        raise AppError(ErrorCode.INVALID_PDF, "The file is not a valid PDF.") from exc

    pages: list[ExtractedPage] = []
    try:
        if document.page_count == 0:
            raise AppError(ErrorCode.INVALID_PDF, "The PDF has no pages.")
        if document.is_encrypted and not document.authenticate(""):
            raise AppError(ErrorCode.INVALID_PDF, "The PDF cannot be opened.")

        settings = get_settings()
        if document.page_count > settings.max_pdf_pages:
            raise AppError(
                ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                f"The PDF has more than the allowed {settings.max_pdf_pages} pages.",
            )
        deadline = time.monotonic() + settings.pdf_processing_timeout_s
        total_chars = 0
        for index in range(document.page_count):
            if time.monotonic() > deadline:
                raise AppError(ErrorCode.PDF_PROCESSING_TIMEOUT, "PDF processing exceeded the time limit.")
            page = document.load_page(index)
            text = page.get_text() or ""
            total_chars += len(text)
            if total_chars > settings.max_extracted_chars:
                raise AppError(
                    ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                    "The PDF contains more text than the allowed limit.",
                )
            pages.append(ExtractedPage(page=index + 1, text=text))
    except AppError:
        raise
    except Exception as exc:
        logger.error("PDF extraction failed path=%s error=%s", pdf_path.name, exc)
        raise AppError(
            ErrorCode.PROCESSING_ERROR,
            "PDF text extraction failed.",
        ) from exc
    finally:
        document.close()

    return pages
