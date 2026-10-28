from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Union

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.schemas.document import ExtractedPage, ProcessedDocument
from app.services import docx_service
from app.services.chunk_service import create_chunks
from app.services.pdf_service import extract_pages, validate_pdf_path
from app.utils.text_utils import clean_text, sanitize_filename

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


def _process(
    kind: str,
    path: Union[str, Path],
    document_id: str,
    source_filename: Optional[str],
    validate: Callable[[str], None],
    extract: Callable[[], list[ExtractedPage]],
) -> ProcessedDocument:
    """Shared pipeline: validate -> extract -> clean -> chunk. Does not embed or index."""
    started = time.perf_counter()
    safe_name = sanitize_filename(source_filename or Path(path).name)

    try:
        validate(safe_name)
        t_stage = time.perf_counter()
        raw_pages = extract()
        metrics.observe_stage("pdf_extract", time.perf_counter() - t_stage)
        cleaned_pages = [
            ExtractedPage(page=page.page, text=clean_text(page.text))
            for page in raw_pages
        ]
        if all(not page.text for page in cleaned_pages):
            raise AppError(
                ErrorCode.NO_EXTRACTABLE_TEXT,
                f"The {kind} does not contain extractable text.",
            )

        t_stage = time.perf_counter()
        chunks = create_chunks(cleaned_pages, document_id, safe_name)
        metrics.observe_stage("chunking", time.perf_counter() - t_stage)
        if len(chunks) > get_settings().max_chunks_per_document:
            raise AppError(
                ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                f"The document produces more than the allowed {get_settings().max_chunks_per_document} chunks.",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Processed %s document_id=%s filename=%s page_count=%s "
            "chunk_count=%s processing_time_ms=%.1f",
            kind,
            document_id,
            safe_name,
            len(raw_pages),
            len(chunks),
            elapsed_ms,
        )
        return ProcessedDocument(
            document_id=document_id,
            filename=safe_name,
            pages=len(raw_pages),
            chunks=chunks,
            status="processed",
        )
    except AppError:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "%s processing failed document_id=%s filename=%s processing_time_ms=%.1f",
            kind.upper(),
            document_id,
            safe_name,
            elapsed_ms,
        )
        raise
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "Unexpected %s processing error document_id=%s filename=%s "
            "processing_time_ms=%.1f error=%s",
            kind.upper(),
            document_id,
            safe_name,
            elapsed_ms,
            exc,
        )
        raise AppError(
            ErrorCode.PROCESSING_ERROR,
            "Document processing failed.",
        ) from exc


def process_pdf(
    path: Union[str, Path],
    document_id: str,
    *,
    source_filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> ProcessedDocument:
    """Validate → extract → clean → chunk. Does not embed or index."""
    return _process(
        "PDF", path, document_id, source_filename,
        lambda _name: validate_pdf_path(
            path,
            max_bytes=max_bytes,
            declared_filename=source_filename or Path(path).name,
            content_type=content_type,
        ),
        lambda: extract_pages(path),
    )


def process_docx(
    path: Union[str, Path],
    document_id: str,
    *,
    source_filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> ProcessedDocument:
    """DOCX variant of :func:`process_pdf`; ``page`` metadata is a logical page (see docx_service)."""
    return _process(
        "DOCX", path, document_id, source_filename,
        lambda _name: docx_service.validate_docx_path(
            path,
            max_bytes=max_bytes,
            declared_filename=source_filename or Path(path).name,
            content_type=content_type,
        ),
        lambda: docx_service.extract_pages(path),
    )


def process_document(
    path: Union[str, Path],
    document_id: str,
    *,
    source_filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> ProcessedDocument:
    """Pick the processor from the (client-declared) extension; each validator still checks the real content."""
    name = source_filename or Path(path).name
    handler = process_docx if Path(sanitize_filename(name)).suffix.lower() == ".docx" else process_pdf
    return handler(path, document_id, source_filename=source_filename, content_type=content_type, max_bytes=max_bytes)
