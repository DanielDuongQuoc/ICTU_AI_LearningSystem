"""DOCX support (Module 1 uploads PDF *and* Word).

Pipeline contract is identical to ``pdf_service``: ``validate_docx_path`` then ``extract_pages`` returning
``ExtractedPage(page, text)`` so chunking, embedding, indexing, RAG and source attribution are shared.

Word files do not store page numbers, so ``page`` is a *logical* page:

* an explicit page break (``<w:br w:type="page"/>``) or a ``<w:lastRenderedPageBreak/>`` written by Word starts a new page;
* a page whose text exceeds ``LOGICAL_PAGE_CHARS`` is split at paragraph boundaries.

Therefore ``source.page`` for a DOCX is stable and traceable, but may differ from the page number Word displays.
Tables are extracted row by row (cells joined with `` | ``). Headers, footers, comments, text boxes, images and
embedded objects are NOT extracted. Only ``.docx`` is accepted (legacy ``.doc`` and macro-enabled ``.docm`` are rejected).

Safety: the file must be a ZIP with the OOXML signature, the uncompressed size is bounded (zip-bomb guard), page and
character limits and the processing deadline are the same settings as for PDF.
"""

from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path
from typing import Optional, Union

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.schemas.document import ExtractedPage

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ZIP_MAGIC = b"PK\x03\x04"
LOGICAL_PAGE_CHARS = 3000
UNCOMPRESSED_FACTOR = 10  # uncompressed OOXML parts may be at most 10x the allowed upload size
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ALLOWED_MIME = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
    "binary/octet-stream",
    "application/zip",
    "application/x-zip-compressed",
}


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def validate_docx_path(
    path: PathLike,
    *,
    max_bytes: Optional[int] = None,
    declared_filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Path:
    """Validate a DOCX on disk. Does not trust client-supplied names or MIME types."""
    docx_path = _as_path(path)
    limit = max_bytes if max_bytes is not None else get_settings().max_upload_size_bytes

    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime and mime not in ALLOWED_MIME:
            raise AppError(ErrorCode.INVALID_FILE, "File MIME type is not DOCX.")
    if not docx_path.exists():
        raise AppError(ErrorCode.FILE_NOT_FOUND, "File was not found.")
    if not docx_path.is_file():
        raise AppError(ErrorCode.INVALID_FILE, "Path is not a file.")
    size = docx_path.stat().st_size
    if size == 0:
        raise AppError(ErrorCode.EMPTY_FILE, "File is empty.")
    if size > limit:
        raise AppError(ErrorCode.FILE_TOO_LARGE, f"File exceeds the maximum upload size of {limit} bytes.")
    if Path(declared_filename or docx_path.name).suffix.lower() != ".docx":
        raise AppError(ErrorCode.INVALID_FILE, "Only .docx Word files are accepted (not .doc or macro-enabled files).")

    with docx_path.open("rb") as handle:
        if handle.read(4) != ZIP_MAGIC:
            raise AppError(ErrorCode.INVALID_FILE, "File content is not a DOCX.")
    try:
        with zipfile.ZipFile(docx_path) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                raise AppError(ErrorCode.INVALID_FILE, "File content is not a DOCX.")
            if any(n.lower().endswith("vbaproject.bin") for n in names):
                raise AppError(ErrorCode.INVALID_FILE, "Macro-enabled Word files are not accepted.")
            if sum(i.file_size for i in archive.infolist()) > limit * UNCOMPRESSED_FACTOR:
                raise AppError(ErrorCode.DOCUMENT_LIMIT_EXCEEDED, "The DOCX expands beyond the allowed size.")
    except zipfile.BadZipFile as exc:
        raise AppError(ErrorCode.INVALID_FILE, "File content is not a DOCX.") from exc
    return docx_path


def _has_page_break(paragraph) -> tuple[bool, bool]:
    """Return (break_before_text, break_after_text) for a paragraph element."""
    before = after = False
    seen_text = False
    for el in paragraph.iter():
        tag = el.tag
        if tag == f"{_W}t" and (el.text or "").strip():
            seen_text = True
        elif tag == f"{_W}lastRenderedPageBreak" or (tag == f"{_W}br" and el.get(f"{_W}type") == "page"):
            if seen_text:
                after = True
            else:
                before = True
    return before, after


def _paragraph_text(paragraph) -> str:
    parts: list[str] = []
    for el in paragraph.iter():
        if el.tag == f"{_W}t" and el.text:
            parts.append(el.text)
        elif el.tag == f"{_W}tab":
            parts.append(" ")
        elif el.tag == f"{_W}br" and el.get(f"{_W}type") != "page":
            parts.append("\n")
    return "".join(parts).strip()


def _table_rows(table) -> list[str]:
    rows: list[str] = []
    for tr in table.iter(f"{_W}tr"):
        cells = []
        for tc in tr.findall(f"{_W}tc"):
            cell = " ".join(t for t in (_paragraph_text(p) for p in tc.iter(f"{_W}p")) if t)
            if cell:
                cells.append(cell)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def extract_pages(path: PathLike) -> list[ExtractedPage]:
    """Extract logical pages from a DOCX in document order (paragraphs and tables)."""
    import docx  # python-docx; imported lazily so PDF-only deployments of the module stay light

    docx_path = _as_path(path)
    settings = get_settings()
    try:
        document = docx.Document(str(docx_path))
    except Exception as exc:
        logger.error("Failed to open DOCX path=%s error=%s", docx_path.name, type(exc).__name__)
        raise AppError(ErrorCode.INVALID_PDF, "The file is not a valid DOCX.") from exc

    deadline = time.monotonic() + settings.pdf_processing_timeout_s
    pages: list[list[str]] = [[]]
    total_chars = 0

    def add(block: str) -> None:
        nonlocal total_chars
        total_chars += len(block)
        if total_chars > settings.max_extracted_chars:
            raise AppError(ErrorCode.DOCUMENT_LIMIT_EXCEEDED, "The document contains more text than the allowed limit.")
        pages[-1].append(block)

    def new_page() -> None:
        if pages[-1]:
            pages.append([])
            if len(pages) > settings.max_pdf_pages:
                raise AppError(ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                               f"The document has more than the allowed {settings.max_pdf_pages} pages.")

    try:
        for element in document.element.body.iterchildren():
            if time.monotonic() > deadline:
                raise AppError(ErrorCode.PDF_PROCESSING_TIMEOUT, "Document processing exceeded the time limit.")
            if element.tag == f"{_W}p":
                before, after = _has_page_break(element)
                if before:
                    new_page()
                text = _paragraph_text(element)
                if text:
                    add(text)
                if after:
                    new_page()
            elif element.tag == f"{_W}tbl":
                for row in _table_rows(element):
                    add(row)
    except AppError:
        raise
    except Exception as exc:
        logger.error("DOCX extraction failed path=%s error=%s", docx_path.name, type(exc).__name__)
        raise AppError(ErrorCode.PROCESSING_ERROR, "DOCX text extraction failed.") from exc

    out: list[ExtractedPage] = []
    for blocks in pages:
        if not blocks:
            continue
        # split an over-long logical page at paragraph boundaries
        current: list[str] = []
        size = 0
        for block in blocks:
            if current and size + len(block) > LOGICAL_PAGE_CHARS:
                out.append(ExtractedPage(page=len(out) + 1, text="\n\n".join(current)))
                current, size = [], 0
            current.append(block)
            size += len(block) + 2
        if current:
            out.append(ExtractedPage(page=len(out) + 1, text="\n\n".join(current)))
    if len(out) > settings.max_pdf_pages:
        raise AppError(ErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                       f"The document has more than the allowed {settings.max_pdf_pages} pages.")
    if not out:
        raise AppError(ErrorCode.NO_EXTRACTABLE_TEXT, "The document does not contain extractable text.")
    return out
