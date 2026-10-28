"""Document API: upload + index, info, delete, per-document semantic search."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.openapi import error_responses
from app.core.rate_limit import rate_limit, upload_slot
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentInfoResponse,
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.services.document_service import process_docx, process_pdf
from app.services.vector_service import VectorService
from app.utils.text_utils import sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


def _get_vector_service() -> VectorService:
    return VectorService()


# ── Upload + Index ──────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and index a PDF or Word (.docx) document",
    description=(
        "Multipart upload (field `file`): `.pdf` or `.docx`. The file is validated (extension, MIME type, `%PDF` / OOXML signature, "
        "size limit MAX_UPLOAD_SIZE_MB), text is extracted and cleaned, split into chunks, embedded "
        "with the configured Ollama embedding model and indexed in ChromaDB. Returns the new `document_id`."
    ),
    responses=error_responses(400, 413, 422, 429, 500, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    settings = get_settings()
    # Cheap early rejection from Content-Length (multipart overhead makes this a
    # lower bound, so it never rejects a valid file).  The streamed check below
    # is authoritative — the header is never trusted on its own.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_size_bytes:
        raise AppError(
            ErrorCode.FILE_TOO_LARGE,
            f"File exceeds the maximum upload size of {settings.max_upload_size_mb} MB.",
        )
    async with upload_slot():
        return await _handle_upload(file, settings)


def _discard_partial_index(vector_svc, document_id: str) -> None:
    """Best-effort compensation: no half-indexed document may survive a failed upload."""
    try:
        removed = vector_svc.delete_document(document_id)
        if removed:
            metrics.event("index", "cleaned_up")
            logger.warning("Removed partial index document_id=%s chunks=%s", document_id, removed)
    except Exception as exc:  # cleanup must never mask the original failure
        logger.error("Cleanup of partial index failed document_id=%s: %s", document_id, exc)


async def _handle_upload(file: UploadFile, settings) -> DocumentUploadResponse:
    document_id = str(uuid.uuid4())
    original_name = file.filename or "document.pdf"
    safe_name = sanitize_filename(original_name)
    is_docx = Path(safe_name).suffix.lower() == ".docx"
    if not is_docx and Path(safe_name).suffix.lower() != ".pdf":
        safe_name = f"{safe_name}.pdf"

    tmp_dir = Path(tempfile.mkdtemp(prefix="m3_doc_"))
    tmp_path = tmp_dir / f"{document_id}{'.docx' if is_docx else '.pdf'}"
    try:
        size = 0
        with tmp_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_size_bytes:
                    raise AppError(
                        ErrorCode.FILE_TOO_LARGE,
                        f"File exceeds the maximum upload size of "
                        f"{settings.max_upload_size_mb} MB.",
                    )
                handle.write(chunk)

        if size == 0:
            raise AppError(ErrorCode.EMPTY_FILE, "File is empty.")

        # Phase 2: extract + chunk (CPU/IO bound -> worker thread, keeps the loop free)
        try:
            result = await asyncio.to_thread(
                process_docx if is_docx else process_pdf,
                tmp_path,
                document_id,
                source_filename=original_name,
                content_type=file.content_type,
                max_bytes=settings.max_upload_size_bytes,
            )
        except AppError as exc:
            metrics.event("document_processing", {
                ErrorCode.PDF_PROCESSING_TIMEOUT: "timeout",
                ErrorCode.DOCUMENT_LIMIT_EXCEEDED: "limit_exceeded",
            }.get(exc.code, "failed"))
            raise
        metrics.event("document_processing", "ok")

        # Phase 3: embed + index into ChromaDB.  A document is only reported "indexed" when the vectors
        # are really there; any failure (or a client disconnect) after this point removes whatever was written.
        vector_svc = _get_vector_service()
        try:
            await vector_svc.index_chunks(
                chunks=result.chunks,
                document_id=document_id,
                source=result.filename,
            )
            stored = vector_svc.get_document_info(document_id).get("chunk_count")
            if isinstance(stored, int) and stored != len(result.chunks):
                metrics.event("index", "verification_failed")
                raise AppError(
                    ErrorCode.INDEXING_ERROR,
                    "Indexing could not be verified; the document was not stored.",
                )
        except BaseException:
            _discard_partial_index(vector_svc, document_id)
            metrics.event("index", "failed")
            raise
        metrics.event("index", "ok")

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            pages=result.pages,
            chunks=len(result.chunks),
            status="indexed",
        )
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Get document info ───────────────────────────────────────────────


@router.get(
    "/{document_id}",
    response_model=DocumentInfoResponse,
    summary="Get document metadata",
    description="Returns the indexed chunk count and source filename for a document.",
    responses=error_responses(404, 503),
)
async def get_document(document_id: str) -> DocumentInfoResponse:
    vector_svc = _get_vector_service()
    info = vector_svc.get_document_info(document_id)

    if info["chunk_count"] == 0:
        raise AppError(
            ErrorCode.DOCUMENT_NOT_FOUND,
            f"Document '{document_id}' not found.",
        )

    return DocumentInfoResponse(
        document_id=info["document_id"],
        filename=info.get("source", ""),
        pages=0,  # page count not stored in ChromaDB; kept for schema compat
        chunks=info["chunk_count"],
        status="indexed",
    )


# ── Delete document ─────────────────────────────────────────────────


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and all its indexed chunks",
    description="Removes every chunk of the document from ChromaDB. Other documents are untouched.",
    responses=error_responses(404, 503),
)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    vector_svc = _get_vector_service()

    if not vector_svc.document_exists(document_id):
        raise AppError(
            ErrorCode.DOCUMENT_NOT_FOUND,
            f"Document '{document_id}' not found.",
        )

    deleted = vector_svc.delete_document(document_id)
    return DocumentDeleteResponse(
        document_id=document_id,
        deleted_chunks=deleted,
        message=f"Document and {deleted} chunks deleted successfully.",
    )


# ── Search within document ─────────────────────────────────────────


@router.post(
    "/{document_id}/search",
    response_model=SearchResponse,
    summary="Semantic search within a document",
    description="Embeds the query and returns the most similar chunks of this document only.",
    responses=error_responses(400, 404, 422, 429, 502, 503, 504),
    dependencies=[Depends(rate_limit())],
)
async def search_document(
    document_id: str,
    body: SearchRequest,
) -> SearchResponse:
    settings = get_settings()
    if len(body.query.strip()) > settings.max_question_chars:
        raise AppError(ErrorCode.INPUT_TOO_LONG, f"query must be at most {settings.max_question_chars} characters.")
    if body.top_k is not None and body.top_k > settings.max_top_k:
        raise AppError(ErrorCode.INVALID_TOP_K, f"top_k must be between 1 and {settings.max_top_k}.")
    vector_svc = _get_vector_service()

    if not vector_svc.document_exists(document_id):
        raise AppError(
            ErrorCode.DOCUMENT_NOT_FOUND,
            f"Document '{document_id}' not found.",
        )

    result = await vector_svc.search(
        query=body.query,
        document_id=document_id,
        top_k=body.top_k,
    )

    items = [SearchResultItem(**r) for r in result["results"]]
    return SearchResponse(results=items)
