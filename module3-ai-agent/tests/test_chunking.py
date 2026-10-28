from pathlib import Path

from app.core.config import get_settings
from app.schemas.document import ExtractedPage
from app.services.chunk_service import create_chunks
from app.services.document_service import process_pdf
from tests.pdf_helpers import long_text, write_pdf


def test_chunking_splits_long_text() -> None:
    pages = [ExtractedPage(page=1, text=long_text(400))]
    chunks = create_chunks(pages, document_id="doc-1", source="sample.pdf", chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_size_respected() -> None:
    settings = get_settings()
    pages = [ExtractedPage(page=1, text=long_text(500))]
    chunks = create_chunks(
        pages,
        document_id="doc-size",
        source="sample.pdf",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    oversized = [chunk for chunk in chunks if len(chunk.text) > settings.chunk_size]
    assert oversized == []


def test_chunk_overlap() -> None:
    pages = [ExtractedPage(page=1, text=long_text(200, prefix="token"))]
    chunks = create_chunks(
        pages,
        document_id="doc-overlap",
        source="sample.pdf",
        chunk_size=80,
        chunk_overlap=25,
    )
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        prev_words = previous.text.split()
        next_words = current.text.split()
        overlap_words = set(prev_words[-8:]) & set(next_words[:12])
        assert overlap_words, "Expected shared tokens between consecutive chunks"


def test_chunk_metadata_single_page() -> None:
    pages = [ExtractedPage(page=5, text=long_text(40))]
    chunks = create_chunks(pages, document_id="doc-meta", source="example.pdf", chunk_size=120, chunk_overlap=20)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata.source == "example.pdf"
        assert chunk.metadata.page == 5
        assert chunk.metadata.pages is None


def test_chunk_metadata_multi_page() -> None:
    pages = [
        ExtractedPage(page=5, text=long_text(60, prefix="alpha")),
        ExtractedPage(page=6, text=long_text(60, prefix="beta")),
    ]
    chunks = create_chunks(pages, document_id="doc-pages", source="example.pdf", chunk_size=200, chunk_overlap=40)
    used_pages = set()
    for chunk in chunks:
        if chunk.metadata.page is not None:
            used_pages.add(chunk.metadata.page)
        if chunk.metadata.pages:
            used_pages.update(chunk.metadata.pages)
    assert used_pages == {5, 6}


def test_unique_chunk_ids() -> None:
    pages = [ExtractedPage(page=1, text=long_text(300))]
    chunks = create_chunks(pages, document_id="doc-ids", source="sample.pdf", chunk_size=90, chunk_overlap=20)
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids
    assert len(ids) == len(set(ids))


def test_document_id_is_preserved() -> None:
    document_id = "11111111-2222-3333-4444-555555555555"
    pages = [ExtractedPage(page=1, text="Gradient descent minimises a loss function.")]
    chunks = create_chunks(pages, document_id=document_id, source="lecture.pdf")
    assert chunks
    assert all(chunk.document_id == document_id for chunk in chunks)


def test_filename_is_not_used_as_document_id(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "example.pdf", ["Decision trees split features to reduce impurity."])
    result = process_pdf(
        pdf_path,
        document_id="kept-document-id",
        source_filename="example.pdf",
    )
    assert result.document_id == "kept-document-id"
    assert result.filename == "example.pdf"
    assert all(chunk.document_id == "kept-document-id" for chunk in result.chunks)
    assert all(chunk.document_id != "example.pdf" for chunk in result.chunks)
