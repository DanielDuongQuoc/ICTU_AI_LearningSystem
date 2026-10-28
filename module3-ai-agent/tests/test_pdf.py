from pathlib import Path

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.services.document_service import process_pdf
from app.services.pdf_service import extract_pages, validate_pdf_path
from app.utils.text_utils import clean_text
from tests.pdf_helpers import write_pdf


def test_valid_pdf(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "valid.pdf", ["Machine learning is a field of AI."])
    pages = extract_pages(pdf_path)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "Machine learning" in pages[0].text


def test_file_is_not_pdf(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("this is not a pdf", encoding="utf-8")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(txt_path)
    assert exc.value.code == ErrorCode.INVALID_FILE


def test_file_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(AppError) as exc:
        validate_pdf_path(missing)
    assert exc.value.code == ErrorCode.FILE_NOT_FOUND


def test_empty_pdf_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(empty)
    assert exc.value.code == ErrorCode.EMPTY_FILE


def test_pdf_without_extractable_text(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "blank.pdf", [""])
    with pytest.raises(AppError) as exc:
        process_pdf(pdf_path, document_id="doc-blank")
    assert exc.value.code == ErrorCode.NO_EXTRACTABLE_TEXT


def test_extract_keeps_page_numbers(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "paged.pdf",
        ["Page one content.", "Page two content.", "Page three content."],
    )
    pages = extract_pages(pdf_path)
    assert [page.page for page in pages] == [1, 2, 3]
    assert "one" in pages[0].text
    assert "two" in pages[1].text
    assert "three" in pages[2].text


def test_text_cleaning() -> None:
    raw = "Machine   Learning\n\n\nis a branch\nof AI."
    assert clean_text(raw) == "Machine Learning\n\nis a branch of AI."


def test_invalid_pdf_bytes(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"%PDF-not-really")
    with pytest.raises(AppError) as exc:
        extract_pages(fake)
    assert exc.value.code in {ErrorCode.INVALID_PDF, ErrorCode.PROCESSING_ERROR}


def test_file_too_large(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "big.pdf", ["tiny"])
    with pytest.raises(AppError) as exc:
        validate_pdf_path(pdf_path, max_bytes=10)
    assert exc.value.code == ErrorCode.FILE_TOO_LARGE


def test_spoofed_pdf_extension(tmp_path: Path) -> None:
    spoofed = tmp_path / "malware.pdf"
    spoofed.write_bytes(b"MZ this is not a pdf")
    with pytest.raises(AppError) as exc:
        validate_pdf_path(spoofed, declared_filename="malware.pdf")
    assert exc.value.code == ErrorCode.INVALID_FILE


def test_multi_page_pdf_processing(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "multi.pdf",
        [
            "Introduction to neural networks and supervised learning.",
            "Backpropagation updates weights using gradient descent.",
            "Regularisation reduces overfitting in deep models.",
        ],
    )
    result = process_pdf(pdf_path, document_id="doc-multi", source_filename="multi.pdf")
    assert result.pages == 3
    assert result.status == "processed"
    assert result.document_id == "doc-multi"
    assert len(result.chunks) >= 1


@pytest.mark.asyncio
async def test_upload_valid_pdf(client, tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    pdf_path = write_pdf(tmp_path / "upload.pdf", ["Supervised learning maps inputs to labels."])
    files = {"file": ("lecture.pdf", pdf_path.read_bytes(), "application/pdf")}

    fake_svc = MagicMock()
    fake_svc.index_chunks = AsyncMock(return_value={"chunk_count": 1})
    with patch("app.api.documents._get_vector_service", return_value=fake_svc):
        response = await client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "lecture.pdf"
    assert data["pages"] == 1
    assert data["chunks"] >= 1
    assert data["status"] == "indexed"
    assert data["document_id"]


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(client) -> None:
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    response = await client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == ErrorCode.INVALID_FILE
    assert "traceback" not in body["error"]["message"].lower()
    assert "Traceback" not in str(body)
