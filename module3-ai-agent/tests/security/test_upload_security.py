"""Upload security: type/size/limits/malformed PDFs/filenames/concurrency (Phase 19)."""

from __future__ import annotations

import re
import uuid

import pytest

from app.core.rate_limit import upload_slot
from tests.fakes import sample_pdf_bytes
from tests.security.conftest import TEXT_A, assert_safe_error

PDF = sample_pdf_bytes([TEXT_A])


async def _post(client, name, data, ctype="application/pdf"):
    return await client.post("/api/documents/upload", files={"file": (name, data, ctype)})


@pytest.mark.parametrize("name,data,ctype", [
    ("a.txt", PDF, "application/pdf"),                              # wrong extension
    ("a.pdf", PDF, "text/plain"),                                   # wrong MIME
    ("a.pdf", b"just some text, renamed to pdf", "application/pdf"),  # not a PDF at all
    ("a.pdf", b"", "application/pdf"),                              # empty
    ("a.pdf", b"%PDF-1.7\n" + b"\x00\xff garbage " * 50, "application/pdf"),  # signature ok, body junk
    ("a.pdf", PDF[: len(PDF) // 2], "application/pdf"),             # truncated PDF
    ("a.exe", b"MZ\x90\x00", "application/octet-stream"),           # executable
])
async def test_invalid_uploads_rejected_safely(client, stack, name, data, ctype):
    resp = await _post(client, name, data, ctype)
    assert 400 <= resp.status_code < 500, resp.text
    assert_safe_error(resp)


async def test_oversized_upload_rejected(client, stack, set_limit):
    set_limit(max_upload_size_mb=1)
    resp = await _post(client, "big.pdf", b"%PDF-1.7\n" + b"0" * (1024 * 1024 + 10))
    assert_safe_error(resp, status=413, code="FILE_TOO_LARGE")


async def test_too_many_pages_rejected(client, stack, set_limit):
    set_limit(max_pdf_pages=2)
    resp = await _post(client, "p.pdf", sample_pdf_bytes(["one", "two", "three"]))
    assert_safe_error(resp, status=413, code="DOCUMENT_LIMIT_EXCEEDED")


async def test_page_limit_boundary_accepted(client, stack, set_limit):
    set_limit(max_pdf_pages=2)
    resp = await _post(client, "p.pdf", sample_pdf_bytes([TEXT_A, "second page of text"]))
    assert resp.status_code == 200, resp.text


async def test_abnormal_text_volume_rejected(client, stack, set_limit):
    set_limit(max_extracted_chars=50)
    resp = await _post(client, "t.pdf", PDF)
    assert_safe_error(resp, status=413, code="DOCUMENT_LIMIT_EXCEEDED")


async def test_too_many_chunks_rejected(client, stack, set_limit):
    set_limit(max_chunks_per_document=0)  # any chunk exceeds the cap
    resp = await _post(client, "c.pdf", sample_pdf_bytes([f"{TEXT_A} page {i}" for i in range(4)]))
    assert_safe_error(resp, status=413, code="DOCUMENT_LIMIT_EXCEEDED")


async def test_pdf_processing_timeout_is_controlled(client, stack, set_limit):
    set_limit(pdf_processing_timeout_s=-1.0)  # deadline already passed
    resp = await _post(client, "t.pdf", PDF)
    assert_safe_error(resp, status=504, code="PDF_PROCESSING_TIMEOUT")


@pytest.mark.parametrize("name", [
    "../../etc/passwd.pdf", "..\\..\\windows\\system32\\x.pdf", "/abs/path/x.pdf", "C:\\Windows\\x.pdf",
    "a\x00b.pdf", "tÃªn-tá»‡p-Ã¼nÃ¯-äººå·¥.pdf", "....//....//x.pdf", "CON.pdf", " .pdf", "x" * 400 + ".pdf",
])
async def test_dangerous_filenames_never_become_paths(client, stack, name):
    resp = await _post(client, name, PDF)
    assert resp.status_code < 500, resp.text
    if resp.status_code == 200:
        body = resp.json()
        uuid.UUID(body["document_id"])                  # server-generated id, never the filename
        fn = body["filename"]
        assert "/" not in fn and "\\" not in fn and ".." not in fn and "\x00" not in fn
        assert len(fn) <= 255
        assert re.fullmatch(r"[0-9a-f-]{36}", body["document_id"])
    else:
        assert_safe_error(resp)


async def test_same_filename_twice_gets_distinct_ids(client, stack):
    a = await _post(client, "same.pdf", PDF)
    b = await _post(client, "same.pdf", PDF)
    assert a.json()["document_id"] != b.json()["document_id"]


async def test_upload_gate_rejects_when_full(client, stack, set_limit):
    set_limit(max_concurrent_uploads=1)
    async with upload_slot():
        resp = await _post(client, "x.pdf", PDF)
        assert_safe_error(resp, status=503, code="SERVICE_OVERLOADED")
    assert (await _post(client, "x.pdf", PDF)).status_code == 200  # slot released
