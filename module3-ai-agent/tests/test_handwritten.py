"""Handwritten submissions: page preparation (masking, limits), transcription parsing/flags, API, and the
inline-image Gemini payload."""

from __future__ import annotations

import base64
import json
from typing import Any

import fitz
import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.services.transcription_service import (
    TranscriptionService, parse_transcription, prepare_pages, transcription_flags,
)
from tests.fakes import gemini_llm, gemini_ok
from tests.test_grading import FakeLLM


def png_page(w: int = 400, h: int = 600, gray: int = 0) -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h), False)
    pix.set_rect(pix.irect, (gray, gray, gray))
    return pix.tobytes("png")


def pdf_pages(n: int) -> bytes:
    doc = fitz.open()
    for i in range(n):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def pixel(png: bytes, x: int, y: int) -> tuple[int, ...]:
    return fitz.Pixmap(png).pixel(x, y)


class ImageLLM(FakeLLM):
    async def generate_json(self, system_prompt, user_prompt, *, required_keys=None, images=None, **kw):
        self.images = images
        return await super().generate_json(system_prompt, user_prompt, required_keys=required_keys)


def tr(*texts: str, conf: str = "high", diagram: bool = False) -> dict:
    return {"pages": [{"page": i + 1, "text": t, "confidence": conf, "diagram_present": diagram}
                      for i, t in enumerate(texts)]}


# ---------------------------------------------------------------- preparation

def test_header_of_first_page_is_masked_only():
    pages = prepare_pages([png_page(), png_page()], mask_header_ratio=0.2)
    first, second = pages[0][1], pages[1][1]
    w = fitz.Pixmap(first).width
    assert pixel(first, w // 2, 5) == (255, 255, 255)          # header blanked (name / MSSV)
    assert pixel(first, w // 2, fitz.Pixmap(first).height - 5) == (0, 0, 0)
    assert pixel(second, w // 2, 5) == (0, 0, 0)               # other pages untouched
    assert all(m == "image/png" for m, _ in pages)


def test_pdf_pages_are_rendered_and_downscaled():
    pages = prepare_pages([pdf_pages(3)], mask_header_ratio=0)
    assert len(pages) == 3
    assert max(fitz.Pixmap(pages[0][1]).width, fitz.Pixmap(pages[0][1]).height) <= 1600


def test_rejects_unknown_format_and_too_many_pages(monkeypatch):
    with pytest.raises(AppError) as e:
        prepare_pages([b"GIF89a...."])
    assert e.value.code == ErrorCode.INVALID_FILE
    monkeypatch.setenv("MAX_HANDWRITTEN_PAGES", "2")
    get_settings.cache_clear()
    try:
        with pytest.raises(AppError) as e:
            prepare_pages([pdf_pages(3)])
        assert e.value.code == ErrorCode.DOCUMENT_LIMIT_EXCEEDED
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- parsing / flags

def test_parse_requires_one_item_per_page():
    with pytest.raises(AppError) as e:
        parse_transcription(tr("a"), 2)
    assert e.value.code == ErrorCode.TRANSCRIPTION_INVALID_RESPONSE
    pages = parse_transcription(tr("Giao dịch là [?] đơn vị", "ACID"), 2)
    assert pages[0].unreadable_count == 1 and pages[1].page == 2


@pytest.mark.parametrize("raw,flags", [
    (tr("Giao dịch là một đơn vị công việc nguyên tử đảm bảo ACID."), []),
    (tr("Giao dịch là đơn vị công việc", conf="low"), ["unreadable"]),
    (tr("[?] [?] [?] [?] là [?]"), ["unreadable"]),
    (tr("Sơ đồ ER [Hình: hai bảng nối khóa ngoại]", diagram=True), ["diagram_present"]),
    (tr(""), ["unreadable"]),
])
def test_flags(raw, flags):
    assert transcription_flags(parse_transcription(raw, 1)) == flags


# ---------------------------------------------------------------- service / API

async def test_service_sends_masked_images_and_returns_joined_text():
    llm = ImageLLM(tr("Trang một", "Trang hai"))
    res = await TranscriptionService(llm_service=llm).transcribe([png_page(), png_page()], question="Câu 1")
    assert res.text == "Trang một\n\nTrang hai" and res.flags == []
    assert len(llm.images) == 2 and "NUMBER OF PAGES: 2" in llm.calls[0][1] and "Câu 1" in llm.calls[0][1]


async def test_invalid_output_retried_once():
    llm = ImageLLM(tr("only one"))
    with pytest.raises(AppError) as e:
        await TranscriptionService(llm_service=llm).transcribe([png_page(), png_page()])
    assert e.value.code == ErrorCode.TRANSCRIPTION_INVALID_RESPONSE and len(llm.calls) == 2


async def test_api_multipart_and_body_limit_exemption(client, monkeypatch):
    llm = ImageLLM(tr("Bài làm viết tay"))
    monkeypatch.setattr("app.api.submissions._get_transcription_service",
                        lambda: TranscriptionService(llm_service=llm))
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1000")      # page images are bigger than the JSON body cap
    get_settings.cache_clear()
    try:
        big = png_page(1200, 1600, gray=30)
        assert len(big) > 1000
        r = await client.post("/api/submissions/handwritten/transcribe",
                              files=[("files", ("p1.png", big, "image/png"))], data={"question": "Câu 1"})
    finally:
        get_settings.cache_clear()
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "Bài làm viết tay" and r.json()["trace"]["prompt_version"] == "handwriting_transcription:v1"

    r = await client.post("/api/submissions/handwritten/transcribe",
                          files=[("files", ("p1.png", png_page(), "image/png"))], data={"mask_header_ratio": "0.9"})
    assert r.status_code == 422


async def test_gemini_payload_carries_inline_images():
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return gemini_ok(json.dumps(tr("x")))

    svc = gemini_llm(handler)
    await svc.generate_json("sys", "user", required_keys=["pages"], images=[("image/png", b"\x89PNGdata")])
    parts = seen[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "user"}
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(b"\x89PNGdata").decode()}}
    await svc.generate_json("sys", "user2")
    assert seen[1]["contents"][0]["parts"] == [{"text": "user2"}]     # text-only calls unchanged
    await svc.close()
