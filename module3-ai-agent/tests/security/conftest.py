"""Shared fixtures for the Phase 19 security tests (real API + real in-memory ChromaDB + recording fake LLM)."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from tests.fakes import Stack, sample_pdf_bytes

TEXT_A = ("Zebracorn systems explain machine learning. Zebracorn models learn zebracorn patterns "
          "from data and make predictions about data.")
TEXT_B = ("Quokkaplex networks describe database indexing. Quokkaplex indexes speed up database "
          "queries and database lookups for data.")

BAD_MARKERS = ("Traceback", "site-packages", "File \"", "module3-ai-agent", "C:\\Users", "/home/")


@pytest.fixture()
def stack(monkeypatch):
    return Stack().install(monkeypatch)


@pytest.fixture()
def settings():
    return get_settings()


@pytest.fixture()
def set_limit(monkeypatch, settings):
    def _set(**kw):
        for key, value in kw.items():
            monkeypatch.setattr(settings, key, value)
    return _set


async def upload_text(client, text: str, name: str = "doc.pdf") -> str:
    resp = await client.post("/api/documents/upload",
                             files={"file": (name, sample_pdf_bytes([text]), "application/pdf")})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["document_id"]


def assert_safe_error(resp, *, status=None, code=None) -> dict:
    """Uniform error schema, no internals, no stack trace."""
    if status is not None:
        assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    err = body["error"]
    assert {"code", "message", "request_id"} <= set(err)
    if code is not None:
        assert err["code"] == code, body
    for marker in BAD_MARKERS:
        assert marker not in resp.text, f"{marker!r} leaked in {resp.text}"
    return err


def eval_body(doc_id: str, answer: str = "Zebracorn models learn from data.") -> dict:
    return {"document_id": doc_id, "question": "What do zebracorn models learn from?",
            "student_answer": answer, "expected_answer": "They learn patterns from data.", "max_score": 10}
