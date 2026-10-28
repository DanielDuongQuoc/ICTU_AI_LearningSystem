"""Path traversal: client-supplied ids/filenames must never reach the filesystem (Phase 19)."""

from __future__ import annotations

from urllib.parse import quote

import pytest

from tests.security.conftest import assert_safe_error, eval_body, upload_text, TEXT_A

EVIL = [
    "../../secret.txt", "..\\..\\secret.txt", "/etc/passwd", "C:\\Windows\\System32\\config",
    "....//....//x", "%2e%2e%2f%2e%2e%2fsecret", "doc\x00.pdf", "..", ".",
]
CLEAN = {"question": "zebracorn data"}


@pytest.mark.parametrize("evil", EVIL)
async def test_path_param_routes_reject_traversal(client, stack, evil):
    seg = quote(evil, safe="")
    for method, url, kw in (
        ("GET", f"/api/documents/{seg}", {}),
        ("DELETE", f"/api/documents/{seg}", {}),
        ("POST", f"/api/documents/{seg}/search", {"json": {"query": "x"}}),
    ):
        resp = await client.request(method, url, **kw)
        assert resp.status_code in (400, 404, 405, 422), (method, url, resp.status_code, resp.text)
        if resp.headers.get("content-type", "").startswith("application/json") and resp.status_code != 405:
            assert_safe_error(resp)


@pytest.mark.parametrize("evil", EVIL)
async def test_body_document_ids_reject_traversal(client, stack, evil):
    for url, body in (
        ("/api/rag/retrieve", {"document_id": evil, **CLEAN}),
        ("/api/rag/answer", {"document_id": evil, **CLEAN}),
        ("/api/questions/generate", {"document_id": evil, "number_of_questions": 1}),
        ("/api/evaluations/evaluate", eval_body(evil)),
    ):
        resp = await client.post(url, json=body)
        assert resp.status_code in (400, 404, 422), (url, resp.status_code, resp.text)
        assert_safe_error(resp)


async def test_traversal_id_cannot_widen_a_real_documents_scope(client, stack):
    real = await upload_text(client, TEXT_A)
    for evil in (f"{real}/../{real}", f"{real}\\..", f"../{real}", f"{real}%00", f"{real}' OR '1'='1"):
        r = await client.post("/api/rag/retrieve", json={"document_id": evil, **CLEAN})
        assert r.status_code in (400, 404, 422), (evil, r.status_code)
    r = await client.post("/api/rag/retrieve", json={"document_id": real, **CLEAN})
    assert r.status_code == 200


async def test_chroma_filter_cannot_be_injected_via_id(client, stack):
    real = await upload_text(client, TEXT_A)
    for evil in ({"$ne": "x"}, {"$or": [{"document_id": real}]}, ["a", real]):
        r = await client.post("/api/rag/retrieve", json={"document_id": evil, **CLEAN})
        assert r.status_code == 422, r.text
