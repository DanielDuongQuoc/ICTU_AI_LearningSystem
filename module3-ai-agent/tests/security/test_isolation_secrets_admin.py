"""Document isolation, secret/log privacy, repository hygiene and admin-plane security (Phase 19)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.services.ai_version_service import reset_ai_version_service
from tests.security.conftest import TEXT_A, TEXT_B, assert_safe_error, eval_body, upload_text

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "security-test-admin-token-VALUE-123"
H = {"X-Admin-Token": TOKEN}


# ---------------------------------------------------------------- isolation / deletion
async def test_every_read_path_is_scoped_to_one_document(client, stack):
    a = await upload_text(client, TEXT_A, "a.pdf")
    b = await upload_text(client, TEXT_B, "b.pdf")
    r = await client.post(f"/api/documents/{a}/search", json={"query": "quokkaplex database indexing"})
    assert r.status_code == 200
    assert "quokkaplex" not in r.text.lower()
    r = await client.post("/api/rag/retrieve", json={"document_id": a, "question": "quokkaplex database"})
    assert all(c["document_id"] == a for c in r.json()["chunks"])
    stack.llm.calls.clear()
    await client.post("/api/evaluations/evaluate", json={**eval_body(a), "question": "What is quokkaplex indexing?"})
    assert "quokkaplex" not in stack.llm.prompts.lower().replace("quokkaplex indexing?", "")
    assert b != a


async def test_deleting_a_document_removes_its_vectors_and_only_its_vectors(client, stack):
    a = await upload_text(client, TEXT_A, "a.pdf")
    b = await upload_text(client, TEXT_B, "b.pdf")
    assert (await client.delete(f"/api/documents/{a}")).status_code == 200
    assert_safe_error(await client.get(f"/api/documents/{a}"), status=404)
    assert_safe_error(await client.post("/api/rag/retrieve", json={"document_id": a, "question": "zebracorn"}), status=404)
    assert (await client.get(f"/api/documents/{b}")).status_code == 200
    assert (await client.delete(f"/api/documents/{a}")).status_code == 404       # idempotent-safe, no 500


async def test_evaluating_against_a_deleted_document_does_not_call_the_llm(client, stack):
    a = await upload_text(client, TEXT_A)
    await client.delete(f"/api/documents/{a}")
    stack.llm.calls.clear()
    r = await client.post("/api/evaluations/evaluate", json=eval_body(a))
    assert r.status_code == 404 and not stack.llm.calls


# ---------------------------------------------------------------- logging privacy
async def test_logs_never_contain_content_secrets_or_prompts(client, stack, caplog):
    secret_answer = "STUDENT-ANSWER-CANARY-7731"
    secret_doc = "DOCUMENT-CANARY-9042 " + TEXT_A
    with caplog.at_level(logging.DEBUG):
        doc = await upload_text(client, secret_doc)
        await client.post("/api/evaluations/evaluate", json=eval_body(doc, secret_answer))
        await client.post("/api/rag/answer", json={"document_id": doc, "question": "QUESTION-CANARY-5521 zebracorn"})
        await client.post("/api/evaluations/evaluate", json={**eval_body(doc, secret_answer), "max_score": "bad"})
    for canary in ("STUDENT-ANSWER-CANARY-7731", "DOCUMENT-CANARY-9042", "QUESTION-CANARY-5521"):
        assert canary not in caplog.text, f"{canary} was logged"


async def test_admin_token_never_logged_even_on_failure(client, caplog, monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_api_token", TOKEN)
    with caplog.at_level(logging.DEBUG):
        await client.get("/api/ai-versions", headers={"X-Admin-Token": "WRONG-TOKEN-CANARY"})
        await client.get("/api/ai-versions", headers=H)
    assert "WRONG-TOKEN-CANARY" not in caplog.text and TOKEN not in caplog.text
    assert "admin authentication failed" in caplog.text


def test_settings_repr_does_not_expose_token(monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_api_token", TOKEN)
    monkeypatch.setattr(get_settings(), "gemini_api_key", "AQ.secret-gemini-key-canary-1234567890")
    assert TOKEN not in repr(get_settings()) and TOKEN not in str(get_settings())
    assert "AQ.secret-gemini-key-canary-1234567890" not in repr(get_settings())
    assert "AQ.secret-gemini-key-canary-1234567890" not in str(get_settings())


# ---------------------------------------------------------------- repository / container hygiene
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{30,}\b"),
    re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)[ \t]*[:=][ \t]*['\"][A-Za-z0-9/+_\-]{16,}['\"]"),
]
SCAN_DIRS = ("app", "scripts", "docs", "tests", ".github")
SCAN_FILES = (".env.example", "Dockerfile", "docker-compose.yml", "README.md", "requirements.txt")


def _files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.exists():
            yield from (p for p in base.rglob("*") if p.is_file() and p.suffix in
                        {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".toml", ".ini", ".sh", ".ps1"}
                        and "__pycache__" not in p.parts and "security" not in p.relative_to(ROOT).parts[1:2])
    for f in SCAN_FILES:
        if (ROOT / f).exists():
            yield ROOT / f


def test_no_hardcoded_secrets_in_tracked_sources():
    hits = []
    for path in _files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                value = m.group(0).lower()
                if any(w in value for w in ("test", "canary", "dummy", "example", "placeholder", "secret-")):
                    continue                                  # obviously fake fixture values
                hits.append(f"{path.relative_to(ROOT)}:{line}")
    assert not hits, hits


def test_env_example_has_no_secret_values():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for secret_key in ("ADMIN_API_TOKEN", "GEMINI_API_KEY"):
        m = re.search(rf"(?m)^{secret_key}=(.*)$", text)
        assert m is not None and m.group(1).strip() == "", secret_key


def test_gitignore_and_dockerignore_protect_secrets_and_data():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\.env\s*$", gi) and re.search(r"(?m)^\.env\s*$", di)
    assert "ai_versions" in gi, ".gitignore must exclude runtime AI-version data"
    assert "data/" in di and "uploads/" in di


def test_dockerfile_runs_non_root_without_env_file():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^USER app\s*$", df)
    assert not re.search(r"(?mi)^(COPY|ADD)\b.*\.env\b", df)
    assert "/app/data/ai_versions" in df


def test_compose_hardening():
    dc = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "no-new-privileges:true" in dc
    assert re.search(r"cap_drop:\s*\n\s*- ALL", dc)
    assert "127.0.0.1:11434:11434" in dc              # model server not exposed on all interfaces
    assert "ai_versions_data" in dc


def test_default_config_is_safe():
    s = get_settings()
    assert s.app_env != "production" or not s.cors_origins or "*" not in s.cors_origins
    assert s.cors_origins == [] or "*" not in s.cors_origins
    assert s.rate_limit_per_minute > 0 and s.max_request_body_bytes > 0 and s.max_pdf_pages > 0


# ---------------------------------------------------------------- admin plane (Phase 18) security
@pytest.fixture()
def admin(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "ai_versions_dir", str(tmp_path / "ai_versions"))
    monkeypatch.setattr(s, "admin_api_token", TOKEN)
    reset_ai_version_service()
    yield s
    reset_ai_version_service()


async def test_admin_plane_closed_without_valid_token(client, admin):
    for method, url in (("GET", "/api/ai-versions"), ("GET", "/api/ai-versions/active"),
                        ("POST", "/api/ai-versions"), ("POST", "/api/ai-versions/rollback"),
                        ("POST", "/api/ai-versions/x/promote"), ("POST", "/api/ai-versions/x/approve")):
        for headers in ({}, {"X-Admin-Token": ""}, {"X-Admin-Token": TOKEN[:-1]}, {"X-Admin-Token": TOKEN + "x"}):
            limiter.reset()
            resp = await client.request(method, url, headers=headers, json={})
            assert resp.status_code == 401, (method, url, headers)


async def test_admin_plane_disabled_when_no_token_configured(client, admin, monkeypatch):
    monkeypatch.setattr(admin, "admin_api_token", "")
    resp = await client.get("/api/ai-versions", headers={"X-Admin-Token": ""})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "ADMIN_DISABLED"


async def test_brute_force_is_locked_out_even_for_the_right_token(client, admin):
    for _ in range(10):
        assert (await client.get("/api/ai-versions", headers={"X-Admin-Token": "guess"})).status_code == 401
    locked = await client.get("/api/ai-versions", headers=H)
    assert locked.status_code == 429 and locked.headers.get("retry-after")
    assert locked.json()["error"]["code"] == "RATE_LIMITED"


async def test_client_cannot_inject_status_or_promote_unvalidated_version(client, admin):
    r = await client.post("/api/ai-versions", headers=H, json={"description": "x", "status": "active", "config": {}})
    assert r.status_code == 422
    for path in ("promote", "approve"):
        r = await client.post(f"/api/ai-versions/does-not-exist/{path}", headers=H, json={"reason": "go"})
        assert r.status_code in (404, 409, 422)
    r = await client.post("/api/ai-versions/rollback", headers=H, json={"reason": "go", "to_version_id": "../../etc"})
    assert r.status_code in (404, 409, 422)


async def test_version_id_traversal_is_rejected(client, admin):
    from urllib.parse import quote
    for evil in ("../../etc/passwd", "..\\..\\x", "/etc/passwd", "C:\\Windows"):
        r = await client.get(f"/api/ai-versions/{quote(evil, safe='')}", headers=H)
        assert r.status_code in (404, 422), (evil, r.status_code)
    assert not (Path(get_settings().ai_versions_dir).parent / "etc").exists()
