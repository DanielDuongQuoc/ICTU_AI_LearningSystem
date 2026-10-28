"""Phase 15: release-candidate validation tests (offline) - configuration failures, error matrix,
upload error codes, isolation after a Chroma restart, empty/missing collections."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings, get_settings
from app.core.exceptions import ERROR_HTTP_STATUS
from app.repositories.chroma_repository import ChromaRepository
from app.services.rag_service import RAGService
from tests.fakes import LexicalEmbedding, RecordingLLM, Stack, sample_pdf_bytes

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ configuration failure matrix

@pytest.mark.parametrize("env,value", [
    ("OLLAMA_BASE_URL", "not-a-url"),
    ("OLLAMA_BASE_URL", "ftp://host:1"),
    ("OLLAMA_TIMEOUT", "0"),
    ("OLLAMA_TIMEOUT", "abc"),
    ("OLLAMA_CONNECT_TIMEOUT", "-1"),
    ("TOP_K", "0"),
    ("TOP_K", "50"),            # > MAX_TOP_K (20)
    ("MAX_TOP_K", "0"),
    ("MAX_UPLOAD_SIZE_MB", "0"),
    ("MAX_UPLOAD_SIZE_MB", "lots"),
    ("MAX_CONTEXT_CHARS", "0"),
    ("MAX_QUESTION_CHARS", "0"),
    ("MAX_ANSWER_CHARS", "-5"),
    ("CHUNK_SIZE", "0"),
    ("CHUNK_OVERLAP", "9999"),  # >= CHUNK_SIZE
    ("SIMILARITY_THRESHOLD", "2"),
    ("APP_ENV", "banana"),
    ("OLLAMA_MODEL", " "),
    ("CHROMA_COLLECTION_NAME", ""),
])
def test_invalid_environment_is_rejected(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_missing_environment_falls_back_to_valid_defaults(monkeypatch):
    for name in list(__import__("os").environ):
        if name.upper() in {f.upper() for f in Settings.model_fields}:
            monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.ollama_base_url.startswith("http") and s.top_k <= s.max_top_k
    assert s.cors_origins == []  # CORS off by default


def test_app_refuses_to_start_with_invalid_configuration(monkeypatch):
    import app.main as main

    monkeypatch.setenv("OLLAMA_BASE_URL", "not-a-url")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="APPLICATION_CONFIGURATION_ERROR"):
            main.create_app()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_source_has_no_hardcoded_deployment_values():
    """Ollama URL / model names live only in config defaults (and tests/docs)."""
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("11434", "qwen2.5", "nomic-embed-text"):
            if needle in text and path.name != "config.py":
                offenders.append(f"{path.name}:{needle}")
    assert offenders == []


# ------------------------------------------------------------ error matrix doc == code

def test_error_matrix_matches_implementation():
    rows = re.findall(r"^\|\s*(\d{3})\s*\|\s*`([A-Z_]+)`\s*\|", (ROOT / "docs" / "ERROR_MATRIX.md").read_text(encoding="utf-8"), re.M)
    documented = {code: int(status) for status, code in rows}
    assert documented == {str(k): v for k, v in ERROR_HTTP_STATUS.items()}
    for must_exist in ("EMPTY_QUERY", "DOCUMENT_NOT_FOUND", "FILE_TOO_LARGE", "OLLAMA_UNAVAILABLE",
                       "OLLAMA_TIMEOUT", "OLLAMA_MODEL_NOT_FOUND", "CHROMA_ERROR", "VALIDATION_ERROR",
                       "INTERNAL_SERVER_ERROR", "INVALID_FILE"):
        assert must_exist in documented


# ------------------------------------------------------------ upload error codes

@pytest.mark.asyncio
async def test_upload_error_codes(client, monkeypatch):
    Stack().install(monkeypatch)
    r = await client.post("/api/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_FILE"
    r = await client.post("/api/documents/upload", files={"file": ("x.pdf", b"", "application/pdf")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "EMPTY_FILE"
    r = await client.post("/api/documents/upload", files={"file": ("x.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert r.status_code in (400, 422) and r.json()["error"]["code"] in ("INVALID_PDF", "NO_EXTRACTABLE_TEXT")
    r = await client.post("/api/documents/upload", files={"file": ("../../evil.pdf", sample_pdf_bytes(["Zebracorn data."]), "application/pdf")})
    assert r.status_code in (200, 201)
    assert ".." not in r.json()["filename"] and "/" not in r.json()["filename"]


@pytest.mark.asyncio
async def test_oversized_upload_returns_413(client, monkeypatch):
    Stack().install(monkeypatch)
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    get_settings.cache_clear()
    try:
        big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
        r = await client.post("/api/documents/upload", files={"file": ("big.pdf", big, "application/pdf")})
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


# ------------------------------------------------------------ chroma: restart, empty, missing

def _persistent(path: str, name: str = "rc_collection") -> ChromaRepository:
    return ChromaRepository(client=chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False)),
                            collection_name=name)


@pytest.mark.asyncio
async def test_isolation_and_retrieval_survive_restart(client, monkeypatch, tmp_path):
    path = str(tmp_path / "chroma")
    monkeypatch.setattr("tests.fakes.make_repo", lambda name: _persistent(path))
    Stack().install(monkeypatch)
    ids = {}
    for key, text in (("a", "Zebracorn models learn patterns from data."), ("b", "Quokkaplex indexes speed up database queries.")):
        r = await client.post("/api/documents/upload", files={"file": (f"{key}.pdf", sample_pdf_bytes([text]), "application/pdf")})
        ids[key] = r.json()["document_id"]

    reopened = _persistent(path)  # fresh client on the same directory == process restart
    rag = RAGService(embedding_service=LexicalEmbedding(), chroma_repo=reopened, llm_service=RecordingLLM(),
                     similarity_threshold=0.0)
    for key, query, other in (("a", "zebracorn data", "quokkaplex"), ("b", "quokkaplex database", "zebracorn")):
        res = await rag.retrieve(ids[key], query)
        assert res["chunks"] and {c["document_id"] for c in res["chunks"]} == {ids[key]}
        assert other not in " ".join(c["text"].lower() for c in res["chunks"])


@pytest.mark.asyncio
async def test_empty_collection_and_deleted_document(client, monkeypatch):
    s = Stack().install(monkeypatch)
    assert s.repo.count() == 0
    r = await client.post("/api/rag/retrieve", json={"document_id": "11111111-1111-4111-8111-111111111111", "question": "q"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    up = await client.post("/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes(["Zebracorn data."]), "application/pdf")})
    doc = up.json()["document_id"]
    assert (await client.delete(f"/api/documents/{doc}")).status_code == 200
    r = await client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn"})
    assert r.status_code == 404 and s.repo.count_by_document(doc) == 0


# ------------------------------------------------------------ release documentation

def test_final_documents_exist_and_do_not_overclaim():
    for name in ("HANDOVER", "DEPLOYMENT", "INTEGRATION_CONTRACT", "ERROR_MATRIX", "BACKUP_AND_RECOVERY",
                 "RELEASE", "FINAL_RELEASE_REPORT"):
        assert (ROOT / "docs" / f"{name}.md").is_file(), name
    report = (ROOT / "docs" / "FINAL_RELEASE_REPORT.md").read_text(encoding="utf-8")
    assert "NOT VERIFIED" in report and ("REJECTED" in report or "APPROVED" in report)
