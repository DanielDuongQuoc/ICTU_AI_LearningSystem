"""Phase 14: production hardening - input limits, readiness alias, version, backup/restore,
smoke-test flow, release checks and deployment-config invariants (all offline)."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import yaml

from app.core.config import Settings
from app.main import app
from app.version import MODULE_VERSION
from scripts import backup, release_check, restore, smoke_test
from tests.fakes import Stack, sample_pdf_bytes

ROOT = Path(__file__).resolve().parents[1]
DOC = "Zebracorn models learn patterns from data. Zebracorn systems explain machine learning."


@pytest.fixture()
def stack(monkeypatch):
    return Stack().install(monkeypatch)


async def _upload(client) -> str:
    r = await client.post("/api/documents/upload", files={"file": ("d.pdf", sample_pdf_bytes([DOC]), "application/pdf")})
    assert r.status_code in (200, 201)
    return r.json()["document_id"]


def _sse(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------- input limits

def test_limits_are_configurable_and_validated():
    assert Settings().max_question_chars == 2000 and Settings().max_answer_chars == 10000
    assert Settings(max_question_chars=50).max_question_chars == 50
    for bad in ({"max_question_chars": 0}, {"max_answer_chars": -1}):
        with pytest.raises(ValueError):
            Settings(**bad)


@pytest.mark.asyncio
async def test_oversized_inputs_rejected_before_any_llm_call(client, stack):
    doc = await _upload(client)
    stack.llm.calls.clear()
    body = {"document_id": doc, "question": "zebracorn?", "student_answer": "a",
            "expected_answer": "b", "max_score": 10}
    cases = {
        "/api/rag/retrieve": {"document_id": doc, "question": "q" * 2001},
        "/api/rag/answer": {"document_id": doc, "question": "q" * 2001},
        "/api/evaluations/evaluate": {**body, "student_answer": "x" * 10001},
    }
    for path, payload in cases.items():
        r = await client.post(path, json=payload)
        assert r.status_code == 400 and r.json()["error"]["code"] == "INPUT_TOO_LONG", path
        assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]
    r = await client.post("/api/evaluations/evaluate", json={**body, "question": "q" * 2001})
    assert r.json()["error"]["code"] == "INPUT_TOO_LONG"
    r = await client.post("/api/evaluations/evaluate", json={**body, "expected_answer": "e" * 10001})
    assert r.json()["error"]["code"] == "INPUT_TOO_LONG"
    assert stack.llm.calls == []


@pytest.mark.asyncio
async def test_stream_oversized_input_is_error_then_done(client, stack):
    doc = await _upload(client)
    r = await client.post("/api/evaluations/evaluate/stream", json={
        "document_id": doc, "question": "zebracorn?", "student_answer": "x" * 10001,
        "expected_answer": "b", "max_score": 10})
    events = _sse(r.text)
    types = [e["type"] for e in events]
    assert types[-1] == "done" and "result" not in types and "token" not in types
    assert next(e for e in events if e["type"] == "error")["code"] == "INPUT_TOO_LONG"
    assert stack.llm.calls == []


@pytest.mark.asyncio
async def test_inputs_at_the_limit_are_accepted(client, stack):
    doc = await _upload(client)
    r = await client.post("/api/evaluations/evaluate", json={
        "document_id": doc, "question": "zebracorn " + "q" * 1990, "student_answer": "x" * 10000,
        "expected_answer": "zebracorn learns", "max_score": 10})
    assert r.status_code == 200


# ---------------------------------------------------------------- readiness + version

@pytest.mark.asyncio
async def test_ready_alias_matches_health_ready(client):
    a, b = await client.get("/ready"), await client.get("/health/ready")
    assert a.status_code == b.status_code and a.json() == b.json()
    assert a.status_code in (200, 503)


@pytest.mark.asyncio
async def test_health_does_not_call_the_llm(client, monkeypatch):
    called = []
    from app.services import llm_service

    async def spy(*a, **k):
        called.append(1)
        return "x"

    monkeypatch.setattr(llm_service.OllamaLLMService, "generate", spy)
    await client.get("/health")
    await client.get("/health/ready")
    assert called == []


@pytest.mark.asyncio
async def test_version_is_single_sourced_and_pre_1_0(client):
    body = (await client.get("/health")).json()
    assert body["version"] == MODULE_VERSION == app.version
    assert not MODULE_VERSION.startswith("1.")
    assert (await client.get("/openapi.json")).json()["info"]["version"] == MODULE_VERSION


# ---------------------------------------------------------------- backup / restore

def _make_chroma(tmp_path: Path) -> Path:
    src = tmp_path / "chroma"
    (src / "idx").mkdir(parents=True)
    (src / "chroma.sqlite3").write_bytes(b"sqlite-data")
    (src / "idx" / "data.bin").write_bytes(b"\x00\x01\x02")
    return src


def test_backup_creates_verified_archive(tmp_path):
    src = _make_chroma(tmp_path)
    archive = backup.create_backup(src, tmp_path / "backups", now=datetime(2026, 1, 2, 3, 4, 5))
    assert archive.name == "chroma-backup-20260102-030405.tar.gz"
    sidecar = archive.with_name(archive.name + ".sha256")
    assert sidecar.read_text().split()[0] == backup.sha256_of(archive)
    with tarfile.open(archive) as tar:
        assert {"chroma/chroma.sqlite3", "chroma/idx/data.bin"} <= set(tar.getnames())
    assert (src / "chroma.sqlite3").exists()  # source untouched


def test_backup_never_overwrites_and_rejects_bad_sources(tmp_path):
    src = _make_chroma(tmp_path)
    when = datetime(2026, 1, 2, 3, 4, 5)
    backup.create_backup(src, tmp_path / "b", now=when)
    with pytest.raises(backup.BackupError, match="overwrite"):
        backup.create_backup(src, tmp_path / "b", now=when)
    with pytest.raises(backup.BackupError, match="not found"):
        backup.create_backup(tmp_path / "missing", tmp_path / "b")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".gitkeep").write_text("")
    with pytest.raises(backup.BackupError, match="empty"):
        backup.create_backup(empty, tmp_path / "b")


def test_restore_requires_confirmation_and_keeps_previous_data(tmp_path):
    src = _make_chroma(tmp_path)
    archive = backup.create_backup(src, tmp_path / "b", now=datetime(2026, 1, 1))
    (src / "chroma.sqlite3").write_bytes(b"CHANGED-AFTER-BACKUP")

    with pytest.raises(backup.BackupError, match="--yes"):
        restore.restore_backup(archive, src, assume_yes=False)
    assert (src / "chroma.sqlite3").read_bytes() == b"CHANGED-AFTER-BACKUP"  # untouched without --yes

    moved = restore.restore_backup(archive, src, assume_yes=True, now=datetime(2026, 2, 2, 2, 2, 2))
    assert (src / "chroma.sqlite3").read_bytes() == b"sqlite-data"
    assert (src / "idx" / "data.bin").read_bytes() == b"\x00\x01\x02"
    assert moved is not None and (moved / "chroma.sqlite3").read_bytes() == b"CHANGED-AFTER-BACKUP"


def test_restore_into_fresh_target_and_cli_exit_codes(tmp_path, capsys):
    src = _make_chroma(tmp_path)
    archive = backup.create_backup(src, tmp_path / "b")
    target = tmp_path / "new" / "chroma"
    assert restore.main(["--archive", str(archive), "--target", str(target), "--yes"]) == 0
    assert (target / "chroma.sqlite3").exists()
    assert restore.main(["--archive", str(tmp_path / "nope.tar.gz"), "--target", str(target), "--yes"]) == 1
    assert backup.main(["--source", str(tmp_path / "missing"), "--dest-dir", str(tmp_path / "b2")]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_restore_rejects_corrupted_or_malicious_archives(tmp_path):
    src = _make_chroma(tmp_path)
    archive = backup.create_backup(src, tmp_path / "b")
    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text("0" * 64 + f"  {archive.name}\n")
    with pytest.raises(backup.BackupError, match="Checksum"):
        restore.restore_backup(archive, tmp_path / "t1", assume_yes=True)

    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../../escaped.txt")
    with pytest.raises(backup.BackupError, match="Unsafe"):
        restore.restore_backup(evil, tmp_path / "t2", assume_yes=True)
    assert not (tmp_path.parent / "escaped.txt").exists()

    other = tmp_path / "other.tar.gz"
    with tarfile.open(other, "w:gz") as tar:
        tar.add(payload, arcname="something.txt")
    with pytest.raises(backup.BackupError, match="chroma"):
        restore.restore_backup(other, tmp_path / "t3", assume_yes=True)


# ---------------------------------------------------------------- smoke flow (offline, fakes)

@pytest.mark.asyncio
async def test_smoke_flow_passes_against_fakes(monkeypatch):
    s = Stack().install(monkeypatch)
    s.llm.questions_payload = {"questions": [{
        "question": "What does photosynthesis do with light energy in plants?",
        "expected_answer": "Plants convert light energy into chemical energy.",
        "difficulty": "medium", "type": "short_answer"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        results = await smoke_test.run_smoke(client, require_ready=False)
    failed = [r for r in results if not r[1]]
    assert not failed, failed
    assert [r[0] for r in results] == ["health", "ready", "upload", "get", "retrieve", "questions",
                                       "evaluate", "stream", "delete", "verify_deleted"]


@pytest.mark.asyncio
async def test_smoke_reports_failure_instead_of_raising():
    class Dead(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=Dead(), base_url="http://t") as client:
        results = await smoke_test.run_smoke(client)
    assert results and not all(ok for _, ok, _ in results)


# ---------------------------------------------------------------- release checks / deployment invariants

def test_release_static_checks_have_no_failures():
    results = release_check.run_checks()
    assert [r for r in results if r[1] == "FAIL"] == []


def test_release_check_detects_a_secret(monkeypatch, tmp_path):
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / ".dockerignore").write_text(".env\n")
    (tmp_path / ".env.example").write_text("API_KEY=abc123\n")
    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    assert release_check.check_env_files()[1] == "FAIL"


def test_compose_and_dockerfile_invariants():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert api["stop_grace_period"] == "30s"
    assert "no-new-privileges:true" in api["security_opt"]
    assert "OLLAMA_BASE_URL=http://ollama:11434" in api["environment"]
    assert all(p.startswith("127.0.0.1:") for p in compose["services"]["ollama"]["ports"])
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER app" in dockerfile and "exec uvicorn" in dockerfile and not any(ln.split()[0] in ("COPY", "ADD") and ".env" in ln for ln in dockerfile.splitlines() if ln.strip())


def test_ci_workflow_runs_lint_and_tests():
    wf = yaml.safe_load((ROOT.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    steps = " ".join(str(s.get("run", "")) for job in wf["jobs"].values() for s in job["steps"])
    assert "pytest" in steps and "ruff check" in steps and "docker compose config" in steps


def test_docs_exist():
    for doc in release_check.REQUIRED_DOCS:
        assert (ROOT / doc).is_file(), doc
