"""Phase 16 - ops scripts (alert evaluation, backup retention) and a real backup -> restore drill."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.repositories.chroma_repository import ChromaRepository
from scripts import backup, ops_check, restore, rotate_backups
from tests.fakes import LexicalEmbedding

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ alert evaluation

def _samples(ok: int, err: int, bucket_le: str = "0.1", rejected: int = 0):
    total = ok + err
    text = (
        f'module3_http_requests_total{{method="POST",route="/api/rag/answer",status="200"}} {ok}\n'
        f'module3_http_requests_total{{method="POST",route="/api/rag/answer",status="503"}} {err}\n'
        f'module3_http_request_duration_seconds_bucket{{method="POST",route="/api/rag/answer",le="{bucket_le}"}} {total}\n'
        f'module3_http_request_duration_seconds_bucket{{method="POST",route="/api/rag/answer",le="+Inf"}} {total}\n'
        f"module3_llm_rejected_total {rejected}\n"
    )
    return ops_check.parse_metrics(text)


KW = dict(max_5xx_ratio=0.05, max_p95_seconds=60, min_requests=20)


def test_alerts_healthy_and_noise_guard():
    assert ops_check.evaluate(_samples(100, 1), **KW) == []
    assert ops_check.evaluate(_samples(3, 3), **KW) == []          # below min_requests


def test_alert_5xx_slow_and_overload():
    assert any(a.startswith("HIGH_5XX_RATIO") for a in ops_check.evaluate(_samples(50, 50), **KW))
    assert any(a.startswith("SLOW_P95") for a in ops_check.evaluate(_samples(100, 0, bucket_le="120.0"), **KW))
    assert any(a.startswith("LLM_OVERLOAD") for a in ops_check.evaluate(_samples(100, 0, rejected=50), **KW))


def test_ops_check_unreachable_returns_2():
    assert ops_check.main(["--base-url", "http://127.0.0.1:9"]) == 2


@pytest.mark.asyncio
async def test_ops_check_parses_real_metrics_output(client):
    from app.core.metrics import metrics

    metrics.reset()
    await client.get("/health/live")
    text = (await client.get("/metrics")).text
    parsed = ops_check.parse_metrics(text)
    assert any(n == "module3_http_requests_total" for n, _, _ in parsed)
    assert ops_check.evaluate(parsed, **KW) == []


# ------------------------------------------------------------------ retention

def _make_backups(tmp_path: Path, n: int) -> list[Path]:
    src = tmp_path / "chroma"
    src.mkdir(exist_ok=True)
    (src / "f.bin").write_bytes(b"data")
    base = datetime(2026, 1, 1, 12, 0, 0)
    return [backup.create_backup(src, tmp_path / "b", now=base + timedelta(days=i)) for i in range(n)]


def test_rotation_dry_run_deletes_nothing(tmp_path):
    archives = _make_backups(tmp_path, 4)
    assert rotate_backups.main(["--dir", str(tmp_path / "b"), "--keep", "2"]) == 0
    assert all(a.exists() for a in archives)


def test_rotation_apply_keeps_newest(tmp_path):
    archives = _make_backups(tmp_path, 4)
    assert rotate_backups.main(["--dir", str(tmp_path / "b"), "--keep", "2", "--apply"]) == 0
    assert [a.exists() for a in archives] == [False, False, True, True]
    assert not archives[0].with_name(archives[0].name + ".sha256").exists()


def test_rotation_never_counts_corrupt_as_good_or_deletes_it(tmp_path):
    archives = _make_backups(tmp_path, 3)
    archives[2].write_bytes(b"corrupted")                # newest is corrupt
    assert rotate_backups.main(["--dir", str(tmp_path / "b"), "--keep", "1", "--apply"]) == 0
    assert archives[1].exists() and not archives[0].exists() and archives[2].exists()


def test_rotation_guards(tmp_path):
    assert rotate_backups.main(["--dir", str(tmp_path), "--keep", "0"]) == 2
    (tmp_path / "notes.tar.gz").write_bytes(b"x")
    assert rotate_backups.main(["--dir", str(tmp_path), "--keep", "1", "--apply"]) == 1   # no valid backup
    assert (tmp_path / "notes.tar.gz").exists()          # foreign file untouched


# ------------------------------------------------------------------ backup -> modify -> restore drill

def _repo(path: Path) -> ChromaRepository:
    return ChromaRepository(
        client=chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False)),
        collection_name="drill",
    )


@pytest.mark.asyncio
async def test_backup_restore_drill_retrieval_works_in_isolated_dir(tmp_path):
    """Real ChromaDB persistence under tmp_path only; production data is never touched."""
    emb = LexicalEmbedding()
    live = tmp_path / "live" / "chroma"
    live.mkdir(parents=True)
    repo = _repo(live)
    texts = ["Zebracorn models learn patterns from data.", "Zebracorn systems explain machine learning."]
    repo.add_documents(
        ids=["c0", "c1"], documents=texts, embeddings=await emb.embed_documents(texts),
        metadatas=[{"document_id": "doc-1", "page": 1, "chunk_index": i, "chunk_id": f"c{i}"} for i in range(2)],
    )
    assert repo.count_by_document("doc-1") == 2

    archive = backup.create_backup(live, tmp_path / "backups")
    assert archive.exists()
    assert backup.sha256_of(archive) in archive.with_name(archive.name + ".sha256").read_text()

    repo.delete_document("doc-1")                        # simulate data loss after the backup
    assert repo.count_by_document("doc-1") == 0

    restored_dir = tmp_path / "restored" / "chroma"
    restore.restore_backup(archive, restored_dir, assume_yes=True)
    restored = _repo(restored_dir)
    assert restored.count_by_document("doc-1") == 2      # integrity: chunks are back
    hits = restored.query(query_embedding=await emb.embed_text("zebracorn data"), document_id="doc-1",
                          top_k=2, similarity_threshold=0.0)
    assert hits                                          # retrieval works on restored data


def test_restore_refuses_without_confirmation_and_on_corruption(tmp_path):
    archives = _make_backups(tmp_path, 1)
    with pytest.raises(backup.BackupError):
        restore.restore_backup(archives[0], tmp_path / "t", assume_yes=False)
    archives[0].write_bytes(b"corrupted")
    with pytest.raises(backup.BackupError):
        restore.restore_backup(archives[0], tmp_path / "t", assume_yes=True)

# ------------------------------------------------------------------ docs

def test_operational_docs_exist_and_state_limits_honestly():
    for name in ("OPERATIONS_RUNBOOK", "INCIDENT_RESPONSE", "MAINTENANCE", "PHASE16_REPORT"):
        assert (ROOT / "docs" / f"{name}.md").is_file(), name
    runbook = (ROOT / "docs" / "OPERATIONS_RUNBOOK.md").read_text(encoding="utf-8")
    for case in ("Case A", "Case B", "Case C", "Case D", "Case E", "Case F", "Case G"):
        assert case in runbook, case
    assert "/metrics" in runbook and "NOT VERIFIED" in (ROOT / "docs" / "PHASE16_REPORT.md").read_text(encoding="utf-8")
