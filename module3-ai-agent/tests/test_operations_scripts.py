"""Backup / restore scripts: a real backup -> data loss -> restore drill on temp data only."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.repositories.chroma_repository import ChromaRepository
from scripts import backup, restore
from tests.fakes import LexicalEmbedding

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers

def _make_backups(tmp_path: Path, n: int) -> list[Path]:
    src = tmp_path / "chroma"
    src.mkdir(exist_ok=True)
    (src / "f.bin").write_bytes(b"data")
    base = datetime(2026, 1, 1, 12, 0, 0)
    return [backup.create_backup(src, tmp_path / "b", now=base + timedelta(days=i)) for i in range(n)]


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

