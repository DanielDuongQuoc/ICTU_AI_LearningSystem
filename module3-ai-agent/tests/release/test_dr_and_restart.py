"""Restart persistence and backup/restore (real PersistentClient, real separate processes, temp data only)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.backup import BackupError, create_backup
from scripts.restore import restore_backup

ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import json, sys
import chromadb
from chromadb.config import Settings
mode, path = sys.argv[1], sys.argv[2]
col = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False)).get_or_create_collection(
    "drill", metadata={"hnsw:space": "cosine"})
if mode == "ingest":
    col.upsert(ids=["c0", "c1"], documents=["zebracorn data", "quokka index"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
               metadatas=[{"document_id": "d", "source": "drill.pdf"}] * 2)
res = col.query(query_embeddings=[[1.0, 0.0]], n_results=1, where={"document_id": "d"})
print(json.dumps({"chunks": col.count(), "results": len(res["ids"][0]),
                  "source": res["metadatas"][0][0]["source"] if res["ids"][0] else None}))
"""


def _spawn(mode: str, data: Path) -> dict:
    """Run ingest/query in a brand-new Python process (= an application restart)."""
    proc = subprocess.run([sys.executable, "-c", _CHILD, mode, str(data)], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_data_survives_process_restart(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    before = _spawn("ingest", data)
    assert before["chunks"] > 0
    after = _spawn("query", data)               # a brand-new Python process = a restart
    assert after["chunks"] == before["chunks"] and after["results"] > 0 and after["source"] == "drill.pdf"


def test_corrupted_backup_is_rejected_and_target_untouched(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    _spawn("ingest", data)
    archive = create_backup(data, tmp_path / "backups")
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(bytes(raw))
    before = sorted(p.name for p in data.rglob("*"))
    with pytest.raises(BackupError, match="(?i)checksum|cannot read"):
        restore_backup(archive, data, assume_yes=True)
    assert sorted(p.name for p in data.rglob("*")) == before            # nothing replaced


def test_restore_requires_confirmation_and_keeps_previous_data(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    _spawn("ingest", data)
    archive = create_backup(data, tmp_path / "backups")
    with pytest.raises(BackupError, match="--yes"):
        restore_backup(archive, data, assume_yes=False)
    moved = restore_backup(archive, data, assume_yes=True)               # target is non-empty
    assert moved is not None and moved.is_dir() and any(moved.iterdir()), "previous data must be kept aside"
    assert _spawn("query", data)["results"] > 0


def test_backup_refuses_missing_empty_and_existing(tmp_path):
    with pytest.raises(BackupError):
        create_backup(tmp_path / "missing", tmp_path / "b")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BackupError):
        create_backup(empty, tmp_path / "b")
    data = tmp_path / "chroma"
    data.mkdir()
    (data / "f").write_text("x")
    from datetime import datetime

    stamp = datetime(2026, 1, 1, 1, 1, 1)
    create_backup(data, tmp_path / "b", now=stamp)
    with pytest.raises(BackupError, match="overwrite"):
        create_backup(data, tmp_path / "b", now=stamp)
    shutil.rmtree(tmp_path / "b")
