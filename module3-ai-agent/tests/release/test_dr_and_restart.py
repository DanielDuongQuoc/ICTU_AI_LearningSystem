"""Phase 21 - restart persistence, backup/restore and the disaster-recovery drill (real PersistentClient,
real separate processes, disposable temp data only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import dr_drill
from scripts.backup import BackupError, create_backup
from scripts.restore import restore_backup

ROOT = Path(__file__).resolve().parents[2]


def test_data_survives_process_restart(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    before = dr_drill._spawn("ingest", data)
    assert before["chunks"] > 0
    after = dr_drill._spawn("query", data)               # a brand-new Python process = a restart
    assert after["chunks"] == before["chunks"] and after["results"] > 0 and after["source"] == "drill.pdf"


def test_full_dr_drill_on_test_data(tmp_path):
    result = dr_drill.run_drill(tmp_path / "work")
    assert result["ok"], result["steps"]
    assert "NOT" in result["scope"] or "not a production" in result["scope"]
    assert set(result["timings_s"]) >= {"backup_s", "restore_s", "rto_observed_s"}


def test_drill_cli_exit_code_zero(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "scripts.dr_drill", "--workdir", str(tmp_path / "w")],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout and "FAIL" not in proc.stdout


def test_drill_refuses_configured_or_nonempty_directory(tmp_path):
    from app.core.config import get_settings

    with pytest.raises(RuntimeError):
        dr_drill.run_drill(Path(get_settings().chroma_persist_directory))
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("keep me")
    with pytest.raises(RuntimeError):
        dr_drill.run_drill(busy)
    assert (busy / "x").read_text() == "keep me"


def test_corrupted_backup_is_rejected_and_target_untouched(tmp_path):
    data = tmp_path / "chroma"
    data.mkdir()
    dr_drill._spawn("ingest", data)
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
    dr_drill._spawn("ingest", data)
    archive = create_backup(data, tmp_path / "backups")
    with pytest.raises(BackupError, match="--yes"):
        restore_backup(archive, data, assume_yes=False)
    moved = restore_backup(archive, data, assume_yes=True)               # target is non-empty
    assert moved is not None and moved.is_dir() and any(moved.iterdir()), "previous data must be kept aside"
    assert dr_drill._spawn("query", data)["results"] > 0


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
