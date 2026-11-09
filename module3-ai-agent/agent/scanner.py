"""Find one submission per student: every sub-folder or ``.zip`` in ``submissions_dir``.

Zip files are extracted into ``output_dir/.work/<student_ref>`` with zip-slip and zip-bomb protection.
"""

from __future__ import annotations

import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

from agent.config import AgentConfig


@dataclass
class Submission:
    student_ref: Optional[str]
    source: Path
    root: Optional[Path] = None          # folder holding the files (extracted for zips)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    submissions: list[Submission]
    ignored: list[tuple[str, str]]       # (entry name, reason)


class UnsafeZip(Exception):
    pass


def _safe_extract(zip_path: Path, dest: Path, cfg: AgentConfig) -> None:
    limit = cfg.max_unzipped_mb * 1024 * 1024
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > cfg.max_zip_entries:
            raise UnsafeZip(f"too many entries ({len(infos)})")
        if sum(i.file_size for i in infos) > limit:
            raise UnsafeZip("uncompressed size exceeds max_unzipped_mb")
        dest_resolved = dest.resolve()
        for info in infos:
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                raise UnsafeZip(f"unsafe path in archive: {name!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                continue  # never materialise symlinks
            target = (dest / name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise UnsafeZip(f"unsafe path in archive: {name!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(65536):
                    written += len(chunk)
                    if written > limit:
                        raise UnsafeZip("uncompressed size exceeds max_unzipped_mb")
                    out.write(chunk)


def _student_ref(name: str, cfg: AgentConfig) -> Optional[str]:
    m = cfg.naming_regex.match(name)
    return m.group("student_ref") if m else None


def scan(cfg: AgentConfig) -> ScanResult:
    if not cfg.submissions_dir.is_dir():
        raise FileNotFoundError(f"submissions_dir not found: {cfg.submissions_dir}")
    subs: list[Submission] = []
    ignored: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in sorted(cfg.submissions_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        is_zip = entry.is_file() and entry.suffix.lower() == ".zip"
        if not (entry.is_dir() or is_zip):
            ignored.append((entry.name, "not_a_folder_or_zip"))
            continue
        stem = entry.stem if is_zip else entry.name
        ref = _student_ref(stem, cfg)
        if ref is None:
            ignored.append((entry.name, "bad_name"))
            continue
        if ref in seen:
            ignored.append((entry.name, "duplicate_student_ref"))
            continue
        seen.add(ref)
        sub = Submission(student_ref=ref, source=entry)
        if entry.is_dir():
            sub.root = entry
        else:
            dest = cfg.work_dir / ref
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                _safe_extract(entry, dest, cfg)
                sub.root = dest
            except (UnsafeZip, zipfile.BadZipFile, OSError) as exc:
                shutil.rmtree(dest, ignore_errors=True)
                sub.errors.append(f"unreadable: {type(exc).__name__}: {exc}"[:300])
        subs.append(sub)
    return ScanResult(submissions=subs, ignored=ignored)
