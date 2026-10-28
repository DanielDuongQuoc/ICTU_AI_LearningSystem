"""Back up the ChromaDB persistent directory to a timestamped, checksummed .tar.gz.

    python scripts/backup.py [--source data/chroma] [--dest-dir backups]

* Never overwrites an existing backup and never deletes anything.
* Writes ``<archive>.sha256`` and verifies the archive can be read back.
* SAFETY: ChromaDB uses SQLite + on-disk index files. Copying while the API is writing can give an
  inconsistent snapshot. Stop writers first (``docker compose stop api``) - the script warns, it
  cannot detect a running writer.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ARCHIVE_PREFIX = "chroma-backup-"


class BackupError(Exception):
    """A backup/restore precondition failed (message is user-facing)."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_source() -> Path:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.core.config import get_settings

        return Path(get_settings().chroma_persist_directory)
    except Exception:  # settings unavailable (e.g. running outside the project)
        return Path("data/chroma")


def create_backup(source: Path, dest_dir: Path, *, now: datetime | None = None) -> Path:
    source, dest_dir = Path(source), Path(dest_dir)
    if not source.is_dir():
        raise BackupError(f"Source directory not found: {source}")
    files = [p for p in source.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not files:
        raise BackupError(f"Source directory is empty (nothing to back up): {source}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    archive = dest_dir / f"{ARCHIVE_PREFIX}{stamp}.tar.gz"
    if archive.exists():
        raise BackupError(f"Refusing to overwrite existing backup: {archive}")
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname="chroma")
    except OSError as exc:
        archive.unlink(missing_ok=True)
        raise BackupError(f"Backup failed: {exc}") from exc
    # Verify: archive is readable and contains every file.
    with tarfile.open(archive, "r:gz") as tar:
        stored = sum(1 for m in tar.getmembers() if m.isfile())
    if stored != len([p for p in source.rglob("*") if p.is_file()]):
        archive.unlink(missing_ok=True)
        raise BackupError("Backup verification failed (file count mismatch); source changed during backup?")
    (archive.with_name(archive.name + ".sha256")).write_text(f"{sha256_of(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=None, help="Chroma directory (default: CHROMA_PERSIST_DIRECTORY)")
    parser.add_argument("--dest-dir", type=Path, default=Path("backups"))
    args = parser.parse_args(argv)
    source = args.source or default_source()
    print("WARNING: stop the API first (docker compose stop api) for a consistent snapshot.", file=sys.stderr)
    try:
        archive = create_backup(source, args.dest_dir)
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Backup created: {archive} (sha256 in {archive.name}.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
