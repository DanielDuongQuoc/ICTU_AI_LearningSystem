"""Restore a ChromaDB backup made by scripts/backup.py.

    python scripts/restore.py --archive backups/chroma-backup-YYYYmmdd-HHMMSS.tar.gz [--target data/chroma] --yes

* DESTRUCTIVE to the target's current contents, so ``--yes`` is required.
* The existing target is MOVED to ``<target>.pre-restore-<timestamp>`` (never deleted) so a bad
  restore can be undone.
* Verifies the ``.sha256`` file when present and rejects archive members that would escape the target.
* Stop the API first (``docker compose stop api``) and start it again afterwards.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

try:  # works both as ``python scripts/restore.py`` and ``import scripts.restore``
    from scripts.backup import BackupError, default_source, sha256_of
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.backup import BackupError, default_source, sha256_of


def _verify_checksum(archive: Path) -> bool:
    sidecar = archive.with_name(archive.name + ".sha256")
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if sha256_of(archive) != expected:
        raise BackupError("Checksum mismatch: the archive is corrupted or was modified.")
    return True


def restore_backup(archive: Path, target: Path, *, assume_yes: bool, now: datetime | None = None) -> Path | None:
    """Restore ``archive`` into ``target``; return the moved-aside previous directory (or None)."""
    archive, target = Path(archive), Path(target)
    if not archive.is_file():
        raise BackupError(f"Archive not found: {archive}")
    if not assume_yes:
        raise BackupError(f"Restore replaces the contents of {target}. Re-run with --yes to confirm.")
    verified = _verify_checksum(archive)

    with tempfile.TemporaryDirectory(prefix="chroma-restore-", dir=target.parent if target.parent.exists() else None) as tmp:
        tmp_path = Path(tmp)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                root = tmp_path.resolve()
                for member in tar.getmembers():
                    dest = (tmp_path / member.name).resolve()
                    if root != dest and root not in dest.parents:
                        raise BackupError(f"Unsafe path in archive: {member.name}")
                    if member.issym() or member.islnk():
                        raise BackupError(f"Links are not allowed in archive: {member.name}")
                tar.extractall(tmp_path)  # noqa: S202 - members validated above
        except (tarfile.TarError, OSError) as exc:
            raise BackupError(f"Cannot read archive: {exc}") from exc
        extracted = tmp_path / "chroma"
        if not extracted.is_dir():
            raise BackupError("Archive does not contain a 'chroma' directory (not a backup of this module).")

        moved: Path | None = None
        if target.exists() and any(target.iterdir()):
            stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
            moved = target.with_name(f"{target.name}.pre-restore-{stamp}")
            if moved.exists():
                raise BackupError(f"Refusing to overwrite {moved}")
            shutil.move(str(target), str(moved))
        elif target.exists():
            target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))
    print(f"Checksum: {'verified' if verified else 'NOT verified (no .sha256 file)'}", file=sys.stderr)
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=None, help="Chroma directory (default: CHROMA_PERSIST_DIRECTORY)")
    parser.add_argument("--yes", action="store_true", help="confirm that the target will be replaced")
    args = parser.parse_args(argv)
    target = args.target or default_source()
    try:
        moved = restore_backup(args.archive, target, assume_yes=args.yes)
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Restored {args.archive} -> {target}")
    if moved:
        print(f"Previous data kept at: {moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
