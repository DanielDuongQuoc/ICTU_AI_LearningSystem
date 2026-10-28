"""Prune old Chroma backups made by ``scripts/backup.py`` (retention policy).

    python scripts/rotate_backups.py [--dir backups] [--keep 7]            # dry run (default)
    python scripts/rotate_backups.py --dir backups --keep 7 --apply        # actually delete

Safety rules:
* Dry run unless ``--apply`` is given.
* Only files named ``chroma-backup-*.tar.gz`` (and their ``.sha256``) inside ``--dir`` are touched.
* The newest ``--keep`` *valid* archives are always kept; ``--keep`` must be >= 1.
* A backup whose checksum is missing/mismatched is reported as CORRUPT, never counted as a good
  copy, and never deleted (left for human inspection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backup import ARCHIVE_PREFIX, sha256_of  # noqa: E402


def is_valid(archive: Path) -> bool:
    sidecar = archive.with_name(archive.name + ".sha256")
    if not sidecar.is_file():
        return False
    try:
        expected = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return False
    return sha256_of(archive) == expected


def plan(directory: Path, keep: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (kept, to_delete, corrupt). Names embed a sortable timestamp, newest last."""
    archives = sorted(p for p in Path(directory).glob(f"{ARCHIVE_PREFIX}*.tar.gz") if p.is_file())
    valid = [a for a in archives if is_valid(a)]
    corrupt = [a for a in archives if a not in valid]
    kept = valid[-keep:]
    to_delete = valid[:-keep] if len(valid) > keep else []
    return kept, to_delete, corrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("backups"))
    parser.add_argument("--keep", type=int, default=7)
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args(argv)
    if args.keep < 1:
        print("ERROR: --keep must be >= 1", file=sys.stderr)
        return 2
    if not args.dir.is_dir():
        print(f"ERROR: backup directory not found: {args.dir}", file=sys.stderr)
        return 1
    kept, to_delete, corrupt = plan(args.dir, args.keep)
    for a in corrupt:
        print(f"CORRUPT (missing/mismatched checksum, left untouched): {a.name}")
    for a in kept:
        print(f"KEEP   {a.name}")
    for a in to_delete:
        print(f"{'DELETE' if args.apply else 'would delete'} {a.name}")
        if args.apply:
            a.unlink(missing_ok=True)
            a.with_name(a.name + ".sha256").unlink(missing_ok=True)
    if not kept:
        print("WARNING: no valid backup exists.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
