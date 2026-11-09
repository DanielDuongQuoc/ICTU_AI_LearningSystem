"""Keep only files whose format is accepted for the assignment, and build the text sent to the grader."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent.config import AgentConfig

SNIFF_BYTES = 8192
_ENTRY_POINT = re.compile(r"\b(static\s+void\s+Main|public\s+static\s+void\s+main|int\s+main\s*\(|"
                          r"if\s+__name__\s*==\s*['\"]__main__['\"]|class\s+Program\b)")


@dataclass
class AcceptedFile:
    rel_path: str
    size: int
    text: str
    entry_point: bool = False


@dataclass
class Selection:
    accepted: list[AcceptedFile] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (rel_path, reason)


def _decode(data: bytes) -> Optional[str]:
    if b"\x00" in data[:SNIFF_BYTES]:
        return None
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1258"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def select_files(root: Path, cfg: AgentConfig) -> Selection:
    sel = Selection()
    excluded = set(cfg.exclude_dirs)
    accepted_ext = set(cfg.accepted_extensions)
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if d.lower() in excluded:
                sel.rejected.append(((base / d).relative_to(root).as_posix() + "/", "excluded_dir"))
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            path = base / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                sel.rejected.append((rel, "symlink"))
                continue
            if path.suffix.lower() not in accepted_ext:
                sel.rejected.append((rel, "extension_not_accepted"))
                continue
            size = path.stat().st_size
            if size > cfg.max_file_kb * 1024:
                sel.rejected.append((rel, "too_large"))
                continue
            text = _decode(path.read_bytes())
            if text is None:
                sel.rejected.append((rel, "binary"))
                continue
            if not text.strip():
                sel.rejected.append((rel, "empty"))
                continue
            sel.accepted.append(AcceptedFile(rel, size, text, bool(_ENTRY_POINT.search(text))))
    # Entry points first, then by path: they are the most informative when the size budget runs out.
    sel.accepted.sort(key=lambda f: (not f.entry_point, f.rel_path.lower()))
    return sel


def build_answer(files: list[AcceptedFile], max_chars: int) -> tuple[str, bool, list[str]]:
    """``(answer, truncated, included_paths)``. Each file is prefixed with a ``// FILE:`` header."""
    parts: list[str] = []
    included: list[str] = []
    used = 0
    truncated = False
    for f in files:
        block = f"// FILE: {f.rel_path}\n{f.text.rstrip()}\n\n"
        if used + len(block) <= max_chars:
            parts.append(block)
            included.append(f.rel_path)
            used += len(block)
            continue
        truncated = True
        marker = "\n// [TRUNCATED]\n"
        room = max_chars - used - len(marker)
        if room > 200:  # include the beginning of the file rather than nothing
            parts.append(block[:room] + marker)
            included.append(f.rel_path + " (truncated)")
        break
    return "".join(parts), truncated, included
