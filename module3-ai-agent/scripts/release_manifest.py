"""Release manifest (Phase 21): a secret-free, reproducible description of one release.

    python -m scripts.release_manifest [--out release-manifest.json] [--gates gates.json]

* ``release_id`` is derived from the CONTENT (application version, git commit, dependency specification,
  Dockerfile, effective AI configuration incl. prompt contents). The same inputs give the same id; the
  build timestamp is recorded but deliberately NOT part of the id.
* Gate results default to ``NOT VERIFIED``; only ``scripts/validate_release.py`` fills them in.
* Only non-secret values are read: model names, prompt versions, numeric RAG parameters. Settings such as
  ``ADMIN_API_TOKEN`` are never touched, and :func:`assert_no_secrets` guards the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATES = ("functional", "security", "reliability", "ai_quality", "release_engineering", "static_checks", "docker")
NOT_VERIFIED = "NOT VERIFIED"
_SECRET_RE = re.compile(r"(?i)(password|secret|api[_-]?key|token|authorization|bearer)\s*[:=]\s*\S+")


def _sha256_file(rel: str) -> Optional[str]:
    path = ROOT / rel
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None


def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": "unavailable (not a git repository or git missing)", "dirty": None}
    return {"commit": commit, "dirty": bool(_git("status", "--porcelain"))}


def docker_base_images() -> list[str]:
    path = ROOT / "Dockerfile"
    if not path.is_file():
        return []
    return re.findall(r"(?mi)^FROM\s+(\S+)", path.read_text(encoding="utf-8", errors="replace"))


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def assert_no_secrets(manifest: dict[str, Any]) -> None:
    text = _canonical(manifest)
    if _SECRET_RE.search(text):
        raise ValueError("release manifest would contain a secret-looking value")


def build_manifest(gates: Optional[dict[str, str]] = None, *, now: Optional[datetime] = None) -> dict[str, Any]:
    from app.core import runtime_config
    from app.core.config import get_settings
    from app.version import MODULE_VERSION

    cfg = runtime_config.build_effective(None)  # environment configuration; no admin/active-version I/O
    settings = get_settings()
    git = git_state()
    inputs = {
        "application_version": MODULE_VERSION,
        "git_commit": git["commit"],
        "requirements_sha256": _sha256_file("requirements.txt"),
        "dockerfile_sha256": _sha256_file("Dockerfile"),
        "ai_config_fingerprint": cfg.fingerprint,
    }
    fingerprint = hashlib.sha256(_canonical(inputs).encode()).hexdigest()
    verdicts = {g: NOT_VERIFIED for g in GATES}
    verdicts.update(gates or {})
    manifest: dict[str, Any] = {
        "release_id": f"{MODULE_VERSION}+{fingerprint[:12]}",
        "application_version": MODULE_VERSION,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "build_timestamp": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": settings.app_env,
        "ai": {
            "ai_version_id": cfg.version_id,
            "llm_model": cfg.llm_model,
            "embedding_model": cfg.embedding_model,
            "prompt_versions": dict(cfg.prompt_versions),
            "config_fingerprint": cfg.fingerprint,
            "rag_fingerprint": cfg.rag_fingerprint,
            "model_tags_pinned": all(":" in m and not m.endswith(":latest") for m in (cfg.llm_model, cfg.embedding_model)),
        },
        "build": {
            "python_version": platform.python_version(),
            "requirements_sha256": inputs["requirements_sha256"],
            "dockerfile_sha256": inputs["dockerfile_sha256"],
            "docker_base_images": docker_base_images(),
        },
        "gates": verdicts,
        "inputs_fingerprint": fingerprint,
    }
    assert_no_secrets(manifest)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gates", type=Path, default=None, help="JSON {gate: PASS|FAIL|PARTIAL|NOT VERIFIED}")
    a = p.parse_args(argv)
    gates = json.loads(a.gates.read_text(encoding="utf-8")) if a.gates else None
    text = json.dumps(build_manifest(gates), indent=2, sort_keys=True)
    if a.out:
        if a.out.exists():
            print(f"ERROR: refusing to overwrite {a.out}", file=sys.stderr)
            return 1
        a.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
