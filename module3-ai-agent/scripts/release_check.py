"""Static release checks for Module 3.

    python scripts/release_check.py               # static checks
    python scripts/release_check.py --run-tests   # + full pytest

Exit code 0 only when no check FAILED. Live behaviour is covered by ``scripts/smoke_test.py``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Result = tuple[str, str, str]  # (name, PASS|FAIL, detail)
SECRET_RE = re.compile(r"(?i)^\s*[A-Z_]*(PASSWORD|SECRET|API_KEY|TOKEN(?!S))[A-Z_]*[ \t]*=[ \t]*\S+", re.M)
REQUIRED_DOCS = ["README.md", "docs/INTEGRATION_CONTRACT.md", "docs/SECURITY.md"]
# Public contract consumed by the C# backend and other modules (method, path).
CONTRACT = {
    ("get", "/health"), ("get", "/health/live"), ("get", "/health/ready"), ("get", "/ready"),
    ("post", "/api/documents/upload"), ("get", "/api/documents/{document_id}"),
    ("delete", "/api/documents/{document_id}"), ("post", "/api/documents/{document_id}/search"),
    ("post", "/api/rag/retrieve"), ("post", "/api/rag/answer"),
    ("post", "/api/questions/generate"), ("post", "/api/questions/generate-matrix"),
    ("post", "/api/evaluations/evaluate"), ("post", "/api/evaluations/evaluate/stream"),
    ("post", "/api/rubrics/generate"), ("post", "/api/rubrics/validate"),
    ("post", "/api/grading/grade"), ("post", "/api/grading/batch"),
    ("post", "/api/submissions/handwritten/transcribe"),
}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def check_env_files() -> Result:
    problems = []
    for name in (".gitignore", ".dockerignore"):
        if ".env" not in {ln.strip() for ln in _read(name).splitlines()}:
            problems.append(f".env missing from {name}")
    if SECRET_RE.search(_read(".env.example")):
        problems.append(".env.example contains a secret-looking value")
    return ("env files / ignore rules", "FAIL" if problems else "PASS", "; ".join(problems))


def _env_example_values() -> dict[str, str]:
    values = {}
    for line in _read(".env.example").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip().lower()] = v.strip()
    return values


def check_env_example() -> Result:
    from app.core.config import Settings

    values = _env_example_values()
    missing = sorted(set(Settings.model_fields) - set(values))
    unknown = sorted(set(values) - set(Settings.model_fields))
    problems = ([f"missing: {', '.join(missing)}"] if missing else []) + ([f"unknown: {', '.join(unknown)}"] if unknown else [])
    try:
        Settings(_env_file=None, **values)
    except Exception as exc:
        problems.append(f"invalid: {str(exc)[:200]}")
    return (".env.example covers every setting and is valid", "FAIL" if problems else "PASS", "; ".join(problems))


def check_requirements() -> Result:
    loose = [ln for ln in _read("requirements.txt").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#") and "==" not in ln]
    return ("requirements pinned", "FAIL" if loose else "PASS", ", ".join(loose))


def check_dockerfile() -> Result:
    problems = []
    dockerfile = _read("Dockerfile")
    if not re.search(r"(?m)^USER\s+(?!root)\S+", dockerfile):
        problems.append("does not switch to a non-root USER")
    if re.search(r"(?mi)^(COPY|ADD)\s.*\.env(\s|$)", dockerfile):
        problems.append("copies .env")
    # Render builds with either the repo root or module3-ai-agent/ as context: both COPY forms must stay.
    for needle in ("COPY requirements.tx[t] module3-ai-agent/requirements.tx[t] ./",
                   "ap[p] module3-ai-agent/ap[p] ./app/", "${PORT:-${APP_PORT}}", "HEALTHCHECK"):
        if needle not in dockerfile:
            problems.append(f"missing {needle!r}")
    return ("Dockerfile (non-root, no .env, dual build context, $PORT)", "FAIL" if problems else "PASS", "; ".join(problems))


def check_openapi() -> Result:
    from app.main import app

    spec = app.openapi()
    have = {(m, p) for p, ops in spec["paths"].items() for m in ops}
    missing = sorted(CONTRACT - have)
    undocumented = [f"{m.upper()} {p}" for p, ops in spec["paths"].items() for m, op in ops.items()
                    if not (op.get("summary") and op.get("description"))]
    problems = ([f"missing: {missing}"] if missing else []) + ([f"undocumented: {undocumented}"] if undocumented else [])
    return ("OpenAPI contract paths present and documented", "FAIL" if problems else "PASS", "; ".join(problems))


def check_version() -> Result:
    from app.api.health import API_VERSION
    from app.main import app
    from app.version import MODULE_VERSION

    ok = app.version == API_VERSION == MODULE_VERSION and not MODULE_VERSION.startswith("1.")
    return ("version consistent (pre-1.0)", "PASS" if ok else "FAIL", MODULE_VERSION)


def check_docs() -> Result:
    missing = [d for d in REQUIRED_DOCS if not (ROOT / d).is_file()]
    return ("documentation present", "FAIL" if missing else "PASS", ", ".join(missing))


def check_pytest() -> Result:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    summary = next((ln for ln in reversed(proc.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    return ("pytest", "PASS" if proc.returncode == 0 else "FAIL", summary)


STATIC_CHECKS: list[Callable[[], Result]] = [
    check_env_files, check_env_example, check_requirements, check_dockerfile, check_openapi, check_version, check_docs,
]


def run_checks(*, run_tests: bool = False) -> list[Result]:
    results: list[Result] = []
    for check in STATIC_CHECKS + ([check_pytest] if run_tests else []):
        try:
            results.append(check())
        except Exception as exc:  # a crashing check is a failed check
            results.append((check.__name__, "FAIL", f"{type(exc).__name__}: {exc}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)
    results = run_checks(run_tests=args.run_tests)
    for name, status, detail in results:
        print(f"{status:5s} {name}" + (f" - {detail}" if detail else ""))
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
