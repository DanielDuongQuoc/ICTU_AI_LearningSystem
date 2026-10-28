"""Release checklist for Module 3 (static checks; optional test / docker steps).

    python scripts/release_check.py                 # fast static checks
    python scripts/release_check.py --run-tests     # + full pytest
    python scripts/release_check.py --docker        # + `docker compose config` (SKIP if Docker missing)

Exit code 0 only when no check FAILED. SKIP is reported honestly and is not a PASS.
Things that need a running stack (build, health, persistence, smoke test) are NOT covered here:
use docs/RELEASE.md and scripts/smoke_test.py.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Result = tuple[str, str, str]  # (name, PASS|FAIL|SKIP, detail)
SECRET_RE = re.compile(r"(?i)^\s*[A-Z_]*(PASSWORD|SECRET|API_KEY|TOKEN)[A-Z_]*[ \t]*=[ \t]*\S+", re.M)
REQUIRED_DOCS = ["README.md", "docs/HANDOVER.md", "docs/INTEGRATION_CONTRACT.md", "docs/DEPLOYMENT.md",
                 "docs/BACKUP_AND_RECOVERY.md", "docs/RELEASE.md", "docs/SECURITY.md",
                 "docs/AI_VERSION_GOVERNANCE.md", "docs/PROMPT_VERSIONING.md", "docs/QUALITY_GATES.md",
                 "docs/MODEL_PROMOTION_ROLLBACK.md", "docs/PHASE18_REPORT.md",
                 "docs/THREAT_MODEL.md", "docs/DATA_PRIVACY.md", "docs/PHASE19_REPORT.md",
                 "docs/OBSERVABILITY.md", "docs/SLO_SLA.md", "docs/RELIABILITY.md",
                 "docs/FAILURE_RECOVERY.md", "docs/PHASE20_REPORT.md",
                 "docs/DEPLOYMENT_RUNBOOK.md", "docs/RELEASE_CHECKLIST.md", "docs/GO_LIVE_CHECKLIST.md",
                 "docs/DISASTER_RECOVERY.md", "docs/RELEASE_ENGINEERING.md", "docs/PHASE21_REPORT.md"]
EXPECTED_PATHS = {
    "/health", "/health/live", "/health/ready", "/ready", "/api/documents/upload",
    "/api/documents/{document_id}", "/api/rag/retrieve", "/api/rag/answer", "/api/questions/generate",
    "/api/evaluations/evaluate", "/api/evaluations/evaluate/stream",
}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def check_env_files() -> Result:
    problems = []
    for name in (".gitignore", ".dockerignore"):
        lines = {ln.strip() for ln in _read(name).splitlines()}
        if ".env" not in lines:
            problems.append(f".env missing from {name}")
    if SECRET_RE.search(_read(".env.example")):
        problems.append(".env.example contains a secret-looking value")
    return ("env files / ignore rules", "FAIL" if problems else "PASS", "; ".join(problems))


def check_env_example_complete() -> Result:
    from app.core.config import Settings

    keys = {m.group(1).lower() for m in re.finditer(r"^\s*([A-Za-z_]+)\s*=", _read(".env.example"), re.M)}
    missing = sorted(set(Settings.model_fields) - keys)
    return ("env.example covers every setting", "FAIL" if missing else "PASS", ", ".join(missing))


def check_env_example_valid() -> Result:
    from app.core.config import Settings

    values = {}
    for line in _read(".env.example").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip().lower()] = v.strip()
    try:
        Settings(_env_file=None, **values)
    except Exception as exc:
        return (".env.example is a valid configuration", "FAIL", str(exc)[:200])
    return (".env.example is a valid configuration", "PASS", "")


def check_requirements() -> Result:
    loose = [ln for ln in _read("requirements.txt").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#") and "==" not in ln]
    return ("requirements pinned", "FAIL" if loose else "PASS", ", ".join(loose))


def check_dockerfile_and_compose() -> Result:
    import yaml

    problems = []
    dockerfile = _read("Dockerfile")
    if not re.search(r"(?m)^USER\s+(?!root)\S+", dockerfile):
        problems.append("Dockerfile does not switch to a non-root USER")
    if re.search(r"(?mi)^(COPY|ADD)\s.*\.env(\s|$)", dockerfile):
        problems.append("Dockerfile copies .env")
    compose = yaml.safe_load(_read("docker-compose.yml"))
    api = compose["services"]["api"]
    env = {str(e).split("=", 1)[0]: str(e).split("=", 1)[1] for e in api.get("environment", []) if "=" in str(e)}
    if env.get("OLLAMA_BASE_URL") != "http://ollama:11434":
        problems.append("api OLLAMA_BASE_URL must be http://ollama:11434 in compose")
    if "healthcheck" not in api:
        problems.append("api has no healthcheck")
    if not any("chroma" in str(v) for v in api.get("volumes", [])):
        problems.append("api has no Chroma volume")
    ports = [str(p) for p in compose["services"]["ollama"].get("ports", [])]
    if any(not p.startswith("127.0.0.1:") for p in ports):
        problems.append("ollama port is exposed beyond loopback")
    return ("Dockerfile / compose hardening", "FAIL" if problems else "PASS", "; ".join(problems))


def check_openapi() -> Result:
    from app.main import app

    spec = app.openapi()
    missing = sorted(EXPECTED_PATHS - set(spec["paths"]))
    undocumented = [f"{m.upper()} {p}" for p, ops in spec["paths"].items() for m, op in ops.items()
                    if not (op.get("summary") and op.get("description"))]
    problems = ([f"missing paths: {missing}"] if missing else []) + ([f"undocumented: {undocumented}"] if undocumented else [])
    return ("OpenAPI complete", "FAIL" if problems else "PASS", "; ".join(problems))


def check_version() -> Result:
    from app.api.health import API_VERSION
    from app.main import app
    from app.version import MODULE_VERSION

    ok = app.version == API_VERSION == MODULE_VERSION and not MODULE_VERSION.startswith("1.")
    return ("version consistent (pre-1.0)", "PASS" if ok else "FAIL", f"{MODULE_VERSION}")


def check_docs() -> Result:
    missing = [d for d in REQUIRED_DOCS if not (ROOT / d).is_file()]
    return ("documentation present", "FAIL" if missing else "PASS", ", ".join(missing))


def check_backup_scripts() -> Result:
    missing = [s for s in ("scripts/backup.py", "scripts/restore.py", "scripts/smoke_test.py") if not (ROOT / s).is_file()]
    return ("backup/restore/smoke scripts present", "FAIL" if missing else "PASS", ", ".join(missing))


def check_no_artifacts() -> Result:
    found = [str(p.relative_to(ROOT)) for pattern in ("benchmark_results.json", "rag_eval_results.json", "*.bak")
             for p in ROOT.glob(pattern)]
    uploads = [p.name for p in (ROOT / "uploads").glob("*") if p.name != ".gitkeep"] if (ROOT / "uploads").is_dir() else []
    return ("no temporary artifacts", "FAIL" if found or uploads else "PASS", ", ".join(found + uploads))


def check_docker_compose_config() -> Result:
    if shutil.which("docker") is None:
        return ("docker compose config", "SKIP", "docker CLI not found")
    proc = subprocess.run(["docker", "compose", "config", "-q"], cwd=ROOT, capture_output=True, text=True)
    return ("docker compose config", "PASS" if proc.returncode == 0 else "FAIL", proc.stderr.strip()[:200])


def check_pytest() -> Result:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    summary = next((ln for ln in reversed(proc.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    return ("pytest", "PASS" if proc.returncode == 0 else "FAIL", summary)


STATIC_CHECKS: list[Callable[[], Result]] = [
    check_env_files, check_env_example_complete, check_env_example_valid, check_requirements,
    check_dockerfile_and_compose, check_openapi, check_version, check_docs, check_backup_scripts,
    check_no_artifacts,
]


def run_checks(*, run_tests: bool = False, docker: bool = False) -> list[Result]:
    checks = list(STATIC_CHECKS)
    if docker:
        checks.append(check_docker_compose_config)
    if run_tests:
        checks.append(check_pytest)
    results: list[Result] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # a crashing check is a failed check
            results.append((check.__name__, "FAIL", f"{type(exc).__name__}: {exc}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args(argv)
    results = run_checks(run_tests=args.run_tests, docker=args.docker)
    for name, status, detail in results:
        print(f"{status:5s} {name}" + (f" - {detail}" if detail else ""))
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
