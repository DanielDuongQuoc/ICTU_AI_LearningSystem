"""Release gate aggregator (Phase 21): ``python -m scripts.validate_release [--skip-tests] [--manifest PATH]``.

Runs every gate that CAN run on this machine and reports PASS / FAIL / PARTIAL / NOT VERIFIED per gate:

* static_checks        scripts/release_check.py
* functional           pytest, excluding the three suites below
* security             pytest tests/security                      (Phase 19)
* reliability          pytest tests/reliability                   (Phase 20)
* release_engineering  pytest tests/release                       (Phase 21: DR drill, restart, rollback...)
* ai_quality           deterministic datasets + scripts/quality_gate.py --deterministic-only (offline). Model
                       quality needs a live Ollama, so this gate is PARTIAL at best here, never PASS.
* docker               docker compose config + build; NOT VERIFIED when Docker is unavailable.

Verdict:  BLOCKED if any gate FAILED (security/functional/reliability/release_engineering/static/ai_quality);
          GO only when every gate is PASS (needs Docker and live-model evidence);
          otherwise GO WITH LIMITATIONS, listing what is NOT VERIFIED. ``NOT VERIFIED`` is never promoted to PASS.
Exit code: 0 for GO / GO WITH LIMITATIONS, 1 for BLOCKED.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import release_manifest  # noqa: E402

BLOCKING = ("static_checks", "functional", "security", "reliability", "release_engineering", "ai_quality")


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _pytest(*args: str) -> tuple[str, str]:
    proc = _run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *args])
    summary = next((ln for ln in reversed(proc.stdout.splitlines()) if "passed" in ln or "failed" in ln or "error" in ln), "")
    if proc.returncode == 0:
        return "PASS", summary
    if proc.returncode == 5:
        return "NOT VERIFIED", "no tests collected"
    return "FAIL", summary or proc.stdout[-200:]


def gate_static() -> tuple[str, str]:
    from scripts.release_check import run_checks

    results = run_checks(docker=False)
    failed = [f"{n}: {d}" for n, s, d in results if s == "FAIL"]
    return ("FAIL", "; ".join(failed)[:300]) if failed else ("PASS", f"{len(results)} static checks")


def gate_ai_quality() -> tuple[str, str]:
    """Deterministic, offline. Evaluates dataset schema + safety invariants only (docs/QUALITY_GATES.md)."""
    out = tempfile.mkdtemp(prefix="m3-quality-")
    try:
        for script in ("evaluate_rag.py", "evaluate_question_generation.py", "evaluate_answer_evaluation.py"):
            for extra in (["--validate-only"], ["--out-dir", out]):
                proc = _run([sys.executable, f"scripts/{script}", *extra], timeout=900)
                if proc.returncode != 0:
                    return "FAIL", f"{script} {' '.join(extra)} exited {proc.returncode}"
        proc = _run([sys.executable, "scripts/quality_gate.py", "--reports", out, "--deterministic-only"], timeout=900)
        if proc.returncode != 0:
            return "FAIL", "quality_gate.py --deterministic-only failed"
        return "PARTIAL", "deterministic checks passed; model-dependent quality NOT VERIFIED (needs live Ollama)"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def gate_docker() -> tuple[str, str]:
    if shutil.which("docker") is None:
        return "NOT VERIFIED", "docker CLI not available on this machine (build, compose up, restart not exercised)"
    proc = _run(["docker", "compose", "config", "-q"], timeout=120)
    if proc.returncode != 0:
        return "FAIL", proc.stderr.strip()[:200]
    return "PARTIAL", "compose file valid; image build / runtime not run by this script"


def verdict(gates: dict[str, str]) -> str:
    if any(gates.get(g) == "FAIL" for g in BLOCKING + ("docker",)):
        return "BLOCKED"
    if all(v == "PASS" for v in gates.values()):
        return "GO"
    return "GO WITH LIMITATIONS"


def run(skip_tests: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    gates: dict[str, str] = {}
    detail: dict[str, str] = {}

    def record(name: str, fn) -> None:
        try:
            gates[name], detail[name] = fn()
        except Exception as exc:  # a crashing gate is a failed gate
            gates[name], detail[name] = "FAIL", f"{type(exc).__name__}: {exc}"

    record("static_checks", gate_static)
    if skip_tests:
        for g in ("functional", "security", "reliability", "release_engineering"):
            gates[g], detail[g] = "NOT VERIFIED", "--skip-tests"
    else:
        record("security", lambda: _pytest("tests/security"))
        record("reliability", lambda: _pytest("tests/reliability"))
        record("release_engineering", lambda: _pytest("tests/release"))
        record("functional", lambda: _pytest("--ignore=tests/security", "--ignore=tests/reliability", "--ignore=tests/release"))
    record("ai_quality", gate_ai_quality)
    record("docker", gate_docker)
    return gates, detail


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--skip-tests", action="store_true", help="do not run pytest suites (gates become NOT VERIFIED)")
    p.add_argument("--manifest", type=Path, default=None, help="write the release manifest (with gate results) here")
    a = p.parse_args(argv)
    gates, detail = run(a.skip_tests)
    v = verdict(gates)
    width = max(len(g) for g in gates)
    for g, s in gates.items():
        print(f"{g.ljust(width)}  {s:13s}  {detail[g]}")
    print(f"\nRELEASE VERDICT: {v}")
    if v != "GO":
        print("Not verified / partial gates keep this from being a full GO; see docs/PHASE21_REPORT.md.")
    manifest = release_manifest.build_manifest(gates)
    manifest["verdict"] = v
    if a.manifest:
        if a.manifest.exists():
            print(f"ERROR: refusing to overwrite {a.manifest}", file=sys.stderr)
            return 1
        a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Manifest written: {a.manifest}")
    return 1 if v == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
