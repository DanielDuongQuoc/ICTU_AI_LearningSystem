"""Phase 18 - produce evaluation EVIDENCE for a candidate AI version (host-side; not part of the Docker image).

The promotion gate (``POST /api/ai-versions/{id}/evaluate``) never runs models itself; it judges evidence
that this script stores under ``<AI_VERSIONS_DIR>/evidence/<version_id>/``:

    rag.json  qgen.json  answer_eval.json   Phase 17 reports (same evaluators, same datasets)
    functional_tests.json                    result of the deterministic pytest run
    evidence_status.json                     infrastructure problems (so they read as BLOCKED, not as a failure)

Commands::

    python scripts/ai_version_cli.py show-env                       # JSON body of the CURRENT .env config
    python scripts/ai_version_cli.py evaluate <version_id> --mode ollama
    python scripts/ai_version_cli.py evaluate <version_id> --mode offline   # harness check only: gate => INCONCLUSIVE

The candidate's model, temperature, top_k, threshold, context limit and prompt versions are applied to the
evaluators for the duration of the run (environment overrides + a pinned runtime configuration); the
production configuration and the live ChromaDB collection are not touched. ``--mode ollama`` needs a running
Ollama with the candidate models installed; if it is not reachable the run is recorded as an infrastructure
error and the gate reports BLOCKED.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITE_SCRIPTS = {"rag": "evaluate_rag", "qgen": "evaluate_question_generation",
                 "answer_eval": "evaluate_answer_evaluation"}
SUITE_PREFIX = {"rag": "rag_eval", "qgen": "qgen_eval", "answer_eval": "answer_eval"}


def _store_dir() -> Path:
    from app.core.config import get_settings

    return Path(get_settings().ai_versions_dir)


def show_env() -> int:
    from app.core import runtime_config as rc

    print(json.dumps({"description": "snapshot of the current environment configuration",
                      "config": rc.build_effective(None).config_dict()}, indent=2))
    return 0


def _env_overrides(config: dict[str, Any]) -> dict[str, str]:
    return {
        "OLLAMA_MODEL": config["llm"]["model"], "OLLAMA_TEMPERATURE": str(config["llm"]["temperature"]),
        "OLLAMA_NUM_PREDICT": str(config["llm"]["num_predict"]), "TOP_K": str(config["rag"]["top_k"]),
        "SIMILARITY_THRESHOLD": str(config["rag"]["similarity_threshold"]),
        "MAX_CONTEXT_CHARS": str(config["rag"]["max_context_chars"]),
    }


def run_functional(out_file: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
           "-o", "log_cli=false"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|error|errors)", tail)}
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    result = {"passed": proc.returncode == 0 and failed == 0 and counts.get("passed", 0) > 0,
              "total": counts.get("passed", 0) + failed, "failed": failed, "exit_code": proc.returncode,
              "command": "pytest -q (integration tests deselected by default)", "summary_line": tail[:200]}
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate(version_id: str, mode: str, skip_functional: bool) -> int:
    from app.core import runtime_config as rc
    from app.core.config import get_settings
    from app.repositories.ai_version_repository import AIVersionRepository
    from app.services.ai_version_service import override_from_manifest

    repo = AIVersionRepository(_store_dir())
    manifest = repo.read_manifest(version_id)
    if manifest is None:
        print(f"unknown version {version_id!r} in {repo.root}", file=sys.stderr)
        return 2
    evidence = repo.evidence_dir(version_id)
    evidence.mkdir(parents=True, exist_ok=True)
    infra: list[str] = []

    saved = {k: os.environ.get(k) for k in _env_overrides(manifest["config"])}
    os.environ.update(_env_overrides(manifest["config"]))
    get_settings.cache_clear()
    try:
        effective = rc.build_effective(override_from_manifest(manifest))
        import importlib

        with rc.pinned(effective), tempfile.TemporaryDirectory() as tmp:
            for suite, module_name in SUITE_SCRIPTS.items():
                out_dir = Path(tmp) / suite
                out_dir.mkdir()
                module = importlib.import_module(f"scripts.{module_name}")
                try:
                    code = module.main(["--mode", mode, "--out-dir", str(out_dir)])
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                    infra.append(f"{suite}: evaluator crashed ({type(exc).__name__}: {str(exc)[:160]})")
                    continue
                reports = sorted(out_dir.glob(f"{SUITE_PREFIX[suite]}*.json"))
                if not reports:
                    infra.append(f"{suite}: evaluator produced no report (exit code {code}); is Ollama reachable?")
                    continue
                shutil.copyfile(reports[-1], evidence / f"{suite}.json")
                print(f"{suite}: stored {reports[-1].name} (evaluator exit {code})")
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        get_settings.cache_clear()

    if skip_functional:
        print("functional tests skipped: the gate will report BLOCKED")
        (evidence / "functional_tests.json").unlink(missing_ok=True)
    else:
        result = run_functional(evidence / "functional_tests.json")
        print(f"functional tests: passed={result['passed']} ({result['summary_line']})")
    (evidence / "evidence_status.json").write_text(json.dumps({"infrastructure_errors": infra}, indent=2),
                                                   encoding="utf-8")
    for problem in infra:
        print(f"INFRASTRUCTURE: {problem}", file=sys.stderr)
    print(f"evidence stored in {evidence}; now call POST /api/ai-versions/{version_id}/evaluate")
    return 0 if not infra else 3


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show-env")
    e = sub.add_parser("evaluate")
    e.add_argument("version_id")
    e.add_argument("--mode", choices=["offline", "ollama"], default="ollama")
    e.add_argument("--skip-functional", action="store_true")
    args = p.parse_args(argv)
    return show_env() if args.cmd == "show-env" else evaluate(args.version_id, args.mode, args.skip_functional)


if __name__ == "__main__":
    raise SystemExit(main())
