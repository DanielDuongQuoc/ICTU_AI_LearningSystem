"""CLI entry point: ``python -m agent {scan,grade,export} -c agent.yaml`` (run from module3-ai-agent/)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent.backends import ApiBackend, LocalBackend
from agent.config import load_config
from agent.exporter import export, write_scan_report
from agent.runner import load_assignment, load_state, run
from agent.scanner import scan
from agent.selector import select_files


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_scan(cfg) -> int:
    scanned = scan(cfg)
    _print(f"{'student_ref':<16}{'accepted':>9}{'rejected':>9}  problems")
    for s in scanned.submissions:
        if s.root is None:
            _print(f"{s.student_ref:<16}{0:>9}{0:>9}  {'; '.join(s.errors)}")
            continue
        sel = select_files(s.root, cfg)
        _print(f"{s.student_ref:<16}{len(sel.accepted):>9}{len(sel.rejected):>9}  "
               f"{'no_valid_files' if not sel.accepted else ''}")
    for name, reason in scanned.ignored:
        _print(f"[skip] {name}: {reason}")
    _print(f"report: {write_scan_report(cfg, scanned)}")
    return 0


async def _grade(cfg) -> int:
    assignment = load_assignment(cfg)
    if assignment.rubric.status != "approved":
        _print("rubric_file must contain an APPROVED rubric (\"status\": \"approved\").")
        return 2
    from app.services.grading_service import check_rubric_usable
    from app.core.exceptions import AppError

    try:
        check_rubric_usable(assignment.rubric, cfg.assessment_context)
    except AppError as exc:
        _print(f"rubric rejected: {exc.message}")
        return 2
    backend = LocalBackend() if cfg.mode == "local" else ApiBackend(cfg.api_url, cfg.api_timeout_s)
    try:
        summary = await run(cfg, backend, assignment=assignment, progress=_print)
    finally:
        await backend.close()
    paths = export(cfg, summary.entries)
    _print(f"graded now: {summary.graded}, already done: {summary.skipped_done}, results: {paths[0]}")
    if summary.stopped:
        _print(f"stopped early ({summary.stop_reason}); run the same command again to continue.")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__)
    parser.add_argument("command", choices=["scan", "grade", "export"])
    parser.add_argument("-c", "--config", required=True, help="assignment YAML file")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "scan":
        return cmd_scan(cfg)
    if args.command == "grade":
        return asyncio.run(_grade(cfg))
    paths = export(cfg, load_state(cfg))
    _print(f"results: {paths[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
