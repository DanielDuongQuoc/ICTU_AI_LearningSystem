"""Grade every submission once, saving progress after each one so an interrupted run can resume."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from agent.backends import GradingBackend
from agent.config import AgentConfig
from agent.scanner import Submission, scan
from agent.selector import build_answer, select_files
from app.core.exceptions import AppError, ErrorCode
from app.schemas.grading import GradeRequest
from app.schemas.rubric import Rubric

logger = logging.getLogger("agent")

# Provider-side problems: stop the run (resume later) instead of marking every remaining student as failed.
STOP_CODES = frozenset({ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_GENERATION_ERROR,
                        ErrorCode.SERVICE_OVERLOADED, ErrorCode.RATE_LIMITED, ErrorCode.LLM_MODEL_NOT_FOUND})


class RunStopped(Exception):
    """Raised when grading must pause (quota / provider outage). Progress is already saved."""


@dataclass
class Assignment:
    question: str
    rubric: Rubric
    expected_answer: str = ""


@dataclass
class RunSummary:
    graded: int = 0
    skipped_done: int = 0
    stopped: bool = False
    stop_reason: str = ""
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_assignment(cfg: AgentConfig) -> Assignment:
    rubric = Rubric.model_validate(json.loads(cfg.rubric_file.read_text(encoding="utf-8")))
    expected = cfg.expected_answer_file.read_text(encoding="utf-8").strip() if cfg.expected_answer_file else ""
    return Assignment(question=cfg.question_file.read_text(encoding="utf-8").strip(), rubric=rubric,
                      expected_answer=expected)


def load_state(cfg: AgentConfig) -> dict[str, dict[str, Any]]:
    if cfg.state_file.is_file():
        return json.loads(cfg.state_file.read_text(encoding="utf-8"))
    return {}


def save_state(cfg: AgentConfig, state: dict[str, dict[str, Any]]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(cfg.state_file)  # atomic: a crash never leaves a half-written state file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _local_entry(cfg: AgentConfig, sub: Submission, flags: list[str], note: str,
                 selection: Optional[dict] = None) -> dict[str, Any]:
    """Result produced without calling the model (nothing gradable): always goes to the lecturer."""
    return {
        "student_ref": sub.student_ref, "assessment_context": cfg.assessment_context,
        "status": "needs_review" if cfg.assessment_context == "homework" else "pending_review",
        "result": None, "suggested_total": None, "unmet_required": [], "criteria": [],
        "feedback": note, "flags": flags, "warnings": [], "files": selection or {}, "graded_at": _now(),
    }


async def run(
    cfg: AgentConfig,
    backend: GradingBackend,
    *,
    assignment: Optional[Assignment] = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> RunSummary:
    assignment = assignment or load_assignment(cfg)
    state = load_state(cfg)
    summary = RunSummary(entries=state)
    scanned = scan(cfg)
    for name, reason in scanned.ignored:
        progress(f"[skip] {name}: {reason}")
    total = len(scanned.submissions)
    for i, sub in enumerate(scanned.submissions, 1):
        ref = sub.student_ref or sub.source.name
        if ref in state:
            summary.skipped_done += 1
            continue
        if sub.errors or sub.root is None:
            state[ref] = _local_entry(cfg, sub, ["unreadable"], "Không đọc được bài nộp: " + "; ".join(sub.errors))
            save_state(cfg, state)
            progress(f"[{i}/{total}] {ref}: unreadable")
            continue
        selection = select_files(sub.root, cfg)
        answer, truncated, included = build_answer(selection.accepted, cfg.max_total_kb * 1024)
        files_info = {"included": included, "rejected": [f"{p} ({r})" for p, r in selection.rejected]}
        if not selection.accepted:
            state[ref] = _local_entry(cfg, sub, ["no_valid_files"],
                                      "Không có file nào đúng định dạng được chấp nhận.", files_info)
            save_state(cfg, state)
            progress(f"[{i}/{total}] {ref}: no_valid_files")
            continue
        req = GradeRequest(
            assessment_context=cfg.assessment_context, submission_kind="code", question=assignment.question,
            rubric=assignment.rubric, answer=answer, expected_answer=assignment.expected_answer or None,
            submission_ref=ref, language=cfg.language, extra_flags=["truncated"] if truncated else [],
        )
        try:
            result = await backend.grade(req)
        except AppError as exc:
            if exc.code in STOP_CODES:
                summary.stopped, summary.stop_reason = True, f"{exc.code}: {exc.message}"
                progress(f"[stop] {exc.code} - run `grade` again later to continue")
                break
            state[ref] = _local_entry(cfg, sub, ["grading_error"], f"Lỗi khi chấm: {exc.code}", files_info)
            save_state(cfg, state)
            progress(f"[{i}/{total}] {ref}: error {exc.code}")
            continue
        entry = result.model_dump(mode="json")
        entry.update({"student_ref": ref, "files": files_info, "graded_at": _now()})
        state[ref] = entry
        save_state(cfg, state)
        summary.graded += 1
        verdict = result.result or (f"{result.suggested_total}/{result.max_score}" if result.suggested_total
                                    is not None else "-")
        progress(f"[{i}/{total}] {ref}: {verdict} ({result.status})")
    summary.entries = state
    return summary
