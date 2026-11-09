"""Sandboxed test runs for code submissions (Docker is mocked; nothing is executed for real)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agent.sandbox as sb
from agent.runner import run
from agent.sandbox import parse_counts, run_tests
from tests.test_agent import ServiceBackend, assignment  # noqa: F401  (fixture)
from tests.test_grading import FakeLLM, judgement


@pytest.mark.parametrize("out,expected", [
    ("===== 3 passed, 1 failed in 0.2s =====", (3, 1)),
    ("5 passed in 0.1s", (5, 0)),
    ("Failed!  - Failed:     2, Passed:     7, Skipped: 0", (7, 2)),
    ("Passed!  - Passed: 4 Failed: 0", (4, 0)),
    ("compilation error", (None, None)),
])
def test_parse_counts(out, expected):
    assert parse_counts(out) == expected


def cfg_with_tests(cfg, tmp_path: Path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    return cfg.model_copy(update={"tests_dir": tests, "test_command": "cd /tmp && python -m pytest -q"})


def test_docker_command_is_locked_down(assignment, tmp_path):  # noqa: F811
    cfg = cfg_with_tests(assignment, tmp_path)
    cmd = sb.docker_command(tmp_path, cfg)
    joined = " ".join(cmd)
    for needle in ("--network none", "--read-only", "--pids-limit 64", "--memory 256m", "no-new-privileges",
                   ":/work/src:ro", ":/work/tests:ro"):
        assert needle in joined, needle


def test_no_tests_or_no_docker_means_not_executed(assignment, tmp_path, monkeypatch):  # noqa: F811
    assert run_tests(tmp_path, assignment).executed is False
    monkeypatch.setattr(sb, "docker_available", lambda: False)
    out = run_tests(tmp_path, cfg_with_tests(assignment, tmp_path))
    assert out.executed is False and "Docker" in out.reason


def test_run_and_timeout(assignment, tmp_path, monkeypatch):  # noqa: F811
    cfg = cfg_with_tests(assignment, tmp_path)
    monkeypatch.setattr(sb, "docker_available", lambda: True)
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "2 passed, 1 failed", ""))
    out = run_tests(tmp_path, cfg)
    assert out.executed and (out.passed, out.failed, out.exit_code) == (2, 1, 1)
    assert "2 passed, 1 failed" in out.summary

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(sb.subprocess, "run", boom)
    assert run_tests(tmp_path, cfg).timed_out


async def test_runner_passes_test_results_or_flags_not_executed(assignment, tmp_path, monkeypatch):  # noqa: F811
    cfg = cfg_with_tests(assignment, tmp_path)
    monkeypatch.setattr(sb, "docker_available", lambda: True)
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "4 passed", ""))
    backend = ServiceBackend(FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Đầy đủ", "C3": "Đầy đủ"})))
    st = (await run(cfg, backend)).entries
    assert all("4 passed" in (r.test_results or "") for r in backend.requests)
    assert st["SV001"]["files"]["tests"]["passed"] == 4 and st["SV001"]["status"] == "final"

    cfg2 = cfg.model_copy(update={"output_dir": tmp_path / "results2"})
    monkeypatch.setattr(sb, "docker_available", lambda: False)
    st2 = (await run(cfg2, ServiceBackend(FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Đầy đủ", "C3": "Đầy đủ"}))))).entries
    assert st2["SV001"]["status"] == "needs_review" and "not_executed" in st2["SV001"]["flags"]

