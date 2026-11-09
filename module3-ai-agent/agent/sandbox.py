"""Run the lecturer's tests against a code submission inside a locked-down Docker container.

docker run --rm --network none --cpus 1 --memory 256m --pids-limit 64 --read-only --tmpfs /tmp
           -v <submission>:/work/src:ro -v <tests_dir>:/work/tests:ro -w /work <image> sh -c "<test_command>"

The submission and the tests are mounted read-only; ``test_command`` should copy what it needs into /tmp, e.g.
``cp -r /work/src /tmp/s && cp -r /work/tests/* /tmp/s && cd /tmp/s && python -m pytest -q``.
Without Docker (or without ``tests_dir``) nothing is executed and the caller marks the result ``not_executed``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from agent.config import AgentConfig

MAX_OUTPUT_CHARS = 3000
_PASSED = re.compile(r"(\d+)\s+passed", re.I)
_FAILED = re.compile(r"(\d+)\s+(failed|errors?)", re.I)
_DOTNET = re.compile(r"Passed:\s*(\d+).*?Failed:\s*(\d+)|Failed:\s*(\d+),\s*Passed:\s*(\d+)", re.I | re.S)


@dataclass
class TestOutcome:
    executed: bool
    passed: Optional[int] = None
    failed: Optional[int] = None
    exit_code: Optional[int] = None
    timed_out: bool = False
    output_tail: str = ""
    reason: str = ""

    @property
    def summary(self) -> str:
        if not self.executed:
            return f"Tests not executed ({self.reason})."
        if self.timed_out:
            return "Tests timed out (possible infinite loop)."
        counts = ""
        if self.passed is not None or self.failed is not None:
            counts = f"{self.passed or 0} passed, {self.failed or 0} failed. "
        verdict = "All tests passed." if self.exit_code == 0 else f"Test run exit code {self.exit_code}."
        return f"{counts}{verdict}\nLast output:\n{self.output_tail}"


@lru_cache(maxsize=1)
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def parse_counts(output: str) -> tuple[Optional[int], Optional[int]]:
    m = _DOTNET.search(output)
    if m:
        if m.group(1) is not None:
            return int(m.group(1)), int(m.group(2))
        return int(m.group(4)), int(m.group(3))
    passed = _PASSED.findall(output)
    failed = _FAILED.findall(output)
    p = int(passed[-1]) if passed else None
    f = int(failed[-1][0]) if failed else (0 if passed else None)
    return p, f


def docker_command(src: Path, cfg: AgentConfig) -> list[str]:
    assert cfg.tests_dir is not None and cfg.test_command
    return [
        "docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", f"{cfg.test_memory_mb}m",
        "--pids-limit", "64", "--read-only", "--tmpfs", "/tmp:rw,exec,size=128m", "--security-opt",
        "no-new-privileges", "-v", f"{src.resolve()}:/work/src:ro", "-v", f"{cfg.tests_dir.resolve()}:/work/tests:ro",
        "-w", "/work", cfg.docker_image, "sh", "-c", cfg.test_command,
    ]


def run_tests(src: Path, cfg: AgentConfig) -> TestOutcome:
    if cfg.tests_dir is None or not cfg.test_command:
        return TestOutcome(executed=False, reason="no tests configured")
    if not docker_available():
        return TestOutcome(executed=False, reason="Docker is not available")
    try:
        proc = subprocess.run(docker_command(src, cfg), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=cfg.test_timeout_s)
    except subprocess.TimeoutExpired:
        return TestOutcome(executed=True, timed_out=True, exit_code=None)
    except OSError as exc:
        return TestOutcome(executed=False, reason=f"docker failed to start: {type(exc).__name__}")
    output = (proc.stdout or "") + (proc.stderr or "")
    passed, failed = parse_counts(output)
    return TestOutcome(executed=True, passed=passed, failed=failed, exit_code=proc.returncode,
                       output_tail=output[-MAX_OUTPUT_CHARS:])
