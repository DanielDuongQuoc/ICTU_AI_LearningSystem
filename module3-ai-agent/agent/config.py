"""Agent configuration (one YAML file per assignment). Relative paths are resolved from the YAML file's folder."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_EXTENSIONS = [".cs", ".java", ".py", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".sql", ".html", ".css"]
DEFAULT_EXCLUDE_DIRS = ["bin", "obj", "node_modules", ".git", ".vs", ".vscode", ".idea", "__pycache__",
                        "venv", ".venv", "dist", "build", "target", "packages", "out"]
# Student code (MSSV) = letters/digits containing at least one digit, then optional "_Name" / "-Name" / " Name".
DEFAULT_NAMING_PATTERN = r"^(?P<student_ref>(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+)(?:[_\- .].*)?$"


class AgentConfig(BaseModel):
    assessment_context: Literal["homework", "in_class"]
    submissions_dir: Path
    question_file: Path
    rubric_file: Path
    output_dir: Path
    expected_answer_file: Optional[Path] = None
    naming_pattern: str = DEFAULT_NAMING_PATTERN
    accepted_extensions: List[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    exclude_dirs: List[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    max_file_kb: int = Field(200, gt=0)
    max_total_kb: int = Field(58, gt=0, description="Text sent to the model per submission (<= MAX_SUBMISSION_CHARS)")
    max_unzipped_mb: int = Field(50, gt=0)
    max_zip_entries: int = Field(2000, gt=0)
    mode: Literal["local", "api"] = "local"
    api_url: str = "http://localhost:8003"
    api_timeout_s: float = Field(180.0, gt=0)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")

    @field_validator("accepted_extensions")
    @classmethod
    def _norm_ext(cls, v: List[str]) -> List[str]:
        out = []
        for e in v:
            e = e.strip().lower()
            if e and not e.startswith("."):
                e = "." + e
            if e:
                out.append(e)
        if not out:
            raise ValueError("accepted_extensions must not be empty")
        return out

    @field_validator("exclude_dirs")
    @classmethod
    def _norm_dirs(cls, v: List[str]) -> List[str]:
        return [d.strip().lower() for d in v if d.strip()]

    @field_validator("naming_pattern")
    @classmethod
    def _check_pattern(cls, v: str) -> str:
        rx = re.compile(v)
        if "student_ref" not in rx.groupindex:
            raise ValueError("naming_pattern must contain a named group (?P<student_ref>...)")
        return v

    @property
    def naming_regex(self) -> re.Pattern[str]:
        return re.compile(self.naming_pattern)

    @property
    def state_file(self) -> Path:
        return self.output_dir / "state.json"

    @property
    def work_dir(self) -> Path:
        return self.output_dir / ".work"


def load_config(path: str | Path) -> AgentConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    for key in ("submissions_dir", "question_file", "rubric_file", "output_dir", "expected_answer_file"):
        if data.get(key):
            p = Path(data[key])
            data[key] = p if p.is_absolute() else (base / p).resolve()
    return AgentConfig.model_validate(data)
