"""Structured course information sent by the main system (never free-form instructions)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

Language = str


def _clean(v: str) -> str:
    return " ".join(v.split())


class CourseLearningOutcome(BaseModel):
    code: str = Field(..., min_length=1, max_length=32)
    description: str = Field(..., min_length=1, max_length=500)

    @field_validator("code", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = _clean(v)
        if not v:
            raise ValueError("must not be blank")
        return v


class CourseContext(BaseModel):
    course_code: Optional[str] = Field(None, max_length=32)
    course_name: str = Field(..., min_length=1, max_length=200)
    clos: List[CourseLearningOutcome] = Field(default_factory=list, max_length=20)
    chapters: List[str] = Field(default_factory=list, max_length=30)
    language: str = Field("vi", pattern=r"^[a-z]{2}$")

    @field_validator("course_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = _clean(v)
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("course_code")
    @classmethod
    def _strip_code(cls, v: Optional[str]) -> Optional[str]:
        return _clean(v) or None if v is not None else None

    @field_validator("chapters")
    @classmethod
    def _strip_chapters(cls, v: List[str]) -> List[str]:
        return [c for c in (_clean(x)[:200] for x in v) if c]

    def as_data_block(self) -> str:
        """Plain-text rendering placed inside the COURSE DATA block of a prompt."""
        lines = [f"Course: {self.course_name}" + (f" ({self.course_code})" if self.course_code else "")]
        if self.clos:
            lines.append("Learning outcomes:")
            lines += [f"- {c.code}: {c.description}" for c in self.clos]
        if self.chapters:
            lines.append("Chapters:")
            lines += [f"- {c}" for c in self.chapters]
        return "\n".join(lines)
