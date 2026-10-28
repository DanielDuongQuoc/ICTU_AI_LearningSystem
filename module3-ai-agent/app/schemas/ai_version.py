"""Pydantic schemas for AI version governance (Phase 18)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.prompt_registry import TASKS

VersionStatus = Literal[
    "draft", "validated", "approved", "active", "superseded", "rolled_back", "rejected", "inconclusive"
]
GateVerdict = Literal[
    "PASS", "FAIL_SAFETY", "FAIL_FUNCTIONAL", "FAIL_QUALITY", "FAIL_INTEGRITY", "INCONCLUSIVE", "BLOCKED"
]

_SECRET_LIKE = re.compile(
    r"(api[_-]?key|secret|passw(or)?d|token|bearer\s+\S+|authorization\s*[:=]|sk-[A-Za-z0-9]{8,})", re.I
)
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,99}$")


def _reject_secret_like(value: str, field: str) -> str:
    if _SECRET_LIKE.search(value):
        raise ValueError(f"{field} must not contain credentials or secret-looking text")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Strict):
    provider: Literal["ollama"] = "ollama"
    model: str = Field(..., description="Exact Ollama model name AND tag, name:tag as shown by `ollama list`")
    temperature: float = Field(..., ge=0.0, le=2.0)
    num_predict: int = Field(..., ge=16, le=8192)

    @field_validator("model")
    @classmethod
    def _model_ok(cls, v: str) -> str:
        if not _MODEL_NAME.match(v):
            raise ValueError("model must be a plain Ollama model name (letters, digits, . _ : / -)")
        return _reject_secret_like(v, "model")


class RAGConfig(_Strict):
    top_k: int = Field(..., ge=1, le=20)
    similarity_threshold: float = Field(..., ge=0.0, le=1.0)
    max_context_chars: int = Field(..., ge=500, le=200_000)


class ChunkingConfig(_Strict):
    chunk_size: int = Field(..., ge=100, le=8000)
    chunk_overlap: int = Field(..., ge=0, le=4000)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        return self


class AIConfig(_Strict):
    llm: LLMConfig
    embedding_model: str
    prompt_versions: dict[str, str]
    rag: RAGConfig
    chunking: ChunkingConfig

    @field_validator("embedding_model")
    @classmethod
    def _emb_ok(cls, v: str) -> str:
        if not _MODEL_NAME.match(v):
            raise ValueError("embedding_model must be a plain Ollama model name")
        return _reject_secret_like(v, "embedding_model")

    @field_validator("prompt_versions")
    @classmethod
    def _prompts_complete(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v) != set(TASKS):
            raise ValueError(f"prompt_versions must define exactly these tasks: {sorted(TASKS)}")
        return v


class AIVersionCreate(_Strict):
    """Body of ``POST /api/ai-versions``. There is deliberately NO ``status`` field: a version is
    always created as ``draft`` and can only move forward through the gated lifecycle."""

    description: str = Field(..., min_length=3, max_length=300)
    config: AIConfig

    @field_validator("description")
    @classmethod
    def _desc_ok(cls, v: str) -> str:
        return _reject_secret_like(v.strip(), "description")


class ReasonBody(_Strict):
    reason: str = Field(..., min_length=3, max_length=500)

    @field_validator("reason")
    @classmethod
    def _reason_ok(cls, v: str) -> str:
        return _reject_secret_like(v.strip(), "reason")


class RollbackBody(ReasonBody):
    target_version_id: str = Field(..., min_length=3, max_length=80)


class EvaluateBody(_Strict):
    baseline_version_id: Optional[str] = Field(
        None, description="Compare against this version's stored evidence (default: the active version)")


class SuiteSummary(BaseModel):
    mode: str
    cases: int
    run_id: str
    metrics: dict[str, Optional[float]] = {}


class EvaluationSummary(BaseModel):
    verdict: GateVerdict
    evaluated_at: datetime
    reasons: list[str]
    suites: dict[str, SuiteSummary] = {}
    baseline_version_id: Optional[str] = None
    policy_fingerprint: str
    evidence_sha256: dict[str, str] = {}
    functional_tests: Optional[dict[str, Any]] = None


class AIVersionRecord(BaseModel):
    version_id: str
    config_fingerprint: str
    status: VersionStatus
    status_reason: Optional[str] = None
    description: str
    created_at: datetime
    created_by: str
    config: AIConfig
    prompt_fingerprints: dict[str, str]
    change_summary: list[str]
    previous_version_id: Optional[str] = None
    evaluation: Optional[EvaluationSummary] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    activated_at: Optional[datetime] = None
    updated_at: datetime


class AIVersionList(BaseModel):
    active_version_id: Optional[str]
    revision: int
    versions: list[AIVersionRecord]


class ActiveVersionResponse(BaseModel):
    active_version_id: Optional[str]
    source: str
    effective_config: dict[str, Any]
    effective_fingerprint: str
    record: Optional[AIVersionRecord] = None
    drift: list[str] = []
    store_degraded: Optional[str] = None
    note: str = ""


class AuditEvent(BaseModel):
    ts: datetime
    event: str
    version_id: Optional[str] = None
    actor: str
    request_id: Optional[str] = None
    from_status: Optional[str] = None
    to_status: Optional[str] = None
    reason: Optional[str] = None
    details: dict[str, Any] = {}
