"""Effective AI configuration = environment settings overlaid with the ACTIVE approved AI version.

* :func:`current` is the single place services read the quality-relevant parameters from
  (prompt versions, top_k, similarity threshold, context limit, LLM model/temperature/num_predict).
* Only parameters that are safe to change at runtime are overridable. The embedding model and the
  chunking parameters are NEVER overridden: they define how the live ChromaDB vectors were produced,
  so a version may only declare values equal to the running environment (checked at promotion).
* A request pins the configuration when it starts (:func:`pin_current`), so a promotion or rollback
  that happens mid-request affects only LATER requests and a request is never half old / half new.
* Pure and I/O free; the persisted state is loaded by ``AIVersionService`` which calls :func:`set_active`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from app.core import prompt_registry as registry
from app.core.config import get_settings

BOOTSTRAP_VERSION_ID = "env-bootstrap"


@dataclass(frozen=True)
class RuntimeOverride:
    """Runtime-applicable fields of the active approved version."""

    version_id: str
    llm_model: str
    temperature: float
    num_predict: int
    top_k: int
    similarity_threshold: float
    max_context_chars: int
    prompt_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EffectiveConfig:
    version_id: str
    source: str  # "env" (no active version) | "active_version" | "pinned"
    llm_provider: str
    llm_model: str
    temperature: float
    num_predict: int
    embedding_model: str
    top_k: int
    similarity_threshold: float
    max_context_chars: int
    chunk_size: int
    chunk_overlap: int
    prompt_versions: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def prompt_version(self, task: str) -> str:
        return dict(self.prompt_versions).get(task) or registry.DEFAULT_PROMPT_VERSIONS[task]

    def config_dict(self) -> dict[str, Any]:
        """Secret-free description used for fingerprints, manifests and reports."""
        return {
            "llm": {"provider": self.llm_provider, "model": self.llm_model,
                    "temperature": self.temperature, "num_predict": self.num_predict},
            "embedding_model": self.embedding_model,
            "prompt_versions": dict(self.prompt_versions),
            "rag": {"top_k": self.top_k, "similarity_threshold": self.similarity_threshold,
                    "max_context_chars": self.max_context_chars},
            "chunking": {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
        }

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self.config_dict())

    @property
    def rag_fingerprint(self) -> str:
        """Fingerprint of the retrieval-relevant part only (embedding, chunking, RAG parameters)."""
        d = self.config_dict()
        return config_fingerprint({k: d[k] for k in ("embedding_model", "rag", "chunking")})

    def with_version_id(self, version_id: str, source: str) -> "EffectiveConfig":
        return EffectiveConfig(**{**self.__dict__, "version_id": version_id, "source": source})


def canonical_json(data: Any) -> str:
    """Key-order and whitespace independent JSON (floats keep their value)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable fingerprint of a configuration dict INCLUDING the content of the selected prompts.

    Editing prompt text under an unchanged version label changes the fingerprint; reordering JSON
    keys or whitespace does not.
    """
    prompts = config.get("prompt_versions", {})
    payload = {
        "config": config,
        "prompt_content": {t: registry.content_fingerprint(t, v) for t, v in sorted(prompts.items())},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


_active: Optional[RuntimeOverride] = None
_pinned: ContextVar[Optional[EffectiveConfig]] = ContextVar("ai_config_pinned", default=None)


def set_active(override: Optional[RuntimeOverride]) -> None:
    """Atomically swap the process-wide active override (reference assignment)."""
    global _active
    _active = override


def get_active_override() -> Optional[RuntimeOverride]:
    return _active


def build_effective(override: Optional[RuntimeOverride]) -> EffectiveConfig:
    s = get_settings()
    base = EffectiveConfig(
        version_id=BOOTSTRAP_VERSION_ID, source="env", llm_provider="ollama",
        llm_model=s.ollama_model, temperature=s.ollama_temperature, num_predict=s.ollama_num_predict,
        embedding_model=s.ollama_embedding_model, top_k=s.top_k,
        similarity_threshold=s.similarity_threshold, max_context_chars=s.max_context_chars,
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
        prompt_versions=tuple(sorted(registry.DEFAULT_PROMPT_VERSIONS.items())),
    )
    if override is None:
        return base
    return EffectiveConfig(**{
        **base.__dict__, "version_id": override.version_id, "source": "active_version",
        "llm_model": override.llm_model, "temperature": override.temperature,
        "num_predict": override.num_predict, "top_k": override.top_k,
        "similarity_threshold": override.similarity_threshold,
        "max_context_chars": override.max_context_chars, "prompt_versions": override.prompt_versions,
    })


def current() -> EffectiveConfig:
    """Configuration for the running request (pinned) or the live active configuration."""
    pinned = _pinned.get()
    return pinned if pinned is not None else build_effective(_active)


_refresh_hook: Optional[Callable[[], None]] = None


def set_refresh_hook(hook: Optional[Callable[[], None]]) -> None:
    """Register a cheap callback that re-reads the persisted active version if another worker changed it."""
    global _refresh_hook
    _refresh_hook = hook


def pin_current() -> Token:
    """Pin the live configuration for the rest of this request/task; reset with :func:`unpin`."""
    hook = _refresh_hook
    if hook is not None:
        try:
            hook()
        except Exception:  # a broken store must never break request handling
            logging.getLogger(__name__).exception("AI version refresh failed; keeping current configuration")
    return _pinned.set(build_effective(_active))


def unpin(token: Token) -> None:
    _pinned.reset(token)


@contextmanager
def pinned(config: EffectiveConfig) -> Iterator[EffectiveConfig]:
    """Run a block under an explicit configuration (used to evaluate a CANDIDATE without activating it)."""
    token = _pinned.set(config.with_version_id(config.version_id, "pinned"))
    try:
        yield config
    finally:
        _pinned.reset(token)
