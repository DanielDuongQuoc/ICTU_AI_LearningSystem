"""Effective AI configuration: the quality-relevant parameters services read at runtime.

* :func:`current` is the single place services read the quality-relevant parameters from
  (prompt versions, top_k, similarity threshold, context limit, LLM model/temperature/max tokens).
* The values come from the environment settings only; there is no runtime override.
* :attr:`EffectiveConfig.fingerprint` identifies the configuration (including prompt TEXT) and is reported
  in the ``X-AI-Config-Fingerprint`` response header and in execution-trace logs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.core import prompt_registry as registry
from app.core.config import get_settings

# Reported as X-AI-Config-Version (value kept stable for existing log/header consumers).
CONFIG_VERSION_ID = "env-bootstrap"


@dataclass(frozen=True)
class EffectiveConfig:
    version_id: str
    source: str  # always "env"
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
        """Secret-free description used for fingerprints and logs."""
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


def current() -> EffectiveConfig:
    """The configuration derived from the current settings."""
    s = get_settings()
    return EffectiveConfig(
        version_id=CONFIG_VERSION_ID, source="env", llm_provider="gemini",
        llm_model=s.gemini_model, temperature=s.llm_temperature, num_predict=s.llm_max_output_tokens,
        embedding_model=s.gemini_embedding_model, top_k=s.top_k,
        similarity_threshold=s.similarity_threshold, max_context_chars=s.max_context_chars,
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
        prompt_versions=tuple(sorted(registry.DEFAULT_PROMPT_VERSIONS.items())),
    )
