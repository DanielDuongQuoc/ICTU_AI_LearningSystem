"""Settings validation (fail fast), secrets never in repr, env/Docker/compose/CI configuration."""

from __future__ import annotations

import pytest
import re
import yaml
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from pathlib import Path


# ====================================================================================================
# settings validation
# ====================================================================================================


def test_default_settings_valid():
    settings = get_settings()
    assert settings.app_name
    assert settings.app_port == 8003
    assert settings.gemini_base_url.startswith("https://")
    assert settings.gemini_model
    assert settings.gemini_embedding_model
    assert settings.chroma_collection_name
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.top_k > 0
    assert 0.0 <= settings.similarity_threshold <= 1.0
    assert settings.max_upload_size_mb > 0


def test_get_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_max_upload_size_bytes():
    settings = get_settings()
    expected = settings.max_upload_size_mb * 1024 * 1024
    assert settings.max_upload_size_bytes == expected


def test_invalid_similarity_threshold():
    with pytest.raises(ValidationError):
        Settings(similarity_threshold=1.5)


def test_invalid_similarity_threshold_negative():
    with pytest.raises(ValidationError):
        Settings(similarity_threshold=-0.1)


def test_invalid_app_env():
    with pytest.raises(ValidationError):
        Settings(app_env="unknown_env")


def test_valid_app_envs():
    for env in ("development", "staging", "production", "test"):
        s = Settings(app_env=env)
        assert s.app_env == env


# ── configuration validation (fail fast) ─────────────────
def _kw(**overrides):
    return overrides


def test_llm_and_observability_defaults():
    s = Settings()
    assert s.llm_max_retries == 1
    assert s.llm_retry_backoff_s >= 0
    assert (s.llm_timeout, s.llm_connect_timeout, s.llm_health_timeout) == (120, 10, 5)
    assert (s.llm_temperature, s.llm_max_output_tokens) == (0.2, 1024)
    assert s.slow_request_threshold_ms > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_overlap": 10_000, "chunk_size": 500},
        {"chunk_overlap": 500, "chunk_size": 500},
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"top_k": 0},
        {"top_k": 50, "max_top_k": 20},
        {"max_top_k": 0},
        {"max_upload_size_mb": 0},
        {"max_context_chars": 0},
        {"gemini_base_url": "not-a-url"},
        {"gemini_base_url": "ftp://localhost:1"},
        {"gemini_model": ""},
        {"gemini_embedding_model": " "},
        {"llm_timeout": 0},
        {"llm_connect_timeout": 0},
        {"llm_health_timeout": -1},
        {"llm_max_retries": -1},
        {"llm_retry_backoff_s": -0.1},
        {"llm_max_output_tokens": 0},
        {"llm_temperature": 3},
        {"slow_request_threshold_ms": 0},
    ],
)
def test_invalid_configuration_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(**overrides)


def test_environment_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "7")
    monkeypatch.setenv("SLOW_REQUEST_THRESHOLD_MS", "1234")
    s = Settings()
    assert (s.chunk_size, s.chunk_overlap, s.max_upload_size_mb) == (800, 100, 7)
    assert s.slow_request_threshold_ms == 1234


def test_create_app_fails_fast_on_invalid_config(monkeypatch):
    from app.main import create_app

    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="APPLICATION_CONFIGURATION_ERROR"):
            create_app()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


# ====================================================================================================
# Docker / compose / env files
# ====================================================================================================


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _env_map(service: dict) -> dict[str, str]:
    env = service.get("environment", {})
    if isinstance(env, list):
        return dict(item.split("=", 1) for item in env)
    return {k: str(v) for k, v in env.items()}


def test_dockerfile_basics():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.(1[1-9]|[2-9]\d)", text, re.M)
    assert "requirements.txt" in text
    assert "EXPOSE 8003" in text
    assert "app.main:app" in text and "uvicorn" in text
    assert re.search(r"^USER\s+(?!root)\S+", text, re.M), "must not run as root"
    assert "/app/data/chroma" in text and "/app/uploads" in text
    assert "HEALTHCHECK" in text


def test_dockerfile_does_not_bake_in_secrets():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert not re.search(r"COPY[^\n]*\.env(\s|$)", text), ".env must not be copied"
    assert "GEMINI_API_KEY" not in text


def test_dockerignore_excludes_env_and_data():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".env", ".venv/", "data/", "uploads/"):
        assert entry in lines


def test_compose_has_required_services(compose):
    assert set(compose["services"]) == {"api"}          # Gemini is a hosted API: no model server


def test_compose_ports(compose):
    api_ports = [str(p) for p in compose["services"]["api"]["ports"]]
    assert any(p.endswith("8003:8003") for p in api_ports)


def test_compose_persistent_volumes(compose):
    volumes = set(compose["volumes"])
    assert {"chroma_data", "uploads_data"} <= volumes
    api_mounts = compose["services"]["api"]["volumes"]
    assert any(m.startswith("chroma_data:/app/data/chroma") for m in api_mounts)
    assert _env_map(compose["services"]["api"])["CHROMA_PERSIST_DIRECTORY"] == "/app/data/chroma"


def test_compose_important_environment_present(compose):
    env = _env_map(compose["services"]["api"])
    for key in ("APP_HOST", "APP_PORT", "CHROMA_PERSIST_DIRECTORY"):
        assert key in env


def test_compose_api_healthcheck_and_env_file(compose):
    api = compose["services"]["api"]
    assert "depends_on" not in api
    assert "healthcheck" in api
    assert any(e.get("path") == ".env" and e.get("required") is False for e in api["env_file"])
    # the API healthcheck must be liveness (never fails because a model is missing)
    assert "/health/live" in " ".join(api["healthcheck"]["test"])
    assert api["build"]["dockerfile"] == "Dockerfile"


def _env_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def test_env_example_covers_settings_and_is_valid():
    env = _env_example()
    for key in (
        "APP_NAME", "APP_ENV", "APP_HOST", "APP_PORT", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL", "LLM_TIMEOUT", "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS",
        "LLM_MAX_RETRIES", "LLM_RETRY_BACKOFF_S",
        "CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "TOP_K", "SIMILARITY_THRESHOLD", "MAX_CONTEXT_CHARS", "MAX_TOP_K", "MAX_UPLOAD_SIZE_MB",
        "SLOW_REQUEST_THRESHOLD_MS",
    ):
        assert key in env, key
    # the example file itself must be a valid configuration
    Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})


def test_env_example_has_no_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"(?i)(api[_-]?key|secret|password|token)[ \t]*=[ \t]*\S+", text)


def test_gitignore_protects_runtime_artifacts():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".venv/", "__pycache__/", ".pytest_cache/", "uploads/*", "data/chroma/*",
                  "benchmark_results.json", "logs/", "*.log"):
        assert entry in text, entry
    # source / tests / config must not be ignored
    for entry in ("tests/", "app/", "scripts/", "requirements.txt", "Dockerfile"):
        assert entry not in text.split()
