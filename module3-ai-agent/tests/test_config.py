import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_default_settings_valid():
    settings = get_settings()
    assert settings.app_name
    assert settings.app_port == 8003
    assert settings.ollama_base_url.startswith("http")
    assert settings.ollama_model
    assert settings.ollama_embedding_model
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
        s = Settings(app_env=env, ollama_model="qwen2.5:7b", ollama_embedding_model="nomic-embed-text:v1.5")  # pinned tags (Phase 21)
        assert s.app_env == env


# ── Phase 9 — configuration validation (fail fast) ─────────────────


def _kw(**overrides):
    return overrides


def test_phase9_defaults():
    s = Settings()
    assert s.ollama_max_retries == 1
    assert s.ollama_retry_backoff_s >= 0
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
        {"ollama_base_url": "not-a-url"},
        {"ollama_base_url": "ftp://localhost:11434"},
        {"ollama_model": ""},
        {"ollama_timeout": 0},
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
