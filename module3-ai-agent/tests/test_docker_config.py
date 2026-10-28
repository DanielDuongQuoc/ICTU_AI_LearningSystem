"""Phase 10 - static checks of Dockerfile / docker-compose.yml / .env.example / .gitignore.

Pure file inspection: no Docker daemon required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app.core.config import Settings

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


def test_dockerfile_does_not_bake_in_secrets_or_ollama_url():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert not re.search(r"COPY[^\n]*\.env(\s|$)", text), ".env must not be copied"
    assert "localhost:11434" not in text


def test_dockerignore_excludes_env_and_data():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".env", ".venv/", "data/", "uploads/"):
        assert entry in lines


def test_compose_has_required_services(compose):
    assert {"api", "ollama"} <= set(compose["services"])


def test_compose_ports(compose):
    api_ports = [str(p) for p in compose["services"]["api"]["ports"]]
    ollama_ports = [str(p) for p in compose["services"]["ollama"]["ports"]]
    assert any(p.endswith("8003:8003") for p in api_ports)
    assert any(p.endswith("11434:11434") for p in ollama_ports)
    # the Ollama port must not be exposed on all interfaces
    assert all(p.startswith("127.0.0.1:") for p in ollama_ports)


def test_compose_ollama_url_uses_service_name_not_localhost(compose):
    env = _env_map(compose["services"]["api"])
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert "localhost" not in env["OLLAMA_BASE_URL"] and "127.0.0.1" not in env["OLLAMA_BASE_URL"]


def test_compose_persistent_volumes(compose):
    volumes = set(compose["volumes"])
    assert {"ollama_data", "chroma_data"} <= volumes
    api_mounts = compose["services"]["api"]["volumes"]
    assert any(m.startswith("chroma_data:/app/data/chroma") for m in api_mounts)
    assert any(m.startswith("ollama_data:/root/.ollama") for m in compose["services"]["ollama"]["volumes"])
    assert _env_map(compose["services"]["api"])["CHROMA_PERSIST_DIRECTORY"] == "/app/data/chroma"


def test_compose_important_environment_present(compose):
    env = _env_map(compose["services"]["api"])
    for key in ("APP_HOST", "APP_PORT", "OLLAMA_BASE_URL", "CHROMA_PERSIST_DIRECTORY"):
        assert key in env


def test_compose_api_waits_for_ollama_and_healthchecks(compose):
    api = compose["services"]["api"]
    assert "ollama" in api["depends_on"]
    assert "healthcheck" in api and "healthcheck" in compose["services"]["ollama"]
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
        "APP_NAME", "APP_ENV", "APP_HOST", "APP_PORT", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
        "OLLAMA_EMBEDDING_MODEL", "OLLAMA_TIMEOUT", "OLLAMA_TEMPERATURE", "OLLAMA_NUM_PREDICT",
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

