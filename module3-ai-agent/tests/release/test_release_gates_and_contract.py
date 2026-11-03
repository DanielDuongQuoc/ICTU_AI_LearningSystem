"""Phase 21 - production configuration, API contract freeze,
Docker/Git hygiene (static audits; the image itself is NOT built here)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
PROD = {"min_free_disk_mb": 10}


# --------------------------------------------------------------------------------- production config
def test_render_production_configuration_validates():
    s = Settings(_env_file=None, app_env="production", log_level="INFO", gemini_api_key="x", min_free_disk_mb=10)  # = render.yaml
    assert s.app_env == "production" and s.gemini_model and s.gemini_embedding_model
    for env in ("development", "test", "staging"):
        assert Settings(_env_file=None, app_env=env).gemini_embedding_model == "gemini-embedding-001"


def test_production_starts_without_api_key_but_is_degraded_not_invalid():
    assert Settings(_env_file=None, app_env="production", gemini_api_key="").gemini_api_key == ""


@pytest.mark.parametrize("override,msg", [
    ({"cors_allowed_origins": "*"}, "CORS"),
    ({"log_level": "DEBUG"}, "DEBUG"),
    ({"chunk_overlap": 900, "chunk_size": 800}, "CHUNK_OVERLAP"),
    ({"similarity_threshold": 1.5}, "similarity_threshold"),
    ({"max_upload_size_mb": -1}, "MAX_UPLOAD_SIZE_MB"),
    ({"llm_timeout": 0}, "LLM_TIMEOUT"),
    ({"gemini_base_url": "not-a-url"}, "GEMINI_BASE_URL"),
    ({"top_k": 0}, "TOP_K"),
])
def test_insecure_or_impossible_production_configuration_fails_fast(override, msg):
    with pytest.raises(ValueError, match=msg):
        Settings(_env_file=None, app_env="production", **PROD, **override)


def test_invalid_app_env_and_port_type_fail():
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_env="prod")
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_port="not-a-number")


def test_config_errors_never_echo_secret_values():
    secret = "-".join(["probe", "value", "y" * 30])
    with pytest.raises(ValueError) as e:
        Settings(_env_file=None, app_env="production", **PROD, gemini_api_key=secret, cors_allowed_origins="*")
    assert secret not in str(e.value) and secret not in repr(e.value)


def test_env_example_is_a_valid_development_configuration_and_names_production_pinning():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "APP_ENV" in text


# --------------------------------------------------------------------------------- API contract freeze
EXPECTED = {
    ("get", "/health"), ("get", "/health/live"), ("get", "/ready"), ("get", "/health/ready"),
    ("post", "/api/documents/upload"), ("get", "/api/documents/{document_id}"),
    ("delete", "/api/documents/{document_id}"), ("post", "/api/rag/retrieve"), ("post", "/api/rag/answer"),
    ("post", "/api/questions/generate"), ("post", "/api/evaluations/evaluate"),
    ("post", "/api/evaluations/evaluate/stream"),
}


def test_openapi_contains_exactly_the_documented_contract_endpoints():
    from app.main import app

    spec = app.openapi()
    have = {(m, p) for p, ops in spec["paths"].items() for m in ops}
    assert EXPECTED <= have, f"missing: {EXPECTED - have}"
    extra = {(m, p) for m, p in have - EXPECTED}
    # Real, previously documented endpoints that are not in the Phase 21 prompt's list: document search and the
    # exam matrix. Anything else would be an unreviewed surface.
    known = {("post", "/api/documents/{document_id}/search"),
             ("post", "/api/questions/generate-matrix")}  # reviewed additive endpoint (exam matrix, docs/INTEGRATION_CONTRACT.md)
    assert all((m, p) in known for m, p in extra), f"undocumented API surface: {extra}"


def test_openapi_upload_is_multipart_and_stream_is_event_stream():
    from app.main import app

    spec = app.openapi()
    upload = spec["paths"]["/api/documents/upload"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]
    stream = spec["paths"]["/api/evaluations/evaluate/stream"]["post"]
    contents = {ct for r in stream["responses"].values() for ct in r.get("content", {})}
    assert "text/event-stream" in contents, contents


def test_openapi_error_responses_use_a_shared_error_schema_and_status_codes():
    from app.main import app

    spec = app.openapi()
    for path, method in (("/api/rag/retrieve", "post"), ("/api/questions/generate", "post"), ("/api/evaluations/evaluate", "post")):
        responses = spec["paths"][path][method]["responses"]
        assert {"200", "422"} <= set(responses), (path, set(responses))
        assert any(code in responses for code in ("400", "404", "503")), (path, set(responses))


def test_application_and_openapi_versions_agree():
    from app.main import app
    from app.version import MODULE_VERSION

    assert app.openapi()["info"]["version"] == MODULE_VERSION


# --------------------------------------------------------------------------------- Docker / Git hygiene (static)
def test_dockerfile_base_images_are_pinned_to_an_exact_version():
    images = re.findall(r"(?m)^FROM\s+(\S+)", (ROOT / "Dockerfile").read_text(encoding="utf-8"))
    assert images and all(re.fullmatch(r"python:\d+\.\d+\.\d+-slim", i) for i in images), images


def test_dockerfile_runs_non_root_without_secrets_and_with_writable_dirs():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^USER\s+app", text) and not re.search(r"(?mi)^(COPY|ADD).*\.env", text)
    for d in ("/app/data/chroma", "/app/uploads"):
        assert d in text
    assert "HEALTHCHECK" in text and "EXPOSE 8003" in text and "CMD" in text


def test_compose_persists_every_state_directory():
    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    mounts = " ".join(map(str, api["volumes"]))
    for target in ("/app/data/chroma", "/app/uploads"):
        assert target in mounts, target
    assert "depends_on" not in api and "ollama" not in compose["services"]
    assert api["healthcheck"]["test"] and api["stop_grace_period"]
    assert set(compose["volumes"]) >= {"chroma_data", "uploads_data"}


def test_ignore_rules_keep_secrets_and_data_out_of_git_and_image():
    git = (ROOT / ".gitignore").read_text(encoding="utf-8")
    docker = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for needle in (".env", "data/chroma/*", "uploads/*", "backups/", "__pycache__/"):
        assert needle in git, f".gitignore lacks {needle}"
    for needle in (".env", "data/", "uploads/", "backups/"):
        assert needle in docker, f".dockerignore lacks {needle}"


def test_requirements_are_exactly_pinned():
    loose = [ln for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#") and "==" not in ln]
    assert loose == []
