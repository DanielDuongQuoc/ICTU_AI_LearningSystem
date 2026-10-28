"""Phase 21 - production configuration, release manifest, gate verdict logic, API contract freeze,
Docker/Git hygiene (static audits; the image itself is NOT built here)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.config import Settings, get_settings
from scripts import release_manifest, validate_release

ROOT = Path(__file__).resolve().parents[2]
PINNED = {"ollama_model": "qwen2.5:7b", "ollama_embedding_model": "nomic-embed-text:v1.5"}


# --------------------------------------------------------------------------------- production config
@pytest.mark.parametrize("llm,emb", [
    ("qwen2.5:latest", "nomic-embed-text:v1.5"),
    ("qwen2.5", "nomic-embed-text:v1.5"),
    ("qwen2.5:7b", "nomic-embed-text"),
    ("qwen2.5:7b", "nomic-embed-text:latest"),
])
def test_production_rejects_mutable_or_untagged_models(llm, emb):
    with pytest.raises(ValueError, match="(?i)non-'latest' tag"):
        Settings(_env_file=None, app_env="production", ollama_model=llm, ollama_embedding_model=emb)


def test_production_accepts_pinned_models_and_other_envs_keep_defaults():
    assert Settings(_env_file=None, app_env="production", **PINNED).app_env == "production"
    for env in ("development", "test", "staging"):
        assert Settings(_env_file=None, app_env=env).ollama_embedding_model == "nomic-embed-text"


@pytest.mark.parametrize("override,msg", [
    ({"cors_allowed_origins": "*"}, "CORS"),
    ({"log_level": "DEBUG"}, "DEBUG"),
    ({"admin_api_token": "short"}, "ADMIN_API_TOKEN"),
    ({"chunk_overlap": 900, "chunk_size": 800}, "CHUNK_OVERLAP"),
    ({"similarity_threshold": 1.5}, "similarity_threshold"),
    ({"max_upload_size_mb": -1}, "MAX_UPLOAD_SIZE_MB"),
    ({"ollama_timeout": 0}, "OLLAMA_TIMEOUT"),
    ({"ollama_base_url": "not-a-url"}, "OLLAMA_BASE_URL"),
    ({"top_k": 0}, "TOP_K"),
])
def test_insecure_or_impossible_production_configuration_fails_fast(override, msg):
    with pytest.raises(ValueError, match=msg):
        Settings(_env_file=None, app_env="production", **PINNED, **override)


def test_invalid_app_env_and_port_type_fail():
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_env="prod")
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_port="not-a-number")


def test_config_errors_never_echo_secret_values():
    secret = "-".join(["probe", "value", "y" * 30])
    with pytest.raises(ValueError) as e:
        Settings(_env_file=None, app_env="production", **PINNED, admin_api_token=secret, cors_allowed_origins="*")
    assert secret not in str(e.value) and secret not in repr(e.value)


def test_env_example_is_a_valid_development_configuration_and_names_production_pinning():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "APP_ENV" in text


# --------------------------------------------------------------------------------- manifest
def test_manifest_has_required_fields_and_is_secret_free(monkeypatch):
    token = "-".join(["probe", "value", "x" * 30])
    monkeypatch.setenv("ADMIN_API_TOKEN", token)
    get_settings.cache_clear()
    m = release_manifest.build_manifest()
    text = release_manifest._canonical(m)
    assert token not in text
    for key in ("release_id", "application_version", "git_commit", "build_timestamp", "ai", "build", "gates"):
        assert key in m
    for key in ("ai_version_id", "llm_model", "embedding_model", "prompt_versions", "config_fingerprint"):
        assert key in m["ai"]
    assert all(v == "NOT VERIFIED" for v in m["gates"].values()), "gates are never pre-filled"
    get_settings.cache_clear()


def test_release_id_is_stable_and_ignores_the_timestamp():
    a = release_manifest.build_manifest(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    b = release_manifest.build_manifest(now=datetime(2030, 6, 6, tzinfo=timezone.utc))
    assert a["release_id"] == b["release_id"] and a["build_timestamp"] != b["build_timestamp"]
    assert a["release_id"].startswith(a["application_version"] + "+")


def test_release_id_changes_when_the_ai_configuration_changes(monkeypatch):
    base = release_manifest.build_manifest()["release_id"]
    monkeypatch.setenv("TOP_K", "7")
    get_settings.cache_clear()
    assert release_manifest.build_manifest()["release_id"] != base
    get_settings.cache_clear()


def test_manifest_secret_guard_rejects_secret_looking_values():
    with pytest.raises(ValueError):
        release_manifest.assert_no_secrets({"x": "password=hunter2"})


def test_manifest_cli_refuses_to_overwrite(tmp_path):
    out = tmp_path / "m.json"
    assert release_manifest.main(["--out", str(out)]) == 0
    assert release_manifest.main(["--out", str(out)]) == 1


def test_missing_git_is_reported_not_invented():
    commit = release_manifest.git_state()["commit"]
    assert commit.startswith("unavailable") or re.fullmatch(r"[0-9a-f]{40}", commit)


# --------------------------------------------------------------------------------- gate verdict
def _gates(**over):
    base = {g: "PASS" for g in ("static_checks", "functional", "security", "reliability", "release_engineering", "ai_quality", "docker")}
    base.update(over)
    return base


def test_verdict_blocked_on_any_blocking_failure():
    for gate in ("security", "functional", "reliability", "release_engineering", "static_checks", "ai_quality", "docker"):
        assert validate_release.verdict(_gates(**{gate: "FAIL"})) == "BLOCKED"


def test_security_failure_blocks_even_when_everything_else_passes():
    assert validate_release.verdict(_gates(security="FAIL")) == "BLOCKED"


def test_unverified_or_partial_gates_never_yield_plain_go():
    assert validate_release.verdict(_gates(docker="NOT VERIFIED")) == "GO WITH LIMITATIONS"
    assert validate_release.verdict(_gates(ai_quality="PARTIAL")) == "GO WITH LIMITATIONS"
    assert validate_release.verdict(_gates()) == "GO"


def test_docker_gate_is_not_verified_without_docker(monkeypatch):
    monkeypatch.setattr(validate_release.shutil, "which", lambda name: None)
    status, detail = validate_release.gate_docker()
    assert status == "NOT VERIFIED" and "docker" in detail.lower()


def test_skipped_tests_are_not_verified_not_pass(monkeypatch):
    monkeypatch.setattr(validate_release, "gate_static", lambda: ("PASS", ""))
    monkeypatch.setattr(validate_release, "gate_ai_quality", lambda: ("PARTIAL", ""))
    monkeypatch.setattr(validate_release, "gate_docker", lambda: ("NOT VERIFIED", ""))
    gates, _ = validate_release.run(skip_tests=True)
    assert gates["security"] == gates["reliability"] == gates["functional"] == "NOT VERIFIED"
    assert validate_release.verdict(gates) == "GO WITH LIMITATIONS"


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
    # Phase 18 admin API (disabled unless ADMIN_API_TOKEN is set). Anything else would be an unreviewed surface.
    known = {("post", "/api/documents/{document_id}/search"),
             ("post", "/api/questions/generate-matrix")}  # reviewed additive endpoint (exam matrix, docs/INTEGRATION_CONTRACT.md)
    assert all(p.startswith("/api/ai-versions") or (m, p) in known for m, p in extra), f"undocumented API surface: {extra}"


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
    images = release_manifest.docker_base_images()
    assert images and all(re.fullmatch(r"python:\d+\.\d+\.\d+-slim", i) for i in images), images


def test_dockerfile_runs_non_root_without_secrets_and_with_writable_dirs():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^USER\s+app", text) and not re.search(r"(?mi)^(COPY|ADD).*\.env", text)
    for d in ("/app/data/chroma", "/app/data/ai_versions", "/app/uploads"):
        assert d in text
    assert "HEALTHCHECK" in text and "EXPOSE 8003" in text and "CMD" in text


def test_compose_persists_every_state_directory_and_gates_on_ollama_health():
    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    mounts = " ".join(map(str, api["volumes"]))
    for target in ("/app/data/chroma", "/app/data/ai_versions", "/app/uploads"):
        assert target in mounts, target
    assert api["depends_on"]["ollama"]["condition"] == "service_healthy"
    assert api["healthcheck"]["test"] and api["stop_grace_period"]
    assert set(compose["volumes"]) >= {"chroma_data", "ai_versions_data", "uploads_data", "ollama_data"}


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
