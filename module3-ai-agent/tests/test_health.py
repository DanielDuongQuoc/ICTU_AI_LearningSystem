import pytest


@pytest.mark.asyncio
async def test_health_all_ok(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama"] is True
    assert data["chromadb"] is True
    assert "version" in data


@pytest.mark.asyncio
async def test_health_ollama_down(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=False)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama"] is False
    assert data["chromadb"] is True


@pytest.mark.asyncio
async def test_health_chroma_down(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=False)
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama"] is True
    assert data["chromadb"] is False


@pytest.mark.asyncio
async def test_health_all_down(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=False)
    mocker.patch("app.api.health.check_chroma_health", return_value=False)
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama"] is False
    assert data["chromadb"] is False


@pytest.mark.asyncio
async def test_health_response_schema(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    response = await client.get("/health")
    data = response.json()
    required_fields = {"status", "ollama", "chromadb", "version"}
    assert required_fields.issubset(data.keys()), f"Missing: {required_fields - data.keys()}"


# ── Phase 9 — services block, liveness/readiness, robustness ───────


@pytest.mark.asyncio
async def test_health_services_block_ok(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    data = (await client.get("/health")).json()
    assert data["status"] == "ok"
    # Phase 20 added an additive "storage" key; the original four keys keep their meaning.
    assert {k: data["services"][k] for k in ("api", "chroma", "ollama", "model")} == {
        "api": "ok", "chroma": "ok", "ollama": "ok", "model": "ok"}
    assert data["services"]["storage"] == "ok"


@pytest.mark.asyncio
async def test_health_degraded_when_ollama_down(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=False)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["services"]["api"] == "ok"
    assert data["services"]["chroma"] == "ok"
    assert data["services"]["ollama"] == "error"
    assert data["services"]["model"] == "error"


@pytest.mark.asyncio
async def test_health_model_missing(client, mocker, monkeypatch):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)

    async def _no_model() -> bool:
        return False

    monkeypatch.setattr("app.api.health.check_model_health", _no_model)
    data = (await client.get("/health")).json()
    assert data["status"] == "degraded"
    assert data["services"]["ollama"] == "ok"
    assert data["services"]["model"] == "error"


@pytest.mark.asyncio
async def test_health_chroma_unavailable_services(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=False)
    data = (await client.get("/health")).json()
    assert data["status"] == "degraded"
    assert data["services"]["chroma"] == "error"


@pytest.mark.asyncio
async def test_health_does_not_crash_when_checks_raise(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", side_effect=RuntimeError("boom"))
    mocker.patch("app.api.health.check_chroma_health", side_effect=RuntimeError("boom"))
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["services"]["api"] == "ok"
    assert "boom" not in resp.text


@pytest.mark.asyncio
async def test_health_live(client):
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_ready_ok(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=True)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    resp = await client.get("/health/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_ready_503_when_degraded(client, mocker):
    mocker.patch("app.api.health.check_ollama_health", return_value=False)
    mocker.patch("app.api.health.check_chroma_health", return_value=True)
    resp = await client.get("/health/ready")
    assert resp.status_code == 503
