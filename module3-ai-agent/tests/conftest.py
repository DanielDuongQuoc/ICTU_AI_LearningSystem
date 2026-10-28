import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest_asyncio.fixture()
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def _no_real_model_check(monkeypatch):
    """Health tests must never depend on a locally running Ollama (tests may override)."""

    async def _model_ok() -> bool:
        return True

    monkeypatch.setattr("app.api.health.check_model_health", _model_ok)


@pytest.fixture(autouse=True)
def _fresh_abuse_limits():
    """Phase 19: the in-process rate limiter is global state; every test starts with an empty budget."""
    from app.core.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()
