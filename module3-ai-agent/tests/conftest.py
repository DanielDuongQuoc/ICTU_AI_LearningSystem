import os

# Tests never use the real Gemini key from .env: environment variables take precedence over the .env file.
# (Only the opt-in integration tests read the real key, see tests/test_integration.py.)
ORIGINAL_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = "test-gemini-key-not-real"

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest_asyncio.fixture()
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def _no_real_network(request, monkeypatch):
    """No unit test may reach the real network (Gemini included). ``-m integration`` tests are exempt.

    ASGITransport (in-process app) and httpx.MockTransport do not go through these transports.
    """
    if request.node.get_closest_marker("integration"):
        return

    async def _blocked_async(self, req):
        raise RuntimeError(f"real network access attempted in a unit test: {req.method} {req.url.host}")

    def _blocked_sync(self, req):
        raise RuntimeError(f"real network access attempted in a unit test: {req.method} {req.url.host}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked_sync)


@pytest.fixture(autouse=True)
def _provider_up(monkeypatch):
    """Health checks report the Gemini provider and model as available unless a test overrides it."""

    async def _ok() -> bool:
        return True

    monkeypatch.setattr("app.api.health.check_provider_health", _ok)
    monkeypatch.setattr("app.api.health.check_model_health", _ok)


@pytest.fixture(autouse=True)
def _fresh_abuse_limits():
    """The in-process rate limiter is global state; every test starts with an empty budget."""
    from app.core.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()
