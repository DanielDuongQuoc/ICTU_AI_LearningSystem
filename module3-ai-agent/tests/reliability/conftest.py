import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture()
async def soft_client():
    """Like production: an unexpected exception becomes the catch-all 500 response instead of being re-raised
    into the test (httpx re-raises app exceptions by default after the response was already produced)."""
    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False),
                           base_url="http://testserver") as ac:
        yield ac
