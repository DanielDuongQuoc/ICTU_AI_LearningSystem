import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_ollama_health_ok():
    from app.services.ollama_checker import check_ollama_health
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("app.services.ollama_checker.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        result = await check_ollama_health()
        assert result is True


@pytest.mark.asyncio
async def test_ollama_health_connect_error():
    from app.services.ollama_checker import check_ollama_health
    with patch("app.services.ollama_checker.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        result = await check_ollama_health()
        assert result is False


@pytest.mark.asyncio
async def test_ollama_health_timeout():
    from app.services.ollama_checker import check_ollama_health
    with patch("app.services.ollama_checker.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        result = await check_ollama_health()
        assert result is False


@pytest.mark.asyncio
async def test_ollama_health_non_200():
    from app.services.ollama_checker import check_ollama_health
    mock_response = MagicMock()
    mock_response.status_code = 503
    with patch("app.services.ollama_checker.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        result = await check_ollama_health()
        assert result is False
