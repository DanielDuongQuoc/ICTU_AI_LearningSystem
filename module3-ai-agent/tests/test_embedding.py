"""Phase 3 — Embedding service tests.

All tests MOCK Ollama.  No real Ollama required for unit tests.
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.core.exceptions import AppError, ErrorCode
from app.services.embedding_service import EmbeddingService


# ── Helpers ─────────────────────────────────────────────────────────

FAKE_EMBEDDING = [0.1, -0.2, 0.3, 0.4, 0.5]
FAKE_DIM = len(FAKE_EMBEDDING)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> httpx.Response:
    """Build a fake httpx.Response."""
    request = httpx.Request("POST", "http://fake/api/embed")
    if json_data is not None:
        return httpx.Response(status_code=status_code, json=json_data, request=request)
    return httpx.Response(status_code=status_code, text=text, request=request)


def _make_service() -> EmbeddingService:
    return EmbeddingService(
        base_url="http://fake-ollama:11434",
        model="test-embed-model",
        timeout=10,
    )


# ── Tests ───────────────────────────────────────────────────────────


class TestEmbedText:
    """embed_text — single text embedding."""

    @pytest.mark.asyncio
    async def test_valid_text(self) -> None:
        service = _make_service()
        mock_resp = _mock_response(json_data={"embeddings": [FAKE_EMBEDDING]})

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            result = await service.embed_text("Hello world")

        assert result == FAKE_EMBEDDING
        assert len(result) == FAKE_DIM

    @pytest.mark.asyncio
    async def test_empty_text_raises(self) -> None:
        service = _make_service()
        with pytest.raises(AppError) as exc_info:
            await service.embed_text("")
        assert exc_info.value.code == ErrorCode.EMBEDDING_EMPTY_TEXT

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self) -> None:
        service = _make_service()
        with pytest.raises(AppError) as exc_info:
            await service.embed_text("   \n\t  ")
        assert exc_info.value.code == ErrorCode.EMBEDDING_EMPTY_TEXT


class TestEmbedDocuments:
    """embed_documents — batch embedding."""

    @pytest.mark.asyncio
    async def test_multiple_texts(self) -> None:
        service = _make_service()
        emb_a = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb_b = [0.5, 0.4, 0.3, 0.2, 0.1]
        mock_resp = _mock_response(json_data={"embeddings": [emb_a, emb_b]})

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            result = await service.embed_documents(["text A", "text B"])

        assert len(result) == 2
        assert result[0] == emb_a
        assert result[1] == emb_b

    @pytest.mark.asyncio
    async def test_consistent_dimension(self) -> None:
        service = _make_service()
        embeddings = [[float(i)] * 768 for i in range(3)]
        mock_resp = _mock_response(json_data={"embeddings": embeddings})

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            result = await service.embed_documents(["a", "b", "c"])

        dims = {len(e) for e in result}
        assert len(dims) == 1
        assert 768 in dims

    @pytest.mark.asyncio
    async def test_empty_list_raises(self) -> None:
        service = _make_service()
        with pytest.raises(AppError) as exc_info:
            await service.embed_documents([])
        assert exc_info.value.code == ErrorCode.EMBEDDING_EMPTY_TEXT

    @pytest.mark.asyncio
    async def test_empty_string_in_batch_raises(self) -> None:
        service = _make_service()
        with pytest.raises(AppError) as exc_info:
            await service.embed_documents(["valid", ""])
        assert exc_info.value.code == ErrorCode.EMBEDDING_EMPTY_TEXT


class TestOllamaErrors:
    """Error handling for Ollama failures."""

    @pytest.mark.asyncio
    async def test_ollama_unavailable(self) -> None:
        service = _make_service()
        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        service = _make_service()
        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_TIMEOUT

    @pytest.mark.asyncio
    async def test_model_not_found(self) -> None:
        service = _make_service()
        mock_resp = _mock_response(status_code=404, text="model not found")

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_MODEL_NOT_FOUND

    @pytest.mark.asyncio
    async def test_invalid_json_response(self) -> None:
        service = _make_service()
        # Response with no valid JSON (override json to raise)
        resp = httpx.Response(
            status_code=200,
            text="not json at all",
            request=httpx.Request("POST", "http://fake/api/embed"),
        )

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=resp)
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE

    @pytest.mark.asyncio
    async def test_missing_embeddings_key(self) -> None:
        service = _make_service()
        mock_resp = _mock_response(json_data={"result": "ok"})

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE

    @pytest.mark.asyncio
    async def test_mismatched_count(self) -> None:
        service = _make_service()
        # Request 2 texts but get 1 embedding
        mock_resp = _mock_response(json_data={"embeddings": [FAKE_EMBEDDING]})

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_documents(["a", "b"])
            assert exc_info.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE

    @pytest.mark.asyncio
    async def test_server_error_500(self) -> None:
        service = _make_service()
        mock_resp = _mock_response(status_code=500, text="internal error")

        with patch.object(service, "_get_client") as mock_client:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            with pytest.raises(AppError) as exc_info:
                await service.embed_text("hello")
            assert exc_info.value.code == ErrorCode.EMBEDDING_INVALID_RESPONSE


class TestModelName:
    """Model name property."""

    def test_model_name(self) -> None:
        service = _make_service()
        assert service.model_name == "test-embed-model"
