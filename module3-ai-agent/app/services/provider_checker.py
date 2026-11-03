"""Health probes for the LLM provider (Google Gemini API). Never raise; never consume tokens."""

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def _get(path: str) -> bool:
    s = get_settings()
    key = s.gemini_api_key.strip()
    if not key:
        logger.warning("GEMINI_API_KEY is not configured; the LLM provider is unavailable.")
        return False
    url = f"{s.gemini_base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=s.llm_health_timeout) as client:
            response = await client.get(url, headers={"x-goog-api-key": key})
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("Gemini API unreachable: %s", type(exc).__name__)
        return False
    except Exception as exc:
        logger.error("Unexpected error during Gemini health check: %s", type(exc).__name__)
        return False
    if response.status_code == 200:
        return True
    logger.warning("Gemini API health probe %s returned status %d", path.split("?")[0], response.status_code)
    return False


async def check_provider_health() -> bool:
    """True when the Gemini API answers with the configured key (lists models, no generation)."""
    return await _get("/models?pageSize=1")


async def check_model_health() -> bool:
    """True when the configured generation model (``GEMINI_MODEL``) exists for this key."""
    model = get_settings().gemini_model.removeprefix("models/")
    return await _get(f"/models/{model}")
