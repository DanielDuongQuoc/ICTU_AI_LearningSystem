import logging
from typing import Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def check_ollama_health() -> bool:
    settings = get_settings()
    if settings.gemini_api_key.strip():
        url = f"{settings.gemini_base_url.rstrip('/')}/models"
        headers = {"x-goog-api-key": settings.gemini_api_key.strip()}
    else:
        url = f"{settings.ollama_base_url}/api/tags"
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_health_timeout) as client:
            response = await client.get(url, headers=headers) if headers else await client.get(url)
            if response.status_code == 200:
                logger.debug("LLM provider health check passed.")
                return True
            logger.warning("LLM provider returned status %d", response.status_code)
            return False
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("LLM provider unreachable: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error during LLM provider health check: %s", exc)
        return False


async def get_available_models() -> Optional[list]:
    settings = get_settings()
    if settings.gemini_api_key.strip():
        url = f"{settings.gemini_base_url.rstrip('/')}/models"
        headers = {"x-goog-api-key": settings.gemini_api_key.strip()}
    else:
        url = f"{settings.ollama_base_url}/api/tags"
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_health_timeout) as client:
            response = await client.get(url, headers=headers) if headers else await client.get(url)
            response.raise_for_status()
            data = response.json()
            names: list[str] = []
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    names.append(name)
                    if name.startswith("models/"):
                        names.append(name.removeprefix("models/"))
            return names
    except Exception as exc:
        logger.warning("Could not fetch model list: %s", exc)
        return None


async def check_model_health() -> bool:
    """True if the configured generation model is available in the active provider."""
    s = get_settings()
    model = s.gemini_model.removeprefix("models/") if s.gemini_api_key.strip() else s.ollama_model
    names = await get_available_models()
    if names is None:
        return False
    # "name" without a tag resolves to "name:latest" in Ollama.
    return model in names or (":" not in model and f"{model}:latest" in names)
