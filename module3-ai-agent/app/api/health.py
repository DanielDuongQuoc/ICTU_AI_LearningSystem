import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.repositories.chroma_repository import check_chroma_health, check_embedding_compat
from app.services.ollama_checker import check_model_health, check_ollama_health
from app.schemas.common import HealthCheckResponse
from app.core.openapi import error_responses

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)

from app.version import MODULE_VERSION as API_VERSION


async def _safe(check, *, is_async: bool) -> bool:
    """Run a dependency check; a failing check is reported, never raised."""
    try:
        result = await check() if is_async else await asyncio.to_thread(check)
        return bool(result)
    except Exception as exc:
        logger.error("Health check %s raised: %s", getattr(check, "__name__", "check"), exc)
        return False


def check_storage_health() -> bool:
    """Phase 20: the data directory has at least MIN_FREE_DISK_MB free (cheap, no generation, no writes)."""
    import shutil
    from pathlib import Path

    from app.core.config import get_settings

    s = get_settings()
    path = Path(s.chroma_persist_directory).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    free_mb = shutil.disk_usage(path).free / (1024 * 1024)
    if free_mb < s.min_free_disk_mb:
        logger.warning("Low disk space: free_mb=%.0f min_free_disk_mb=%d", free_mb, s.min_free_disk_mb)
        return False
    return True


async def _collect() -> HealthCheckResponse:
    chroma_ok = await _safe(check_chroma_health, is_async=False)
    ollama_ok = await _safe(check_ollama_health, is_async=True)
    # The model can only be available if Ollama is reachable.
    model_ok = ollama_ok and await _safe(check_model_health, is_async=True)

    storage_ok = await _safe(check_storage_health, is_async=False)
    embedding_index_ok = await _safe(check_embedding_compat, is_async=False)

    def label(ok: bool) -> str:
        return "ok" if ok else "error"

    overall = "ok" if (ollama_ok and chroma_ok and model_ok and storage_ok and embedding_index_ok) else "degraded"
    if overall == "degraded":
        logger.warning(
            "Health degraded: ollama=%s model=%s chromadb=%s storage=%s", ollama_ok, model_ok, chroma_ok, storage_ok
        )
    return HealthCheckResponse(
        status=overall,
        ollama=ollama_ok,
        chromadb=chroma_ok,
        version=API_VERSION,
        services={
            "api": "ok",
            "chroma": label(chroma_ok),
            "ollama": label(ollama_ok),
            "model": label(model_ok),
            "storage": label(storage_ok),
            "embedding_index": label(embedding_index_ok),
        },
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    responses=error_responses(500),
    description=(
        "Returns operational status of Module 3 and its dependencies (Chroma, Ollama, "
        "Qwen model). Always returns HTTP 200; check the status field for degradation."
    ),
)
async def health_check() -> JSONResponse:
    payload = await _collect()
    return JSONResponse(content=payload.model_dump(), status_code=200)


@router.get(
    "/health/live",
    summary="Liveness probe (process is running)",
    description="Always 200 while the process is up; checks no dependency.",
)
async def liveness() -> dict:
    return {"status": "ok"}


@router.get(
    "/health/ready",
    summary="Readiness probe (dependencies available)",
    description="200 when every dependency is available, 503 otherwise (body is the same as /health).",
    responses={
        200: {"model": HealthCheckResponse, "description": "All dependencies available."},
        503: {"model": HealthCheckResponse, "description": "At least one dependency is unavailable."},
    },
)
@router.get(
    "/ready",
    summary="Readiness probe (alias of /health/ready)",
    description="Same as GET /health/ready: 200 when every dependency is available, 503 otherwise.",
    responses={
        200: {"model": HealthCheckResponse, "description": "All dependencies available."},
        503: {"model": HealthCheckResponse, "description": "At least one dependency is unavailable."},
    },
)
async def readiness() -> JSONResponse:
    payload = await _collect()
    return JSONResponse(
        content=payload.model_dump(), status_code=200 if payload.status == "ok" else 503
    )
