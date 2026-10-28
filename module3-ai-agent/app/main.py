import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.exceptions import ErrorCode
from app.core.logging import setup_logging
from app.core.openapi import install_openapi
from app.version import MODULE_VERSION
from app.core.middleware import RequestContextMiddleware
from app.repositories.chroma_repository import init_chroma
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.documents import router as documents_router
from app.api.questions import router as questions_router
from app.api.evaluations import router as evaluations_router
from app.api.rag import router as rag_router
from app.api.ai_versions import router as ai_versions_router


async def _close_shared_clients(logger: logging.Logger) -> None:
    """Close the shared httpx clients (LLM + embedding) if they were ever created."""
    from app.services import embedding_service, llm_service

    for name, service in (
        ("llm", llm_service._llm_service),
        ("embedding", embedding_service._embedding_service),
    ):
        if service is None:
            continue
        try:
            await service.close()
        except Exception as exc:  # shutdown must never raise
            logger.warning("Closing %s HTTP client failed: %s", name, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # already validated in create_app (fail fast)
    setup_logging(
        level=settings.log_level.strip().upper()
        or ("DEBUG" if settings.app_env == "development" else "INFO"),
        fmt=settings.log_format,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)

    # Dependencies are checked, but an unavailable one only degrades the app (see /health).
    try:
        init_chroma()
    except Exception as exc:
        logger.error("ChromaDB initialisation failed (starting degraded): %s", exc)

    try:
        from app.services.ai_version_service import init_ai_versions

        init_ai_versions()  # applies the persisted ACTIVE approved version (never raises)
    except Exception as exc:
        logger.error("AI version governance init failed (using environment configuration): %s", exc)

    from app.services.ollama_checker import check_ollama_health

    if not await check_ollama_health():
        logger.warning(
            "Ollama is not reachable at startup (starting degraded); "
            "LLM-backed endpoints will fail until it is available."
        )

    logger.info(
        "Module 3 ready at http://%s:%s | Docs: http://%s:%s/docs",
        settings.app_host, settings.app_port,
        settings.app_host, settings.app_port,
    )
    try:
        yield
    finally:
        await _close_shared_clients(logger)
        logger.info("Module 3 shutting down.")


def create_app() -> FastAPI:
    try:
        settings = get_settings()
    except ValidationError as exc:
        # Fail fast: refuse to start with an invalid configuration.
        logging.getLogger(__name__).critical(
            "%s: %s", ErrorCode.APPLICATION_CONFIGURATION_ERROR, exc
        )
        raise RuntimeError(
            f"{ErrorCode.APPLICATION_CONFIGURATION_ERROR}: invalid configuration, see log."
        ) from exc

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Module 3 - AI Agent for ICTU_AI_LearningSystem. "
            "Provides RAG-based document indexing, question generation, "
            "and student answer evaluation powered by Ollama/Qwen."
        ),
        version=MODULE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        # Added last so it is the outermost middleware and answers preflights first.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Accept", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(documents_router)
    app.include_router(rag_router)
    app.include_router(questions_router)
    app.include_router(evaluations_router)
    app.include_router(ai_versions_router)
    register_exception_handlers(app)
    install_openapi(app)
    return app


app = create_app()
