"""Shared OpenAPI response documentation.

Every error body has the same shape (``ErrorResponse``); this helper lets each route declare which
HTTP statuses it can return without repeating the descriptions.
"""

from __future__ import annotations

from typing import Any

from app.schemas.common import ErrorResponse

ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "Invalid input (domain validation code such as EMPTY_QUERY, INVALID_TOP_K, INVALID_PDF).",
    404: "Document not found (DOCUMENT_NOT_FOUND).",
    401: "Admin token missing or wrong (ADMIN_AUTH_REQUIRED).",
    409: "Version-state conflict or refusal (AI_VERSION_INVALID_TRANSITION, AI_VERSION_GATE_BLOCKED, AI_VERSION_CONFLICT, ...).",
    413: "File or document too large (FILE_TOO_LARGE, DOCUMENT_LIMIT_EXCEEDED: pages/text/chunks).",
    429: "Rate limit exceeded (RATE_LIMITED); see the Retry-After header.",
    422: "Invalid request data (VALIDATION_ERROR) or not enough document context (INSUFFICIENT_CONTEXT).",
    500: "Unexpected server error (INTERNAL_SERVER_ERROR) - no internals are exposed.",
    502: "Upstream model error (EMBEDDING_ERROR, LLM_INVALID_JSON, LLM_EMPTY_RESPONSE, ...).",
    503: "Dependency unavailable or overloaded (OLLAMA_UNAVAILABLE, OLLAMA_MODEL_NOT_FOUND, CHROMA_ERROR, SERVICE_OVERLOADED).",
    504: "Upstream timeout (OLLAMA_TIMEOUT).",
}


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for the given statuses, all using ``ErrorResponse``."""
    return {
        status: {"model": ErrorResponse, "description": ERROR_DESCRIPTIONS[status]}
        for status in statuses
    }


def install_openapi(app: Any) -> None:
    """Make the generated schema match reality.

    FastAPI documents its default ``HTTPValidationError`` for 422, but this service replaces it with
    the standard ``ErrorResponse`` (code ``VALIDATION_ERROR``), so the schema is rewritten accordingly.
    """
    from fastapi.openapi.utils import get_openapi

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title, version=app.version, description=app.description, routes=app.routes
        )
        schemas = schema.get("components", {}).get("schemas", {})
        ref = {"$ref": "#/components/schemas/ErrorResponse"}
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                resp = operation.get("responses", {}).get("422")
                if not resp:
                    continue
                resp["description"] = ERROR_DESCRIPTIONS[422]
                resp["content"] = {"application/json": {"schema": ref}}
        schemas.pop("HTTPValidationError", None)
        schemas.pop("ValidationError", None)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
