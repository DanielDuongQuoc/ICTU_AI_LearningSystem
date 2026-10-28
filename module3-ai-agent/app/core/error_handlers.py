"""Global exception handlers: every API error uses the same contract.

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

No traceback, exception text or Pydantic internals are ever sent to the client; the
details are logged server-side together with the request id.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError, ErrorCode
from app.core.metrics import metrics
from app.core.request_context import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger("app.exceptions")

_DOCUMENT_ID_ERROR = (ErrorCode.INVALID_DOCUMENT_ID, "document_id must be a valid UUID.")

# Endpoints whose request-field problems keep specific domain codes (HTTP 400).
_QUESTION_FIELD_ERRORS = {
    "document_id": _DOCUMENT_ID_ERROR,
    "number_of_questions": (
        ErrorCode.INVALID_NUMBER_OF_QUESTIONS, "number_of_questions must be between 1 and 20."
    ),
    "difficulty": (
        ErrorCode.INVALID_DIFFICULTY, "difficulty must be one of ['easy', 'medium', 'hard']."
    ),
    "question_type": (
        ErrorCode.INVALID_QUESTION_TYPE,
        "question_type must be one of "
        "['multiple_choice', 'true_false', 'short_answer', 'essay'].",
    ),
}
_EVALUATION_FIELD_ERRORS = {
    "document_id": _DOCUMENT_ID_ERROR,
    "question": (ErrorCode.EMPTY_QUESTION, "question must not be empty."),
    "student_answer": (ErrorCode.EMPTY_STUDENT_ANSWER, "student_answer must not be empty."),
    "expected_answer": (ErrorCode.EMPTY_EXPECTED_ANSWER, "expected_answer must not be empty."),
    "max_score": (ErrorCode.INVALID_SCORE, "max_score must be greater than 0 and at most 100."),
}
ENDPOINT_FIELD_ERRORS: dict[str, dict[str, tuple[str, str]]] = {
    "/api/questions/generate": _QUESTION_FIELD_ERRORS,
    "/api/evaluations/evaluate": _EVALUATION_FIELD_ERRORS,
}

_HTTP_STATUS_CODES: dict[int, tuple[str, str]] = {
    404: (ErrorCode.NOT_FOUND, "Resource not found."),
    405: (ErrorCode.METHOD_NOT_ALLOWED, "Method not allowed."),
    413: (ErrorCode.REQUEST_TOO_LARGE, "Request body is too large."),
}


def error_response(
    request: Request, code: str, message: str, status_code: int
) -> JSONResponse:
    request_id = get_request_id(request)
    metrics.error(code)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={REQUEST_ID_HEADER: request_id},
    )


def _log_context(request: Request) -> str:
    document_id = request.path_params.get("document_id") if request.scope.get("path_params") else None
    return (
        f"request_id={get_request_id(request)} method={request.method} "
        f"endpoint={request.url.path} document_id={document_id}"
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log = logger.error if exc.status_code >= 500 else logger.warning
    log(
        "AppError error_code=%s status=%s exception_type=%s %s",
        exc.code, exc.status_code, type(exc).__name__, _log_context(request),
    )
    response = error_response(request, exc.code, exc.message, exc.status_code)
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
    return response


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message = _HTTP_STATUS_CODES.get(
        exc.status_code, (ErrorCode.HTTP_ERROR, "The request could not be processed.")
    )
    logger.warning(
        "HTTPException error_code=%s status=%s %s", code, exc.status_code, _log_context(request)
    )
    response = error_response(request, code, message, exc.status_code)
    for key, value in (getattr(exc, "headers", None) or {}).items():  # e.g. Allow
        response.headers[key] = value
    return response


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Log field + error type only (never the submitted values).
    details = [f"{'.'.join(str(p) for p in e.get('loc', ()))}:{e.get('type')}" for e in exc.errors()]

    field_errors = ENDPOINT_FIELD_ERRORS.get(request.url.path.rstrip("/"))
    if field_errors is not None:
        for err in exc.errors():
            loc = err.get("loc", ())
            field = loc[-1] if loc else None
            if field == "document_id":
                value = err.get("input")
                if err.get("type") == "missing" or (isinstance(value, str) and not value.strip()):
                    code, message = ErrorCode.EMPTY_DOCUMENT_ID, "document_id must not be empty."
                else:
                    code, message = field_errors["document_id"]
            elif field in field_errors:
                code, message = field_errors[field]
            else:
                continue
            logger.warning("Validation error error_code=%s fields=%s %s", code, details, _log_context(request))
            return error_response(request, code, message, 400)

    logger.warning(
        "Validation error error_code=%s fields=%s %s",
        ErrorCode.VALIDATION_ERROR, details, _log_context(request),
    )
    return error_response(request, ErrorCode.VALIDATION_ERROR, "Invalid request data.", 422)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception error_code=%s exception_type=%s %s",
        ErrorCode.INTERNAL_SERVER_ERROR, type(exc).__name__, _log_context(request),
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return error_response(
        request, ErrorCode.INTERNAL_SERVER_ERROR, "An unexpected error occurred.", 500
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
