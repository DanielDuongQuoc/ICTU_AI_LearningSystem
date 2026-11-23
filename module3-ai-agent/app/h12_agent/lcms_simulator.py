"""Documented C#-compatible internal LCMS HTTP simulator for H12 integration tests & local demos (Milestone 5)."""

from __future__ import annotations

import hmac
import os
from typing import Any, Optional

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.exceptions import ERROR_HTTP_STATUS, AppError, ErrorCode
from app.h12_agent.lcms_tools import (
    _CAMEL_TO_SNAKE_KEYS,
    DialogueOutcomeRecordInput,
    GradingSuggestionRecordInput,
    MockLcmsToolClient,
    QuestionDraftRecordInput,
    _normalize_lcms_keys,
)
from app.h12_agent.types import VisibilityScope

_SNAKE_TO_CAMEL_KEYS: dict[str, str] = {v: k for k, v in _CAMEL_TO_SNAKE_KEYS.items()}


def _to_camel_payload(obj: Any) -> Any:
    """Convert documented snake_case payload keys to ASP.NET Core System.Text.Json camelCase."""
    if isinstance(obj, list):
        return [_to_camel_payload(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _SNAKE_TO_CAMEL_KEYS.get(str(k), str(k)): _to_camel_payload(v)
            for k, v in obj.items()
        }
    return obj


def _format_payload(data: dict[str, Any], *, camel_case: bool) -> dict[str, Any]:
    return _to_camel_payload(data) if camel_case else data


def _app_error_response(exc: AppError) -> JSONResponse:
    status_code = ERROR_HTTP_STATUS.get(exc.code, exc.status_code)
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "code": exc.code, "message": exc.message},
    )


def _record_and_authorize(
    request: Request,
    *,
    x_agent_service_key: Optional[str],
    x_request_id: Optional[str],
    idempotency_key: Optional[str],
) -> Optional[JSONResponse]:
    request.app.state.request_log.append(
        {
            "method": request.method,
            "path": request.url.path,
            "request_id": x_request_id or "",
            "idempotency_key": idempotency_key or "",
            "has_service_key": bool(x_agent_service_key),
        }
    )
    expected = str(request.app.state.service_secret or "")
    if not expected:
        return None
    provided = (x_agent_service_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "code": ErrorCode.AGENT_UNAUTHORIZED,
                "message": "Invalid or missing X-Agent-Service-Key for internal LCMS endpoint.",
            },
        )
    return None


def create_lcms_simulator_app(
    *,
    mock_client: Optional[MockLcmsToolClient] = None,
    service_secret: str = "",
    camel_case: bool = False,
) -> FastAPI:
    """Build a standalone FastAPI simulator matching the 6 internal C# LCMS agent endpoints."""
    sim_app = FastAPI(title="ICTU LCMS Internal Agent Simulator", version="0.1.0")
    sim_app.state.mock_client = mock_client or MockLcmsToolClient()
    sim_app.state.service_secret = service_secret.strip()
    sim_app.state.camel_case = camel_case
    sim_app.state.request_log = []

    @sim_app.get("/api/internal/agent/lessons/{lesson_id}")
    async def sim_fetch_lesson(
        lesson_id: str,
        request: Request,
        version: str = Query("v1"),
        visibility: VisibilityScope = Query("student_safe"),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
    ) -> JSONResponse:
        auth_err = _record_and_authorize(
            request,
            x_agent_service_key=x_agent_service_key,
            x_request_id=x_request_id,
            idempotency_key=None,
        )
        if auth_err is not None:
            return auth_err
        try:
            rec = await request.app.state.mock_client.fetch_authorized_lesson(
                lesson_id=lesson_id,
                lesson_version=version,
                visibility_scope=visibility,
                request_id=x_request_id or "",
            )
        except AppError as exc:
            return _app_error_response(exc)
        payload = {**rec.model_dump(mode="json"), "source_mode": "http"}
        return JSONResponse(content=_format_payload(payload, camel_case=request.app.state.camel_case))

    @sim_app.get("/api/internal/agent/lessons/{lesson_id}/assessment-materials")
    async def sim_fetch_assessment_materials(
        lesson_id: str,
        request: Request,
        version: str = Query("v1"),
        rubric_id: Optional[str] = Query(None),
        visibility: VisibilityScope = Query("student_safe"),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
    ) -> JSONResponse:
        auth_err = _record_and_authorize(
            request,
            x_agent_service_key=x_agent_service_key,
            x_request_id=x_request_id,
            idempotency_key=None,
        )
        if auth_err is not None:
            return auth_err
        try:
            bundle = await request.app.state.mock_client.fetch_approved_questions_rubric(
                lesson_id=lesson_id,
                lesson_version=version,
                rubric_id=rubric_id,
                visibility_scope=visibility,
                request_id=x_request_id or "",
            )
        except AppError as exc:
            return _app_error_response(exc)
        payload = {**bundle.model_dump(mode="json"), "source_mode": "http"}
        return JSONResponse(content=_format_payload(payload, camel_case=request.app.state.camel_case))

    @sim_app.post("/api/internal/agent/assessment-context")
    async def sim_fetch_assessment_context(
        request: Request,
        raw_body: dict[str, Any] = Body(...),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
    ) -> JSONResponse:
        auth_err = _record_and_authorize(
            request,
            x_agent_service_key=x_agent_service_key,
            x_request_id=x_request_id,
            idempotency_key=None,
        )
        if auth_err is not None:
            return auth_err
        body = _normalize_lcms_keys(raw_body)
        try:
            ctx = await request.app.state.mock_client.fetch_assessment_context(
                lesson_id=str(body.get("lesson_id") or ""),
                lesson_version=str(body.get("lesson_version") or "v1"),
                student_id=body.get("student_id"),
                attempt_id=body.get("attempt_id"),
                requested_mode=body.get("requested_mode") or "home_understanding",
                request_id=x_request_id or "",
            )
        except AppError as exc:
            return _app_error_response(exc)
        payload = {**ctx.model_dump(mode="json"), "source_mode": "http"}
        return JSONResponse(content=_format_payload(payload, camel_case=request.app.state.camel_case))

    async def _handle_post_record(
        request: Request,
        raw_body: dict[str, Any],
        model_cls: type[QuestionDraftRecordInput]
        | type[GradingSuggestionRecordInput]
        | type[DialogueOutcomeRecordInput],
        handler_name: str,
        x_agent_service_key: Optional[str],
        x_request_id: Optional[str],
        idempotency_key_header: Optional[str],
    ) -> JSONResponse:
        normalized = _normalize_lcms_keys(raw_body)
        if idempotency_key_header and not normalized.get("idempotency_key"):
            normalized["idempotency_key"] = idempotency_key_header
        eff_idem = idempotency_key_header or normalized.get("idempotency_key")
        auth_err = _record_and_authorize(
            request,
            x_agent_service_key=x_agent_service_key,
            x_request_id=x_request_id,
            idempotency_key=str(eff_idem) if eff_idem else None,
        )
        if auth_err is not None:
            return auth_err
        try:
            parsed = model_cls.model_validate(normalized)
            fn = getattr(request.app.state.mock_client, handler_name)
            receipt = await fn(parsed, request_id=x_request_id or "")
        except ValidationError as exc:
            return JSONResponse(
                status_code=422,
                content={"success": False, "code": ErrorCode.VALIDATION_ERROR, "message": str(exc)},
            )
        except AppError as exc:
            return _app_error_response(exc)
        payload = {**receipt.model_dump(mode="json"), "source_mode": "http"}
        return JSONResponse(content=_format_payload(payload, camel_case=request.app.state.camel_case))

    @sim_app.post("/api/internal/agent/drafts/questions")
    async def sim_record_question_draft(
        request: Request,
        raw_body: dict[str, Any] = Body(...),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
        idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        return await _handle_post_record(
            request,
            raw_body,
            QuestionDraftRecordInput,
            "record_question_draft",
            x_agent_service_key,
            x_request_id,
            idempotency_key_header,
        )

    @sim_app.post("/api/internal/agent/suggestions/grading")
    async def sim_record_grading_suggestion(
        request: Request,
        raw_body: dict[str, Any] = Body(...),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
        idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        return await _handle_post_record(
            request,
            raw_body,
            GradingSuggestionRecordInput,
            "record_grading_suggestion",
            x_agent_service_key,
            x_request_id,
            idempotency_key_header,
        )

    @sim_app.post("/api/internal/agent/outcomes/dialogue")
    async def sim_record_dialogue_outcome(
        request: Request,
        raw_body: dict[str, Any] = Body(...),
        x_agent_service_key: Optional[str] = Header(default=None, alias="X-Agent-Service-Key"),
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
        idempotency_key_header: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        return await _handle_post_record(
            request,
            raw_body,
            DialogueOutcomeRecordInput,
            "record_dialogue_outcome",
            x_agent_service_key,
            x_request_id,
            idempotency_key_header,
        )

    return sim_app


def _main() -> None:
    import uvicorn

    camel = os.environ.get("LCMS_SIM_CAMEL_CASE", "1").strip().lower() in {"1", "true", "yes"}
    secret = os.environ.get("AGENT_SERVICE_SECRET", "").strip()
    host = os.environ.get("LCMS_SIM_HOST", "127.0.0.1")
    port = int(os.environ.get("LCMS_SIM_PORT", "5099"))
    app = create_lcms_simulator_app(service_secret=secret, camel_case=camel)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    _main()
