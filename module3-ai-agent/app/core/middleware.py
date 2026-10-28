"""Pure-ASGI middleware: request id + request logging + slow-request warning.

Implemented at ASGI level (not ``BaseHTTPMiddleware``) so that streaming (SSE)
responses are never buffered and client-disconnect cancellation propagates unchanged.
Request bodies are never read or logged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Awaitable, Callable, MutableMapping

import asyncio

from app.core import runtime_config
from app.core.config import get_settings
from app.core.exceptions import ErrorCode
from app.core.metrics import metrics
from app.core.request_context import (
    REQUEST_ID_HEADER, deadline_var, request_id_var, resolve_request_id, slowest_stage, stage_times_var,
)
from app.utils.performance import PerformanceTimer

from starlette.exceptions import HTTPException

logger = logging.getLogger("app.request")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


UPLOAD_PATH = "/api/documents/upload"
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")
_SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
]
_API_CSP = (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'")


# Routes whose success/failure is also exposed as a named event (client errors 4xx are not counted as failures).
_ROUTE_EVENTS = {"/api/questions/generate": "question_generation", "/api/evaluations/evaluate": "evaluation"}


def _slowest_text() -> str:
    """``stage:ms`` of the slowest recorded pipeline stage of this request, or ``none``."""
    found = slowest_stage()
    return "none" if found is None else f"{found[0]}:{found[1] * 1000:.0f}ms"


class RequestContextMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        client_value = next(
            (v.decode("latin-1") for k, v in raw_headers if k.lower() == b"x-request-id"), None
        )
        request_id = resolve_request_id(client_value)
        scope.setdefault("state", {})["request_id"] = request_id  # -> request.state.request_id
        token = request_id_var.set(request_id)
        # Phase 18: the AI configuration is pinned per request (a promotion mid-request only affects
        # later requests) and reported in response headers for traceability.
        config_token = runtime_config.pin_current()
        ai_cfg = runtime_config.current()
        # Phase 20: time budget for LLM/embedding calls + per-request stage timings (slow-request logs).
        deadline_token = deadline_var.set(time.monotonic() + get_settings().request_timeout_s)
        stages_token = stage_times_var.set({})

        method = scope.get("method", "")
        path = scope.get("path", "")

        # Phase 19: cap JSON request bodies (uploads are bounded by MAX_UPLOAD_SIZE_MB instead). The
        # AppError raised from receive() is rendered by the exception handlers like any other error.
        body_limit = get_settings().max_request_body_bytes
        declared = next((v for k, v in raw_headers if k.lower() == b"content-length"), b"")
        seen = {"bytes": 0}

        async def limited_receive() -> Message:
            message = await receive()
            if message.get("type") == "http.request" and path != UPLOAD_PATH:
                seen["bytes"] += len(message.get("body", b""))
                if seen["bytes"] > body_limit:
                    # HTTPException passes through FastAPI's body parsing (a bare AppError would become 400).
                    raise HTTPException(status_code=413, detail="Request body too large.")
            return message
        timer = PerformanceTimer().start()
        status: dict[str, int] = {"code": 0}
        logged = {"done": False}

        def finish(note: str = "") -> None:
            if logged["done"]:
                return
            logged["done"] = True
            duration_ms = timer.elapsed_ms()
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            metrics.request_finished(method, route, status["code"], duration_ms / 1000.0)
            event = _ROUTE_EVENTS.get(route)
            if event and status["code"]:
                if status["code"] < 400:
                    metrics.event(event, "ok")
                elif status["code"] >= 500:
                    metrics.event(event, "error")
            logger.info(
                "[request] completed request_id=%s method=%s path=%s status=%s duration_ms=%.1f "
                "config_version=%s%s",
                request_id, method, path, status["code"], duration_ms, ai_cfg.version_id, note,
            )
            threshold = get_settings().slow_request_threshold_ms
            if duration_ms > threshold:
                logger.warning(
                    "[request] slow request_id=%s method=%s path=%s status=%s "
                    "duration_ms=%.1f threshold_ms=%d slowest_stage=%s",
                    request_id, method, path, status["code"], duration_ms, threshold,
                    _slowest_text(),
                )

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() not in (b"x-request-id", b"x-ai-config-version", b"x-ai-config-fingerprint")]
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode("latin-1")))
                headers.append((b"x-ai-config-version", ai_cfg.version_id.encode("latin-1")))
                headers.append((b"x-ai-config-fingerprint", ai_cfg.fingerprint[:16].encode("latin-1")))
                present = {k.lower() for k, _ in headers}
                headers += [h for h in _SECURITY_HEADERS if h[0] not in present]
                if path not in DOC_PATHS and _API_CSP[0] not in present:
                    headers.append(_API_CSP)
                message["headers"] = headers
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finish()

        metrics.request_started()
        logger.info("[request] started request_id=%s method=%s path=%s", request_id, method, path)
        try:
            if path != UPLOAD_PATH and declared.isdigit() and int(declared) > body_limit:
                # Declared size already over the cap: answer without reading (or routing) the body.
                payload = json.dumps({"error": {"code": ErrorCode.REQUEST_TOO_LARGE,
                                                "message": f"Request body exceeds {body_limit} bytes.",
                                                "request_id": request_id}}).encode()
                await send_wrapper({"type": "http.response.start", "status": 413, "headers": [
                    (b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
                await send_wrapper({"type": "http.response.body", "body": payload})
                return
            await self.app(scope, limited_receive, send_wrapper)
        except BaseException as exc:
            # Unhandled errors are rendered by the outer ServerErrorMiddleware; cancellation
            # (client disconnect) is re-raised untouched.
            if isinstance(exc, asyncio.CancelledError):
                metrics.request_cancelled()
            if status["code"] == 0 and isinstance(exc, Exception):
                status["code"] = 500
            finish(f" aborted={type(exc).__name__}")
            raise
        finally:
            deadline_var.reset(deadline_token)
            stage_times_var.reset(stages_token)
            runtime_config.unpin(config_token)
            request_id_var.reset(token)
