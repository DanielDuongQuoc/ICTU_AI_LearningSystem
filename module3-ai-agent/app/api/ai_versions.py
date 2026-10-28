"""AI version governance API (Phase 18) - ADMIN PLANE.

Protected by a shared admin token (``X-Admin-Token`` == ``ADMIN_API_TOKEN``). This is deliberately modest:
one static secret, no users, no roles, no per-person approval. It stops accidental and unauthenticated use
and is disabled entirely (503) when no token is configured; it is NOT a substitute for network isolation or
a real identity system. The optional ``X-Admin-Actor`` header is an unauthenticated label for the audit log.

Clients can never supply a status: versions are created as ``draft`` and only the gated actions move them.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Request

from app.core.config import get_settings
from app.core.exceptions import AppError, ErrorCode
from app.core.openapi import error_responses
from app.core.rate_limit import RateLimited, client_key, limiter
from app.core.request_context import get_request_id
from app.schemas.ai_version import (
    ActiveVersionResponse, AIVersionCreate, AIVersionList, AIVersionRecord, AuditEvent, EvaluateBody,
    ReasonBody, RollbackBody,
)
from app.services.ai_version_service import AIVersionService, clean_actor, get_ai_version_service


logger = logging.getLogger("app.security")
ADMIN_FAIL_LIMIT = 10
ADMIN_FAIL_WINDOW_S = 300.0


def require_admin(request: Request, x_admin_token: Optional[str] = Header(default=None)) -> None:
    configured = get_settings().admin_api_token
    if not configured:
        raise AppError(ErrorCode.ADMIN_DISABLED, "The admin API is disabled: ADMIN_API_TOKEN is not configured.")
    key = client_key(request)
    failures, wait = limiter.count("admin-auth-fail", key, ADMIN_FAIL_WINDOW_S)
    if failures >= ADMIN_FAIL_LIMIT:  # locked out: do not even evaluate the token
        raise RateLimited(wait)
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), configured.encode()):
        limiter.add("admin-auth-fail", key)
        logger.warning("admin authentication failed request_id=%s", get_request_id(request))  # never the value
        raise AppError(ErrorCode.ADMIN_AUTH_REQUIRED, "A valid X-Admin-Token header is required.")


router = APIRouter(prefix="/api/ai-versions", tags=["AI Versions (admin)"], dependencies=[Depends(require_admin)])


def _get_service() -> AIVersionService:
    return get_ai_version_service()


def _actor(x_admin_actor: Optional[str]) -> str:
    return clean_actor(x_admin_actor)


ERRS = error_responses(401, 404, 409, 422, 429, 500, 503)


@router.get("", response_model=AIVersionList, summary="List AI versions", description="List AI versions. Admin plane: requires the X-Admin-Token header.", responses=ERRS)
async def list_versions() -> AIVersionList:
    return _get_service().list()


@router.get("/active", response_model=ActiveVersionResponse, summary="Active version and effective configuration", description="Active version and effective configuration. Admin plane: requires the X-Admin-Token header.",
            responses=ERRS)
async def active_version() -> ActiveVersionResponse:
    return _get_service().active()


@router.get("/audit", response_model=list[AuditEvent], summary="Audit trail (newest last)", description="Audit trail (newest last). Admin plane: requires the X-Admin-Token header.", responses=ERRS)
async def audit(limit: int = Query(100, ge=1, le=1000), version_id: Optional[str] = None) -> list[AuditEvent]:
    return _get_service().audit(limit, version_id)


@router.post("/rollback", response_model=ActiveVersionResponse, responses=ERRS,
             summary="Roll back to an approved/previously active version (or 'env-bootstrap')", description="Roll back to an approved/previously active version (or 'env-bootstrap'). Admin plane: requires the X-Admin-Token header.")
async def rollback(body: RollbackBody, request: Request,
                   x_admin_actor: Optional[str] = Header(default=None)) -> ActiveVersionResponse:
    svc = _get_service()
    await svc.rollback(body.target_version_id, body.reason, _actor(x_admin_actor), get_request_id(request))
    return svc.active()


@router.post("", response_model=AIVersionRecord, status_code=201, responses=ERRS,
             summary="Create a DRAFT version (never active; status cannot be supplied)", description="Create a DRAFT version (never active; status cannot be supplied). Admin plane: requires the X-Admin-Token header.")
async def create_version(body: AIVersionCreate, request: Request,
                         x_admin_actor: Optional[str] = Header(default=None)) -> AIVersionRecord:
    return _get_service().create(body, _actor(x_admin_actor), get_request_id(request))


@router.get("/{version_id}", response_model=AIVersionRecord, summary="Manifest, status and evaluation", description="Manifest, status and evaluation. Admin plane: requires the X-Admin-Token header.", responses=ERRS)
async def get_version(version_id: str) -> AIVersionRecord:
    return _get_service().get(version_id)


@router.post("/{version_id}/evaluate", response_model=AIVersionRecord, responses=ERRS,
             summary="Run the promotion gate over the stored Phase 17 evidence", description="Run the promotion gate over the stored Phase 17 evidence. Admin plane: requires the X-Admin-Token header.")
async def evaluate_version(version_id: str, request: Request, body: Optional[EvaluateBody] = None,
                           x_admin_actor: Optional[str] = Header(default=None)) -> AIVersionRecord:
    return _get_service().evaluate(version_id, body.baseline_version_id if body else None,
                                   _actor(x_admin_actor), get_request_id(request))


@router.post("/{version_id}/approve", response_model=AIVersionRecord, responses=ERRS,
             summary="Approve a validated (gate PASS) version", description="Approve a validated (gate PASS) version. Admin plane: requires the X-Admin-Token header.")
async def approve_version(version_id: str, body: ReasonBody, request: Request,
                          x_admin_actor: Optional[str] = Header(default=None)) -> AIVersionRecord:
    return _get_service().approve(version_id, body.reason, _actor(x_admin_actor), get_request_id(request))


@router.post("/{version_id}/reject", response_model=AIVersionRecord, responses=ERRS,
             summary="Reject a version (terminal)", description="Reject a version (terminal). Admin plane: requires the X-Admin-Token header.")
async def reject_version(version_id: str, body: ReasonBody, request: Request,
                         x_admin_actor: Optional[str] = Header(default=None)) -> AIVersionRecord:
    return _get_service().reject(version_id, body.reason, _actor(x_admin_actor), get_request_id(request))


@router.post("/{version_id}/promote", response_model=AIVersionRecord, responses=ERRS,
             summary="Activate an approved version after preflight + smoke test", description="Activate an approved version after preflight + smoke test. Admin plane: requires the X-Admin-Token header.")
async def promote_version(version_id: str, body: ReasonBody, request: Request,
                          x_admin_actor: Optional[str] = Header(default=None)) -> AIVersionRecord:
    return await _get_service().promote(version_id, body.reason, _actor(x_admin_actor), get_request_id(request))
