"""How the Agent reaches the grading core: in-process (``local``) or through the AI Service REST API (``api``)."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from app.core.exceptions import AppError, ErrorCode
from app.schemas.grading import GradeRequest, GradeResult


class GradingBackend(Protocol):
    async def grade(self, req: GradeRequest) -> GradeResult: ...
    async def close(self) -> None: ...


class LocalBackend:
    """Runs the same GradingService as the API, in this process (Gemini key from .env / environment)."""

    def __init__(self) -> None:
        from app.services.grading_service import GradingService

        self._svc = GradingService()

    async def grade(self, req: GradeRequest) -> GradeResult:
        return await self._svc.grade(req)

    async def close(self) -> None:
        from app.services.llm_service import get_llm_service

        await get_llm_service().close()


class ApiBackend:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def grade(self, req: GradeRequest) -> GradeResult:
        try:
            resp = await self._client.post("/api/grading/grade", json=req.model_dump(mode="json"))
        except httpx.TimeoutException as exc:
            raise AppError(ErrorCode.LLM_TIMEOUT, "AI Service did not answer in time.") from exc
        except httpx.HTTPError as exc:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "AI Service is not reachable.") from exc
        if resp.status_code == 200:
            return GradeResult.model_validate(resp.json())
        err: dict[str, Any] = {}
        try:
            err = resp.json().get("error", {})
        except ValueError:
            pass
        raise AppError(err.get("code") or ErrorCode.GRADING_ERROR,
                       err.get("message") or f"AI Service returned HTTP {resp.status_code}.",
                       status_code=resp.status_code)

    async def close(self) -> None:
        await self._client.aclose()
