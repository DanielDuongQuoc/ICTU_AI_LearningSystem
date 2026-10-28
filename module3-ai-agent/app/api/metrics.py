from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.metrics import metrics

router = APIRouter(tags=["Operations"])


@router.get(
    "/metrics",
    include_in_schema=False,  # operational endpoint; keeps the frozen public API contract unchanged
    response_class=PlainTextResponse,
)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus text metrics. Internal only: no authentication, do not expose publicly (docs/OPERATIONS_RUNBOOK.md)."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")
