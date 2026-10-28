from typing import Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(None, description="Request-scoped UUID")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="ok or degraded")
    ollama: bool = Field(..., description="True if Ollama is reachable")
    chromadb: bool = Field(..., description="True if ChromaDB is operational")
    version: str = Field(..., description="API version string")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Per-dependency status (api, chroma, ollama, model): ok or error",
    )
