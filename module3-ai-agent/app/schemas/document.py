from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ExtractedPage(BaseModel):
    page: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(..., description="Raw or cleaned page text")


class ChunkMetadata(BaseModel):
    source: str = Field(..., description="Sanitised source filename")
    page: Optional[int] = Field(
        default=None,
        description="Set when the chunk lies entirely on one page",
    )
    pages: Optional[list[int]] = Field(
        default=None,
        description="Set when the chunk spans more than one page",
    )

    @model_validator(mode="after")
    def require_page_info(self) -> "ChunkMetadata":
        if self.page is None and not self.pages:
            raise ValueError("Chunk metadata must include page or pages")
        return self


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Owning document UUID")
    text: str
    metadata: ChunkMetadata


class ProcessedDocument(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: list[Chunk]
    status: Literal["processed"] = "processed"


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="UUID of the processed document")
    filename: str = Field(..., description="Sanitised original filename")
    pages: int = Field(..., description="Number of pages extracted")
    chunks: int = Field(..., description="Number of text chunks produced")
    status: str = Field("processed", description="Document processing status")


class DocumentInfoResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    message: str


# ── Phase 3 — Search schemas ───────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for semantic search within a document."""
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Max results to return")


class SearchResultItem(BaseModel):
    """A single search hit."""
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    score: float = Field(..., description="Cosine similarity score (1 - distance)")


class SearchResponse(BaseModel):
    """Response from the search endpoint."""
    results: list[SearchResultItem]

