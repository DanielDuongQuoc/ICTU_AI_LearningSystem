from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.key_pool import parse_api_keys


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field("ICTU_AI_LearningSystem_Module3")
    app_env: str = Field("development")
    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8003)

    # Gemini (Google AI Studio) - the only LLM + embedding provider.
    # GEMINI_API_KEY is a secret: set it only through .env / environment, never commit it.
    # Empty key = the app still starts, /health reports degraded and model calls fail with 503.
    gemini_api_key: str = Field("", repr=False)
    gemini_base_url: str = Field("https://generativelanguage.googleapis.com/v1beta")
    gemini_model: str = Field("gemini-3.5-flash-lite")
    gemini_embedding_model: str = Field("gemini-embedding-001")

    # LLM call parameters (provider-neutral names). Timeouts in seconds.
    llm_timeout: float = Field(120.0)
    # Connection-establishment timeout and the short timeout used by health probes.
    llm_connect_timeout: float = Field(10.0)
    llm_health_timeout: float = Field(5.0)
    llm_temperature: float = Field(0.2)
    llm_max_output_tokens: int = Field(1024)
    # Transient-error retries (HTTP 429/500/502/503/504 and dropped connections) and the backoff between them.
    llm_max_retries: int = Field(1)
    llm_retry_backoff_s: float = Field(0.5)

    # Backpressure: max simultaneous LLM generation calls (0 = unlimited) and how long a call
    # may wait for a slot before being rejected with SERVICE_OVERLOADED (503).
    llm_max_concurrency: int = Field(4)
    llm_queue_timeout_s: float = Field(30.0)

    # Embeddings are bounded the same way. A document is embedded in batches of at most
    # EMBEDDING_BATCH_SIZE texts (sequentially, so memory and provider load stay bounded) and at most
    # MAX_CONCURRENT_EMBEDDINGS embedding calls run at once (0 = unlimited).
    embedding_batch_size: int = Field(64)
    max_concurrent_embeddings: int = Field(4)
    embedding_queue_timeout_s: float = Field(30.0)

    # Exam matrix: maximum total number of questions in one /api/questions/generate-matrix request.
    max_matrix_questions: int = Field(50)

    # Rubrics and rubric-based grading.
    # AI review pass after rubric generation (coverage / observable descriptors); costs one extra model call.
    rubric_review_enabled: bool = Field(True)
    # A criterion counts as "met" (homework PASS/FAIL) when the chosen level is worth >= this share of its points.
    # 0.6: on a 3-level rubric only the top level counts; on a 4-level rubric the top two do (a "partial" level
    # such as "only names the concept" must not let a homework pass).
    grading_criterion_met_ratio: float = Field(0.6)
    # Max characters of one graded submission (typed answer, transcript or concatenated source files).
    max_submission_chars: int = Field(60000)
    # Max submissions in one /api/grading/batch request.
    max_batch_submissions: int = Field(20)
    # Output-token budgets: a full rubric (criteria x levels, Vietnamese) or a per-criterion grading result is
    # much longer than a short answer, so these calls get their own limits instead of LLM_MAX_OUTPUT_TOKENS.
    rubric_max_output_tokens: int = Field(4096)
    grading_max_output_tokens: int = Field(2048)
    # Handwritten answers: max page images per submission sent to the vision model.
    max_handwritten_pages: int = Field(10)

    # Observability
    slow_request_threshold_ms: int = Field(5000)
    # Phase 20: per-stage slow warnings (LLM stages are slow by nature, so they get their own threshold).
    slow_stage_threshold_ms: int = Field(5000)
    slow_llm_threshold_ms: int = Field(60000)
    # Whole-request time budget (seconds). LLM/embedding calls never wait past it and retries are skipped
    # when it is nearly spent. Must be >= LLM_TIMEOUT so one full model call always fits.
    request_timeout_s: float = Field(300.0)
    # Readiness: the persistence volume must be writable and have at least this much free space.
    min_free_disk_mb: int = Field(100)
    # "text" (default, key=value lines) or "json" (one JSON object per line, same fields + redaction).
    log_format: str = Field("text")
    # Explicit log level (DEBUG/INFO/WARNING/ERROR/CRITICAL). Empty = DEBUG in development, else INFO.
    log_level: str = Field("")

    # ChromaDB
    chroma_persist_directory: str = Field("./data/chroma")
    chroma_collection_name: str = Field("learning_documents")

    # Chunking
    chunk_size: int = Field(800)
    chunk_overlap: int = Field(150)

    # Retrieval
    top_k: int = Field(5)
    similarity_threshold: float = Field(0.3)
    max_top_k: int = Field(20)
    max_context_chars: int = Field(12000)

    # Upload
    max_upload_size_mb: int = Field(30)

    # Request text limits (characters). Protect embedding/LLM cost from oversized input.
    max_question_chars: int = Field(2000)
    max_answer_chars: int = Field(10000)

    # Phase 19 - resource-exhaustion limits (all configurable; startup fails if non-positive).
    # Ingestion: a 30 MB upload can still expand into huge text/chunk counts, so these bound the work.
    max_pdf_pages: int = Field(500)
    max_extracted_chars: int = Field(2_000_000)
    max_chunks_per_document: int = Field(3000)
    # Cooperative deadline checked between pages (a single pathological page cannot be interrupted).
    pdf_processing_timeout_s: float = Field(60.0)
    # Simultaneous uploads being processed (0 = unlimited); extra ones are rejected, not queued.
    max_concurrent_uploads: int = Field(2)
    # JSON request body cap in bytes (uploads use MAX_UPLOAD_SIZE_MB instead).
    max_request_body_bytes: int = Field(1_048_576)
    # Cap on raw model output (characters) and on the wall-clock duration of one SSE stream.
    max_llm_output_chars: int = Field(60_000)
    stream_max_duration_s: float = Field(300.0)
    # Per-client sliding-window limit (requests/minute) on expensive endpoints; 0 = disabled.
    # In-process (single instance); clients behind one proxy share a bucket - see docs/SECURITY.md.
    rate_limit_per_minute: int = Field(120)

    # CORS: comma-separated origins. Empty (default) = CORS disabled (server-to-server use).
    cors_allowed_origins: str = Field("")

    @property
    def gemini_api_keys(self) -> list[str]:
        """All configured Gemini API keys (comma/semicolon/whitespace-separated in GEMINI_API_KEY)."""
        return parse_api_keys(self.gemini_api_key)

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def validate_gemini_api_key(cls, v: object) -> str:
        from pathlib import Path

        s = str(v or "").strip().strip("'\"")
        if s.startswith("="):
            env_path = Path(".env")
            if env_path.is_file():
                for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("GEMINI_API_KEY="):
                        file_val = stripped.split("=", 1)[1].strip().strip("'\"")
                        if file_val and not file_val.startswith("="):
                            return file_val
            return s.lstrip("=").strip()
        return s

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v.lower()

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """Cross-field checks; an invalid configuration must stop the app at startup."""
        errors: list[str] = []
        parsed_gemini = urlparse(self.gemini_base_url)
        if parsed_gemini.scheme not in ("http", "https") or not parsed_gemini.netloc:
            errors.append("GEMINI_BASE_URL must be a valid http(s) URL")
        if not self.gemini_model.strip():
            errors.append("GEMINI_MODEL must not be empty")
        if not self.gemini_embedding_model.strip():
            errors.append("GEMINI_EMBEDDING_MODEL must not be empty")
        if self.llm_timeout <= 0:
            errors.append("LLM_TIMEOUT must be > 0")
        if self.llm_connect_timeout <= 0:
            errors.append("LLM_CONNECT_TIMEOUT must be > 0")
        if self.llm_health_timeout <= 0:
            errors.append("LLM_HEALTH_TIMEOUT must be > 0")
        if not (0.0 <= self.llm_temperature <= 2.0):
            errors.append("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm_max_output_tokens <= 0:
            errors.append("LLM_MAX_OUTPUT_TOKENS must be > 0")
        if self.llm_max_retries < 0:
            errors.append("LLM_MAX_RETRIES must be >= 0")
        if self.llm_retry_backoff_s < 0:
            errors.append("LLM_RETRY_BACKOFF_S must be >= 0")
        if not self.chroma_persist_directory.strip():
            errors.append("CHROMA_PERSIST_DIRECTORY must not be empty")
        if not self.chroma_collection_name.strip():
            errors.append("CHROMA_COLLECTION_NAME must not be empty")
        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be > 0")
        if self.chunk_overlap < 0:
            errors.append("CHUNK_OVERLAP must be >= 0")
        if self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be < CHUNK_SIZE")
        if self.top_k <= 0:
            errors.append("TOP_K must be > 0")
        if self.max_top_k <= 0:
            errors.append("MAX_TOP_K must be > 0")
        if self.top_k > self.max_top_k:
            errors.append("TOP_K must be <= MAX_TOP_K")
        if self.max_context_chars <= 0:
            errors.append("MAX_CONTEXT_CHARS must be > 0")
        if self.max_upload_size_mb <= 0:
            errors.append("MAX_UPLOAD_SIZE_MB must be > 0")
        if self.max_question_chars <= 0:
            errors.append("MAX_QUESTION_CHARS must be > 0")
        if self.max_answer_chars <= 0:
            errors.append("MAX_ANSWER_CHARS must be > 0")
        if self.slow_request_threshold_ms <= 0:
            errors.append("SLOW_REQUEST_THRESHOLD_MS must be > 0")
        if self.slow_stage_threshold_ms <= 0 or self.slow_llm_threshold_ms <= 0:
            errors.append("SLOW_STAGE_THRESHOLD_MS and SLOW_LLM_THRESHOLD_MS must be > 0")
        if self.request_timeout_s <= 0:
            errors.append("REQUEST_TIMEOUT_S must be > 0")
        elif self.request_timeout_s < self.llm_timeout:
            errors.append("REQUEST_TIMEOUT_S must be >= LLM_TIMEOUT (the request would always time out before the model call)")
        if self.min_free_disk_mb < 0:
            errors.append("MIN_FREE_DISK_MB must be >= 0")
        if self.log_format.strip().lower() not in {"text", "json"}:
            errors.append("LOG_FORMAT must be text or json")
        if self.llm_max_concurrency < 0:
            errors.append("LLM_MAX_CONCURRENCY must be >= 0 (0 = unlimited)")
        if self.llm_queue_timeout_s <= 0:
            errors.append("LLM_QUEUE_TIMEOUT_S must be > 0")
        if self.embedding_batch_size <= 0:
            errors.append("EMBEDDING_BATCH_SIZE must be > 0")
        if self.max_concurrent_embeddings < 0:
            errors.append("MAX_CONCURRENT_EMBEDDINGS must be >= 0 (0 = unlimited)")
        if self.embedding_queue_timeout_s <= 0:
            errors.append("EMBEDDING_QUEUE_TIMEOUT_S must be > 0")
        if self.max_matrix_questions <= 0:
            errors.append("MAX_MATRIX_QUESTIONS must be > 0")
        if not (0.0 < self.grading_criterion_met_ratio <= 1.0):
            errors.append("GRADING_CRITERION_MET_RATIO must be in (0, 1]")
        if self.max_submission_chars <= 0:
            errors.append("MAX_SUBMISSION_CHARS must be > 0")
        if self.max_batch_submissions <= 0:
            errors.append("MAX_BATCH_SUBMISSIONS must be > 0")
        if self.rubric_max_output_tokens <= 0 or self.grading_max_output_tokens <= 0:
            errors.append("RUBRIC_MAX_OUTPUT_TOKENS and GRADING_MAX_OUTPUT_TOKENS must be > 0")
        if self.max_handwritten_pages <= 0:
            errors.append("MAX_HANDWRITTEN_PAGES must be > 0")
        if self.log_level.strip() and self.log_level.strip().upper() not in {
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        }:
            errors.append("LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL (or empty)")
        for name, value in (
            ("MAX_PDF_PAGES", self.max_pdf_pages), ("MAX_EXTRACTED_CHARS", self.max_extracted_chars),
            ("MAX_CHUNKS_PER_DOCUMENT", self.max_chunks_per_document),
            ("MAX_REQUEST_BODY_BYTES", self.max_request_body_bytes),
            ("MAX_LLM_OUTPUT_CHARS", self.max_llm_output_chars),
        ):
            if value <= 0:
                errors.append(f"{name} must be > 0")
        if self.pdf_processing_timeout_s <= 0:
            errors.append("PDF_PROCESSING_TIMEOUT_S must be > 0")
        if self.stream_max_duration_s <= 0:
            errors.append("STREAM_MAX_DURATION_S must be > 0")
        if self.max_concurrent_uploads < 0:
            errors.append("MAX_CONCURRENT_UPLOADS must be >= 0 (0 = unlimited)")
        if self.rate_limit_per_minute < 0:
            errors.append("RATE_LIMIT_PER_MINUTE must be >= 0 (0 = disabled)")
        if self.app_env == "production":
            if "*" in self.cors_origins:
                errors.append("CORS_ALLOWED_ORIGINS must not contain '*' in production")
            if self.log_level.strip().upper() == "DEBUG":
                errors.append("LOG_LEVEL=DEBUG is not allowed in production")
        if errors:
            raise ValueError("; ".join(errors))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
