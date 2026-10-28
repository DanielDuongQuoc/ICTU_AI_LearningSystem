from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Gemini (Google AI Studio)
    # GEMINI_API_KEY is a secret: set it only through .env / environment, never commit it.
    # When set, Module 3 uses Google AI Studio (Gemini) for LLM generation and embeddings.
    gemini_api_key: str = Field("", repr=False)
    gemini_base_url: str = Field("https://generativelanguage.googleapis.com/v1beta")
    gemini_model: str = Field("gemini-3.5-flash-lite")
    gemini_embedding_model: str = Field("gemini-embedding-001")

    # Ollama
    ollama_base_url: str = Field("http://localhost:11434")
    ollama_model: str = Field("qwen2.5:7b")
    ollama_embedding_model: str = Field("nomic-embed-text")
    ollama_timeout: int = Field(120)
    # Connection-establishment timeout and the short timeout used by health probes (seconds).
    ollama_connect_timeout: float = Field(10.0)
    ollama_health_timeout: float = Field(5.0)
    ollama_temperature: float = Field(0.2)
    ollama_num_predict: int = Field(1024)
    # Transient-error retries for Ollama chat calls (HTTP 502/503/504 only).
    ollama_max_retries: int = Field(1)
    ollama_retry_backoff_s: float = Field(0.5)

    # Backpressure: max simultaneous Ollama generation calls (0 = unlimited) and how long a call
    # may wait for a slot before being rejected with SERVICE_OVERLOADED (503).
    llm_max_concurrency: int = Field(4)
    llm_queue_timeout_s: float = Field(30.0)

    # Phase 22: embeddings are bounded the same way. A document is embedded in batches of at most
    # EMBEDDING_BATCH_SIZE texts (sequentially, so memory and Ollama load stay bounded) and at most
    # MAX_CONCURRENT_EMBEDDINGS Ollama embedding calls run at once (0 = unlimited).
    embedding_batch_size: int = Field(64)
    max_concurrent_embeddings: int = Field(4)
    embedding_queue_timeout_s: float = Field(30.0)
    # Allowed p95 slowdown versus a baseline before scripts/perf_regression.py reports PERFORMANCE REGRESSION.
    performance_regression_threshold_percent: float = Field(20.0)

    # Exam matrix: maximum total number of questions in one /api/questions/generate-matrix request.
    max_matrix_questions: int = Field(50)

    # Observability
    slow_request_threshold_ms: int = Field(5000)
    # Phase 20: per-stage slow warnings (LLM stages are slow by nature, so they get their own threshold).
    slow_stage_threshold_ms: int = Field(5000)
    slow_llm_threshold_ms: int = Field(60000)
    # Whole-request time budget (seconds). LLM/embedding calls never wait past it and retries are skipped
    # when it is nearly spent. Must be >= OLLAMA_TIMEOUT so one full model call always fits.
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

    # AI governance (Phase 18). ADMIN_API_TOKEN empty = the /api/ai-versions admin API is DISABLED.
    # The token is a shared secret: set it only through the environment, never commit it.
    admin_api_token: str = Field("", repr=False)
    ai_versions_dir: str = Field("./data/ai_versions")
    # Optional JSON file overriding the promotion-gate policy defaults (see docs/QUALITY_GATES.md).
    ai_gate_policy_file: str = Field("")

    # CORS: comma-separated origins. Empty (default) = CORS disabled (server-to-server use).
    cors_allowed_origins: str = Field("")

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
        parsed = urlparse(self.ollama_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            errors.append("OLLAMA_BASE_URL must be a valid http(s) URL")
        if not self.ollama_model.strip():
            errors.append("OLLAMA_MODEL must not be empty")
        if not self.ollama_embedding_model.strip():
            errors.append("OLLAMA_EMBEDDING_MODEL must not be empty")
        if self.ollama_timeout <= 0:
            errors.append("OLLAMA_TIMEOUT must be > 0")
        if self.ollama_connect_timeout <= 0:
            errors.append("OLLAMA_CONNECT_TIMEOUT must be > 0")
        if self.ollama_health_timeout <= 0:
            errors.append("OLLAMA_HEALTH_TIMEOUT must be > 0")
        if self.ollama_max_retries < 0:
            errors.append("OLLAMA_MAX_RETRIES must be >= 0")
        if self.ollama_retry_backoff_s < 0:
            errors.append("OLLAMA_RETRY_BACKOFF_S must be >= 0")
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
        elif self.request_timeout_s < self.ollama_timeout:
            errors.append("REQUEST_TIMEOUT_S must be >= OLLAMA_TIMEOUT (the request would always time out before the model call)")
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
        if self.performance_regression_threshold_percent <= 0:
            errors.append("PERFORMANCE_REGRESSION_THRESHOLD_PERCENT must be > 0")
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
        if self.admin_api_token and len(self.admin_api_token) < 16:
            errors.append("ADMIN_API_TOKEN must be at least 16 characters when set")
        if self.app_env == "production":
            if "*" in self.cors_origins:
                errors.append("CORS_ALLOWED_ORIGINS must not contain '*' in production")
            if self.log_level.strip().upper() == "DEBUG":
                errors.append("LOG_LEVEL=DEBUG is not allowed in production")
            # Phase 21 model immutability: an untagged or ":latest" model silently changes when Ollama re-pulls it.
            for env_name, model in (("OLLAMA_MODEL", self.ollama_model), ("OLLAMA_EMBEDDING_MODEL", self.ollama_embedding_model)):
                tag = model.strip().rsplit(":", 1)[1] if ":" in model.strip() else ""
                if tag in ("", "latest"):
                    errors.append(f"{env_name} must carry an explicit, non-'latest' tag in production (got '{model}')")
        if errors:
            raise ValueError("; ".join(errors))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
