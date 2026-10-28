import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Optional

from app.core.request_context import request_id_var

# key=value-friendly line: every record carries the request id (``-`` outside a request).
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | request_id=%(request_id)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


SERVICE_NAME = "module3-ai-agent"

# Phase 20: defence-in-depth redaction. Application code must not log secrets or content in the first place
# (tests enforce that); this catches an accidental ``token=...`` / ``Authorization: Bearer ...`` / Gemini API key anyway.
_REDACTIONS = (
    re.compile(
        r"(?i)\b(authorization|x-admin-token|x-goog-api-key|admin[_-]?token|gemini[_-]?api[_-]?key|api[_-]?key|passwd|password|secret|token)\b"
        r"(\s*[=:]\s*)(?:bearer\s+)?[^\s,;\"']+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:AIza[A-Za-z0-9_\-]{30,}|AQ\.[A-Za-z0-9_\-]{20,})\b"),
)
# Only these key=value pairs are lifted into JSON fields; free text stays in ``message``.
_JSON_FIELDS = re.compile(
    r"\b(operation|task|stage|status|outcome|duration_ms|error_code|code|document_id|version_id|config_version|"
    r"model|llm_model|prompt_version)=([^\s|,]+)"
)


def _replace_match(m: re.Match[str]) -> str:
    if m.lastindex and m.lastindex >= 2:
        return m.group(1) + m.group(2) + "[REDACTED]"
    if m.group(0).lower().startswith("bearer"):
        return "Bearer [REDACTED]"
    return "[REDACTED]"


def redact(text: str) -> str:
    for pattern in _REDACTIONS:
        text = pattern.sub(_replace_match, text)
    return text


class RedactionFilter(logging.Filter):
    """Redact credential-looking values from every record before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:  # a logging problem must never break the request
            pass
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, service, module, request_id, message (+ lifted fields)."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "module": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": message,
        }
        for key, value in _JSON_FIELDS.findall(message):
            payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure root logger. Call once at application startup."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.addFilter(RequestIdFilter())
    handler.addFilter(RedactionFilter())
    if fmt.strip().lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str, request_id: Optional[str] = None) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
