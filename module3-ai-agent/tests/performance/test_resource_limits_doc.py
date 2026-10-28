"""Phase 22 - docs/RESOURCE_LIMITS.md must state exactly the defaults the code uses."""

import re
from pathlib import Path

from app.core.config import Settings

DOC = Path(__file__).resolve().parents[2] / "docs" / "RESOURCE_LIMITS.md"
ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*([0-9.]+)\s*\|", re.M)


def test_documented_defaults_match_settings():
    rows = ROW.findall(DOC.read_text(encoding="utf-8"))
    assert len(rows) >= 20
    defaults = Settings(_env_file=None)
    for name, value in rows:
        assert hasattr(defaults, name.lower()), f"{name} is documented but not a setting"
        assert float(getattr(defaults, name.lower())) == float(value), f"{name}: doc {value}, code {getattr(defaults, name.lower())}"


def test_every_limit_setting_is_documented():
    documented = {n.lower() for n, _ in ROW.findall(DOC.read_text(encoding="utf-8"))}
    limits = {"max_upload_size_mb", "max_pdf_pages", "max_extracted_chars", "max_chunks_per_document",
              "max_concurrent_uploads", "max_top_k", "max_context_chars", "llm_max_concurrency",
              "max_concurrent_embeddings", "embedding_batch_size", "request_timeout_s", "stream_max_duration_s",
              "ollama_timeout", "max_llm_output_chars", "rate_limit_per_minute", "ollama_num_predict"}
    assert limits <= documented, f"undocumented limits: {limits - documented}"
