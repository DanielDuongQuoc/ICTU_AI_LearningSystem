"""Phase 10 - benchmark script structure (offline fakes) and graceful dependency failure.

The real benchmark (scripts/benchmark.py) runs against live Ollama; here the same code path is
driven with offline stand-ins only to verify record shape, skipping logic and the JSON/CLI output.
No latency thresholds are asserted (they would be hardware dependent and flaky).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.core.exceptions import AppError, ErrorCode
from scripts.benchmark import format_table, make_sample_pdf, run_benchmark
from tests.fakes import LexicalEmbedding, ScriptedLLM, make_repo

LOW = {"similarity_threshold": 0.0}
REQUIRED = {"operation", "elapsed_ms", "success", "error"}


class DownEmbedding(LexicalEmbedding):
    async def embed_documents(self, texts):
        raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "Embedding service (Ollama) is not reachable.")

    async def embed_text(self, text):
        raise AppError(ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE, "down")


class DownLLM(ScriptedLLM):
    async def generate(self, *a, **k):
        raise AppError(ErrorCode.OLLAMA_UNAVAILABLE, "Ollama server is not reachable.")


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "bench.pdf"
    make_sample_pdf(path)
    return path


def _by_op(records):
    return {r["operation"]: r for r in records}


@pytest.mark.asyncio
async def test_benchmark_records_every_stage(pdf):
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(pdf, embedding=LexicalEmbedding(), llm=ScriptedLLM(), repo=repo, rag_options=LOW)
    ops = _by_op(records)
    for op in ("pdf_processing", "text_extraction", "chunking", "embedding", "chroma_insertion",
               "rag_embedding", "rag_chroma_query", "rag_context_build", "rag_retrieval_total",
               "llm_generation", "question_generation", "evaluation", "streaming_evaluation",
               "total_pipeline"):
        assert op in ops, op
    for r in records:
        assert REQUIRED <= set(r)
        assert r["elapsed_ms"] >= 0
    assert all(r["success"] for r in records), [r for r in records if not r["success"]]
    assert ops["rag_retrieval_total"]["retrieved_chunks"] >= 1
    assert ops["rag_retrieval_total"]["context_chars"] > 0
    assert ops["llm_generation"]["output_chars"] > 0
    stream = ops["streaming_evaluation"]
    assert stream["first_token_latency_ms"] is not None
    assert stream["first_token_latency_ms"] <= stream["total_latency_ms"]
    assert stream["event_order"][0] == "status" and stream["event_order"][-1] == "done"
    assert "result" in stream["event_order"]


@pytest.mark.asyncio
async def test_benchmark_skip_llm(pdf):
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(pdf, embedding=LexicalEmbedding(), llm=ScriptedLLM(), repo=repo, skip_llm=True, rag_options=LOW)
    ops = _by_op(records)
    assert "llm_generation" not in ops and "evaluation" not in ops
    assert ops["total_pipeline"]["success"]


@pytest.mark.asyncio
async def test_benchmark_reports_embedding_failure_and_skips_dependents(pdf):
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(pdf, embedding=DownEmbedding(), llm=ScriptedLLM(), repo=repo, rag_options=LOW)
    ops = _by_op(records)
    assert ops["pdf_processing"]["success"]
    assert not ops["indexing_total"]["success"] and "EMBEDDING_SERVICE_UNAVAILABLE" in ops["indexing_total"]["error"]
    for op in ("rag_retrieval_total", "llm_generation", "question_generation", "evaluation",
               "streaming_evaluation"):
        assert not ops[op]["success"] and ops[op]["error"].startswith("skipped:")
    assert not ops["total_pipeline"]["success"]


@pytest.mark.asyncio
async def test_benchmark_reports_llm_failure_and_skips_dependents(pdf):
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(pdf, embedding=LexicalEmbedding(), llm=DownLLM(), repo=repo, rag_options=LOW)
    ops = _by_op(records)
    assert ops["rag_retrieval_total"]["success"]
    assert not ops["llm_generation"]["success"] and "OLLAMA_UNAVAILABLE" in ops["llm_generation"]["error"]
    assert ops["question_generation"]["error"].startswith("skipped:")


@pytest.mark.asyncio
async def test_benchmark_output_is_json_serialisable_and_has_no_content(pdf):
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(pdf, embedding=LexicalEmbedding(), llm=ScriptedLLM(), repo=repo, rag_options=LOW)
    text = json.dumps(records)
    assert "Machine learning is a field" not in text  # no document text in results
    assert "computers learn from data to predict" not in text.lower()  # no student answer
    table = format_table(records)
    assert "operation" in table and "total_pipeline" in table


@pytest.mark.asyncio
async def test_benchmark_invalid_pdf_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    repo = make_repo(f"bench_{uuid.uuid4().hex[:8]}")
    records = await run_benchmark(bad, embedding=LexicalEmbedding(), llm=ScriptedLLM(), repo=repo, rag_options=LOW)
    ops = _by_op(records)
    assert not ops["pdf_processing"]["success"] and ops["pdf_processing"]["error"]
    assert not ops["total_pipeline"]["success"]
