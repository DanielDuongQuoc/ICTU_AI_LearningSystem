"""Phase 10 - integration tests against a REAL Ollama (+ models) and ChromaDB.

Deselected by default (``pytest -v``).  Run explicitly:

    pytest -v -m integration

They are skipped (with a reason) when Ollama or the configured models are not available.
"""

from __future__ import annotations

import configparser
import shutil
import tempfile
from pathlib import Path

import chromadb
import pytest
import pytest_asyncio
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import OllamaLLMService
from app.services.ollama_checker import check_model_health, check_ollama_health
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from scripts.benchmark import make_sample_pdf, run_benchmark
from scripts.evaluate_rag import DEFAULT_DATASET, evaluate, ingest_corpus, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_integration_marker_registered_and_deselected_by_default():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "pytest.ini")
    assert "integration" in cfg["pytest"]["markers"]
    assert "not integration" in cfg["pytest"]["addopts"]


@pytest_asyncio.fixture()
async def live():
    """Real services on a throw-away ChromaDB; skips when Ollama / models are missing."""
    if not await check_ollama_health():
        pytest.skip("Ollama is not reachable (start it, or run `docker compose up -d`).")
    if not await check_model_health():
        pytest.skip(f"Model {get_settings().ollama_model!r} is not pulled in Ollama.")
    embedding, llm = EmbeddingService(), OllamaLLMService()
    try:
        # fail fast (skip) if the embedding model is missing
        await embedding.embed_text("ping")
    except Exception as exc:
        await embedding.close()
        await llm.close()
        pytest.skip(f"Embedding model unavailable: {getattr(exc, 'message', exc)}")
    tmp = tempfile.mkdtemp(prefix="m3_it_")
    client = chromadb.PersistentClient(path=tmp, settings=ChromaSettings(anonymized_telemetry=False))
    repo = ChromaRepository(client=client, collection_name="integration")
    try:
        yield embedding, llm, repo, Path(tmp)
    finally:
        await embedding.close()
        await llm.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_pipeline_pdf_to_streaming_evaluation(live):
    embedding, llm, repo, tmp = live
    pdf = tmp / "it.pdf"
    make_sample_pdf(pdf)
    records = await run_benchmark(pdf, embedding=embedding, llm=llm, repo=repo)
    failed = [r for r in records if not r["success"]]
    assert not failed, failed
    ops = {r["operation"]: r for r in records}
    assert ops["rag_retrieval_total"]["retrieved_chunks"] >= 1
    assert ops["streaming_evaluation"]["first_token_latency_ms"] is not None
    assert ops["streaming_evaluation"]["event_order"][-1] == "done"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_rag_evaluation_quality_is_reasonable(live):
    embedding, _, repo, _ = live
    data = load_dataset(DEFAULT_DATASET)
    corpus = await ingest_corpus(data, VectorService(embedding_service=embedding, chroma_repo=repo))
    rag = RAGService(embedding_service=embedding, chroma_repo=repo)
    report = await evaluate(data, rag, corpus, k=get_settings().top_k)
    assert report["samples"] == len(data["samples"])
    # sanity floor only (model/hardware dependent): the right chunk should usually be retrieved
    assert report["metrics"]["hit"] >= 0.5
