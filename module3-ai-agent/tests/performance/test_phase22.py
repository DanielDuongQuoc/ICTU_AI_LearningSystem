"""Phase 22 - performance, capacity and scalability tests (offline; no Ollama).

These verify behaviour (bounds, equivalence, classification), not machine-dependent speed, so they are stable in CI.
Measured numbers live in docs/PERFORMANCE.md and come from ``scripts.benchmark_scale``.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import uuid

import httpx
import pytest

from app.core.concurrency import embedding_gate, llm_gate
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, ErrorCode
from app.schemas.document import ExtractedPage
from app.services import chunk_service
from app.services.chunk_service import _Unit, _join_units, _metadata_for_pages, _overlap_units, _units_from_page
from app.services.embedding_service import EmbeddingService
from scripts import benchmark_scale, perf_regression
from tests.fakes import Stack, sample_pdf_bytes


@pytest.fixture(autouse=True)
def _settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ------------------------------------------------------------------ chunking: faster, identical output
def _reference_chunks(pages, size, overlap):
    """The pre-Phase-22 selection loop (rebuilds the candidate string for every unit)."""
    units = []
    for page in pages:
        units.extend(_units_from_page(page, size))
    out, index, previous = [], 0, []
    while index < len(units):
        prefix = _overlap_units(previous, overlap) if out else []
        prefix_len = len(_join_units(prefix)) if prefix else 0
        budget = size if not prefix else max(size - prefix_len, 1)
        selected, cursor = [], index
        while cursor < len(units):
            candidate = _join_units(selected + [units[cursor]])
            if selected and len(candidate) > budget:
                break
            selected.append(units[cursor])
            cursor += 1
            if len(candidate) >= budget:
                break
        if not selected:
            selected, cursor = [units[index]], index + 1
        combined = prefix + selected
        meta = _metadata_for_pages([u.page for u in combined], "s")
        out.append((_join_units(combined), meta.page, meta.pages))
        previous, index = selected, cursor
    return out


def _random_pages(rng: random.Random, n_pages: int) -> list[ExtractedPage]:
    words = "alpha beta gamma delta epsilon model data train loss layer".split()
    pages = []
    for p in range(1, n_pages + 1):
        paras = []
        for _ in range(rng.randint(1, 5)):
            sentences = []
            for _ in range(rng.randint(1, 12)):
                s = " ".join(rng.choice(words) for _ in range(rng.randint(3, 40)))
                sentences.append(s + rng.choice([".", "!", "?", ""]))
            paras.append(" ".join(sentences))
        if rng.random() < 0.3:
            paras.insert(0, "1.2 Section Title")
        pages.append(ExtractedPage(page=p, text="\n\n".join(paras)))
    return pages


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("size,overlap", [(800, 150), (200, 40), (60, 10), (1500, 0)])
def test_optimised_chunking_matches_previous_algorithm(seed, size, overlap):
    pages = _random_pages(random.Random(seed), 6)
    new = chunk_service.create_chunks(pages, "doc", "s", chunk_size=size, chunk_overlap=overlap)
    got = [(c.text, c.metadata.page, c.metadata.pages) for c in new]
    assert got == _reference_chunks(pages, size, overlap)
    assert len({c.chunk_id for c in new}) == len(new)


def test_chunking_scales_roughly_linearly_with_input():
    """Guards against the old quadratic loop: 8x the text must not cost anywhere near 64x (generous bound)."""
    import time

    def run(n):
        pages = [ExtractedPage(page=p, text=" ".join(["word"] * 330)) for p in range(1, n + 1)]
        t = time.perf_counter()
        chunk_service.create_chunks(pages, "d", "s")
        return time.perf_counter() - t

    run(4)  # warm up
    small, large = min(run(10) for _ in range(3)), min(run(80) for _ in range(3))
    assert large < small * 8 * 4, f"chunking looks super-linear: 10 pages {small:.4f}s, 80 pages {large:.4f}s"


# ------------------------------------------------------------------ embedding batching + concurrency bound
def _embed_service(handler) -> EmbeddingService:
    svc = EmbeddingService(base_url="http://ollama.test", model="m:1")
    svc._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return svc


@pytest.mark.asyncio
async def test_embeddings_are_sent_in_bounded_batches_and_keep_order(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "10")
    get_settings.cache_clear()
    sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        sizes.append(len(inputs))
        return httpx.Response(200, json={"embeddings": [[float(t)] for t in inputs]})

    svc = _embed_service(handler)
    texts = [str(i) for i in range(1, 36)]
    vectors = await svc.embed_documents(texts)
    assert sizes == [10, 10, 10, 5] and max(sizes) <= 10
    assert vectors == [[float(t)] for t in texts], "order and count must be preserved across batches"
    await svc.close()


@pytest.mark.asyncio
async def test_small_input_is_still_a_single_request(monkeypatch):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        calls.append(n)
        return httpx.Response(200, json={"embeddings": [[1.0]] * n})

    svc = _embed_service(handler)
    await svc.embed_documents(["a", "b", "c"])
    assert calls == [3]
    await svc.close()


@pytest.mark.asyncio
async def test_embedding_concurrency_never_exceeds_the_configured_limit(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "2")
    monkeypatch.setenv("EMBEDDING_QUEUE_TIMEOUT_S", "10")
    get_settings.cache_clear()
    active = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    svc = _embed_service(handler)
    await asyncio.gather(*[svc.embed_text(f"q{i}") for i in range(12)])
    assert peak == 2
    await svc.close()


@pytest.mark.asyncio
async def test_embedding_overload_is_rejected_with_503_not_queued_forever(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "1")
    monkeypatch.setenv("EMBEDDING_QUEUE_TIMEOUT_S", "0.05")
    get_settings.cache_clear()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    svc = _embed_service(handler)
    first = asyncio.create_task(svc.embed_text("slow"))
    await asyncio.sleep(0.01)
    with pytest.raises(AppError) as err:
        await svc.embed_text("second")
    assert err.value.code == ErrorCode.SERVICE_OVERLOADED
    release.set()
    assert await first == [1.0]
    assert await svc.embed_text("after") == [1.0], "service recovers once the slot is free"
    await svc.close()


@pytest.mark.asyncio
async def test_unlimited_embedding_gate_does_not_block(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_EMBEDDINGS", "0")
    get_settings.cache_clear()
    async with embedding_gate.slot():
        async with embedding_gate.slot():
            pass


# ------------------------------------------------------------------ configuration limits
@pytest.mark.parametrize("field,value", [
    ("embedding_batch_size", 0), ("embedding_batch_size", -1), ("max_concurrent_embeddings", -1),
    ("embedding_queue_timeout_s", 0), ("performance_regression_threshold_percent", 0),
])
def test_new_limits_are_validated(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_new_limit_defaults_are_sane():
    s = Settings(_env_file=None)
    assert (s.embedding_batch_size, s.max_concurrent_embeddings, s.performance_regression_threshold_percent) == (64, 4, 20.0)


# ------------------------------------------------------------------ regression gate classification
def _run(**metrics_ms):
    meta = {"dataset_version": "d1", "embedding": "e", "configured_embedding_model": "m", "configured_llm_model": "l",
            "prompt_version": "p", "chunk_size": 800, "chunk_overlap": 150, "top_k": 5,
            "similarity_threshold": 0.0, "max_context_chars": 12000}
    return {"metadata": meta, "metrics": {k: {"value_ms": v, "samples": 50} for k, v in metrics_ms.items()}}


def test_within_threshold_passes():
    r = perf_regression.compare(_run(retrieval_p95_ms=100), _run(retrieval_p95_ms=115), 20)
    assert r["verdict"] == perf_regression.PASS and r["rows"][0]["regression_percent"] == 15.0


def test_over_threshold_is_a_performance_regression():
    r = perf_regression.compare(_run(retrieval_p95_ms=100), _run(retrieval_p95_ms=130), 20)
    assert r["verdict"] == perf_regression.REGRESSION


def test_improvement_passes():
    assert perf_regression.compare(_run(a=100), _run(a=50), 20)["verdict"] == perf_regression.PASS


def test_regression_is_not_hidden_by_missing_data_elsewhere():
    r = perf_regression.compare(_run(a=100), _run(a=200, b=10), 20)
    assert r["verdict"] == perf_regression.REGRESSION


def test_missing_baseline_metric_is_insufficient_data_never_pass():
    r = perf_regression.compare(_run(a=100), _run(a=100, b=10), 20)
    assert r["verdict"] == perf_regression.INSUFFICIENT


def test_incomparable_configuration_is_insufficient_data():
    base, cur = _run(a=100), _run(a=100)
    cur["metadata"]["configured_embedding_model"] = "other"
    r = perf_regression.compare(base, cur, 20)
    assert r["verdict"] == perf_regression.INSUFFICIENT and "configured_embedding_model" in r["reason"]


def test_too_few_samples_is_insufficient_data():
    base, cur = _run(a=100), _run(a=500)
    cur["metrics"]["a"]["samples"] = 1
    assert perf_regression.compare(base, cur, 20, min_samples=30)["verdict"] == perf_regression.INSUFFICIENT


def test_noise_floor_ignores_sub_millisecond_changes():
    assert perf_regression.compare(_run(a=0.3), _run(a=0.8), 20, noise_floor_ms=1.0)["verdict"] == perf_regression.PASS


def test_cli_exit_codes_distinguish_failure_classes(tmp_path):
    good, bad = tmp_path / "g.json", tmp_path / "b.json"
    good.write_text(json.dumps(_run(a=100)))
    bad.write_text(json.dumps(_run(a=300)))
    assert perf_regression.main([str(good), str(good), "--threshold", "20"]) == 0
    assert perf_regression.main([str(good), str(bad), "--threshold", "20"]) == 1
    assert perf_regression.main([str(good), str(tmp_path / "missing.json")]) == 3


# ------------------------------------------------------------------ benchmark helpers
def test_percentile_nearest_rank_and_no_data():
    assert perf_regression is not None
    assert benchmark_scale.percentile([], 95) is None
    assert benchmark_scale.percentile(list(range(1, 101)), 95) == 95
    assert benchmark_scale.percentile([5], 99) == 5


def test_hash_embedding_is_deterministic_and_unit_length():
    a = benchmark_scale.hash_embed("machine learning data")
    assert a == benchmark_scale.hash_embed("machine learning data")
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9


def test_benchmark_run_keeps_document_isolation_and_leaves_no_orphans():
    result = benchmark_scale.run([2, 3], queries=3, extra_documents=2)
    meta = result["metadata"]
    assert meta["embedding"] == "fake-hash-64" and meta["dataset_version"] and meta["run_id"]
    assert all(d["wrong_document_hits"] == 0 and d["retrieval_hits_total"] > 0 for d in result["documents"])
    life = result["lifecycle"]
    assert life["orphans_after_delete"] == 0 and life["chunks_removed"] == life["expected_removed"]
    assert result["growth"][-1]["chunks"] > result["growth"][0]["chunks"]
    assert not re.search(r"(token|secret|password)", json.dumps(meta), re.I)


# ------------------------------------------------------------------ load / burst through the real API (fakes behind it)
TEXT = "Zebracorn systems explain machine learning. Zebracorn models learn patterns from data."


async def _upload(client):
    r = await client.post("/api/documents/upload",
                          files={"file": ("a.pdf", sample_pdf_bytes([TEXT]), "application/pdf")})
    assert r.status_code in (200, 201), r.text
    return r.json()["document_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [10, 20, 60])
async def test_burst_of_retrievals_succeeds_with_isolated_request_ids(client, monkeypatch, n):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    Stack().install(monkeypatch)
    doc = await _upload(client)
    rs = await asyncio.gather(*[client.post("/api/rag/retrieve", json={"document_id": doc, "question": "zebracorn data"})
                                for _ in range(n)])
    assert [r.status_code for r in rs] == [200] * n
    assert len({r.headers["x-request-id"] for r in rs}) == n


@pytest.mark.asyncio
async def test_llm_burst_over_capacity_is_rejected_cleanly_then_recovers(client, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("LLM_QUEUE_TIMEOUT_S", "0.05")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    stack = Stack().install(monkeypatch)
    doc = await _upload(client)
    payload = {"document_id": doc, "question": "What do zebracorn models learn from?",
               "student_answer": "They learn from data.", "expected_answer": "They learn patterns from data.",
               "max_score": 10}
    original = stack.llm.generate_json

    async def slow(*a, **kw):  # the real OllamaLLMService takes the slot inside its call; the fake does not
        async with llm_gate.slot():
            await asyncio.sleep(0.25)
            return await original(*a, **kw)

    monkeypatch.setattr(stack.llm, "generate_json", slow)
    rs = await asyncio.gather(*[client.post("/api/evaluations/evaluate", json=payload) for _ in range(6)])
    codes = sorted(r.status_code for r in rs)
    assert set(codes) <= {200, 503} and 200 in codes and 503 in codes, codes
    assert all("Traceback" not in r.text for r in rs)
    monkeypatch.setattr(stack.llm, "generate_json", original)
    again = await client.post("/api/evaluations/evaluate", json=payload)
    assert again.status_code == 200, "no permanent broken state after overload"
