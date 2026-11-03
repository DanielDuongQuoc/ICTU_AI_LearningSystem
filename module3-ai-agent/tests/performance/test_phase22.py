"""Phase 22 - performance, capacity and scalability tests (offline; Gemini is faked).

These verify behaviour (bounds, equivalence, classification), not machine-dependent speed, so they are stable in CI.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from app.core.concurrency import embedding_gate, llm_gate
from app.core.config import Settings, get_settings
from app.schemas.document import ExtractedPage
from app.services import chunk_service
from app.services.chunk_service import _join_units, _metadata_for_pages, _overlap_units, _units_from_page
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


# ------------------------------------------------------------------ embedding gate
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
    ("embedding_queue_timeout_s", 0),
])
def test_new_limits_are_validated(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_new_limit_defaults_are_sane():
    s = Settings(_env_file=None)
    assert (s.embedding_batch_size, s.max_concurrent_embeddings) == (64, 4)


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

    async def slow(*a, **kw):  # the real GeminiLLMService takes the slot inside its call; the fake does not
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
