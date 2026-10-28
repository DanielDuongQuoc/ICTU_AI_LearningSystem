"""Phase 22 - offline scalability benchmark (``python -m scripts.benchmark_scale``).

Measures the parts of the pipeline that do NOT need Ollama, on this machine, with the real libraries:

    generated PDF -> PyMuPDF extraction -> chunking -> ChromaDB insert -> query -> delete

Embeddings are a deterministic hash embedder (``fake-hash-64``). They stand in for Ollama so the numbers isolate
extraction / chunking / Chroma cost. They are NOT embedding-model latency, NOT LLM latency and NOT production traffic.
Every record carries ``"embedding": "fake-hash-64"`` and the run metadata so results are never compared across
different configurations by accident (see ``scripts/perf_regression.py``).

A throw-away ChromaDB in a temp directory is used; application data is never touched. Nothing sensitive is stored
(the generated text is synthetic).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.chunk_service import create_chunks  # noqa: E402
from app.services.pdf_service import extract_pages  # noqa: E402

DATASET_VERSION = "synthetic-text-v1"
EMBEDDING_LABEL = "fake-hash-64"
DIM = 64
_WORDS = ("machine learning model training data neural network gradient descent loss function overfitting "
          "regularization validation feature vector tensor layer activation optimizer batch epoch inference "
          "classifier regression cluster dataset label supervised unsupervised reinforcement agent reward policy").split()


def hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-words hash embedding (unit length). Test double for Ollama."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        vec[h % DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def page_text(page_no: int, words: int = 330) -> str:
    """~2000 characters of reproducible synthetic text for one page."""
    return " ".join(_WORDS[(page_no * 7 + i * 3) % len(_WORDS)] + (str(page_no) if i % 40 == 0 else "")
                    for i in range(words))


def make_pdf(path: Path, pages: int) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 555, 800), page_text(n), fontsize=9)
    doc.save(str(path))
    doc.close()


def percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; ``None`` when there is no data (never a made-up number)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return round(ordered[int(rank) - 1], 3)


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=str(ROOT))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def run_metadata(queries: int) -> dict[str, Any]:
    s = get_settings()
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),  # None = not a git checkout (NOT VERIFIED)
        "dataset_version": DATASET_VERSION,
        "embedding": EMBEDDING_LABEL,
        "configured_llm_model": s.ollama_model,
        "configured_embedding_model": s.ollama_embedding_model,
        "prompt_version": "n/a (no LLM in this benchmark)",
        "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
        "top_k": s.top_k, "similarity_threshold": 0.0, "max_context_chars": s.max_context_chars,
        "queries_per_document": queries,
        "python": platform.python_version(), "platform": platform.platform(),
    }


def bench_document(pages: int, repo: Any, workdir: Path, queries: int) -> dict[str, Any]:
    """Process one synthetic document end to end and return measured stage timings."""
    pdf = workdir / f"doc_{pages}p.pdf"
    make_pdf(pdf, pages)
    document_id = str(uuid.uuid4())

    tracemalloc.start()
    t = time.perf_counter()
    extracted = extract_pages(pdf)
    extract_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    chunks = create_chunks(extracted, document_id, pdf.name)
    chunk_ms = (time.perf_counter() - t) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t = time.perf_counter()
    vectors = [hash_embed(c.text) for c in chunks]
    embed_ms = (time.perf_counter() - t) * 1000

    ids = [f"{document_id}_{c.chunk_id}" for c in chunks]
    metas = [{"document_id": document_id, "chunk_id": c.chunk_id, "source": pdf.name,
              "page": c.metadata.page if c.metadata.page is not None else (c.metadata.pages or [0])[0],
              "embedding_model": EMBEDDING_LABEL} for c in chunks]
    t = time.perf_counter()
    repo.add_documents(ids=ids, documents=[c.text for c in chunks], embeddings=vectors, metadatas=metas)
    insert_ms = (time.perf_counter() - t) * 1000

    query_ms: list[float] = []
    wrong_document = 0
    total_hits = 0
    for i in range(queries):
        q = hash_embed(" ".join(_WORDS[(i * 5 + j) % len(_WORDS)] for j in range(8)))
        t = time.perf_counter()
        hits = repo.query(query_embedding=q, document_id=document_id, top_k=get_settings().top_k,
                          similarity_threshold=0.0)
        query_ms.append((time.perf_counter() - t) * 1000)
        wrong_document += sum(1 for h in hits if h["document_id"] != document_id)
        total_hits += len(hits)

    return {
        "pages": pages, "chunks": len(chunks), "pdf_bytes": pdf.stat().st_size,
        "pdf_extraction_ms": round(extract_ms, 2), "chunking_ms": round(chunk_ms, 2),
        "embedding_ms_fake": round(embed_ms, 2), "vector_store_ms": round(insert_ms, 2),
        "retrieval_p50_ms": percentile(query_ms, 50), "retrieval_p95_ms": percentile(query_ms, 95),
        "retrieval_samples": len(query_ms),
        "extract_ms_per_page": round(extract_ms / pages, 3), "chunks_per_page": round(len(chunks) / pages, 2),
        "python_peak_kb_extract_chunk": round(peak / 1024, 1),
        "peak_kb_per_page": round(peak / 1024 / pages, 1),
        "wrong_document_hits": wrong_document, "retrieval_hits_total": total_hits, "document_id": document_id,
    }


def run(page_sizes: list[int], queries: int, extra_documents: int) -> dict[str, Any]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app.repositories.chroma_repository import ChromaRepository

    tmp = Path(tempfile.mkdtemp(prefix="m3_scale_"))
    try:
        chroma_dir = tmp / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_dir),
                                           settings=ChromaSettings(anonymized_telemetry=False))
        repo = ChromaRepository(client=client, collection_name="scale_bench")
        documents = [bench_document(p, repo, tmp, queries) for p in page_sizes]
        growth = [{"documents": len(documents), "chunks": repo.count(),
                   "storage_bytes": dir_size_bytes(chroma_dir)}]

        # Growth: add more small documents; query latency on the first document must stay stable.
        base_id = documents[0]["document_id"]
        for n in range(extra_documents):
            bench_document(5, repo, tmp, 0)
            if (n + 1) % max(1, extra_documents // 4) == 0 or n + 1 == extra_documents:
                q_ms = []
                for i in range(queries):
                    q = hash_embed(" ".join(_WORDS[(i * 5 + j) % len(_WORDS)] for j in range(8)))
                    t = time.perf_counter()
                    repo.query(query_embedding=q, document_id=base_id, top_k=get_settings().top_k,
                               similarity_threshold=0.0)
                    q_ms.append((time.perf_counter() - t) * 1000)
                growth.append({"documents": len(documents) + n + 1, "chunks": repo.count(),
                               "storage_bytes": dir_size_bytes(chroma_dir),
                               "retrieval_p50_ms": percentile(q_ms, 50), "retrieval_p95_ms": percentile(q_ms, 95)})

        # Lifecycle: delete the first document; nothing may remain and others must be untouched.
        before = repo.count()
        n_chunks = documents[0]["chunks"]
        t = time.perf_counter()
        repo.delete_document(base_id)
        delete_ms = (time.perf_counter() - t) * 1000
        lifecycle = {"delete_ms": round(delete_ms, 2), "chunks_removed": before - repo.count(),
                     "expected_removed": n_chunks, "orphans_after_delete": repo.count_by_document(base_id)}
        for d in documents:
            d.pop("document_id", None)
        return {"metadata": run_metadata(queries), "documents": documents, "growth": growth,
                "lifecycle": lifecycle, "metrics": flat_metrics(documents)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def flat_metrics(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-stage metrics in the shape ``scripts/perf_regression.py`` compares (largest document)."""
    big = documents[-1]
    out: dict[str, dict[str, Any]] = {}
    for key in ("pdf_extraction_ms", "chunking_ms", "vector_store_ms"):
        out[f"{key}@{big['pages']}p"] = {"value_ms": big[key], "samples": 1}
    out["retrieval_p95_ms"] = {"value_ms": big["retrieval_p95_ms"], "samples": big["retrieval_samples"]}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline scalability benchmark (fake embeddings, real Chroma).")
    ap.add_argument("--pages", default="5,25,100", help="comma separated document sizes in pages")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--extra-documents", type=int, default=20)
    ap.add_argument("--output", default="", help="JSON output file (refuses to overwrite)")
    args = ap.parse_args(argv)
    sizes = sorted({int(p) for p in args.pages.split(",") if p.strip()})
    result = run(sizes, args.queries, args.extra_documents)
    text = json.dumps(result, indent=2)
    if args.output:
        out = Path(args.output)
        if out.exists():
            print(f"refusing to overwrite {out}", file=sys.stderr)
            return 2
        out.write_text(text, encoding="utf-8")
        print(f"written {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
