"""Module 3 performance benchmark - measures the REAL pipeline (no mocks).

    PDF -> extraction -> chunking -> embedding -> Chroma insert -> retrieval -> context
        -> LLM -> question generation -> evaluation -> streaming evaluation

Every operation yields a record ``{operation, elapsed_ms, success, error, ...extras}``.
A throw-away ChromaDB in a temp directory is used, so the application's data is never touched.
If a dependency step fails (e.g. Ollama is down) the steps that depend on it are reported as
failed with ``skipped: ...`` instead of hanging on timeouts.

Usage::

    python scripts/benchmark.py                       # generated sample PDF
    python scripts/benchmark.py --document my.pdf
    python scripts/benchmark.py --skip-llm            # RAG-only (embedding + Chroma)
    python scripts/benchmark.py --output results.json

Needs a running Ollama with OLLAMA_MODEL and OLLAMA_EMBEDDING_MODEL pulled.
Nothing sensitive is printed or stored: no PDF text, no answers, no prompts.
Default output file ``benchmark_results.json`` is git-ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.chunk_service import create_chunks  # noqa: E402
from app.services.document_service import process_pdf  # noqa: E402
from app.services.pdf_service import extract_pages  # noqa: E402
from app.services.question_service import QuestionGenerationService  # noqa: E402
from app.services.evaluation_service import EvaluationService  # noqa: E402
from app.services.rag_service import RAGService, build_context  # noqa: E402
from app.services.vector_service import VectorService  # noqa: E402
from app.utils.performance import PerformanceTimer  # noqa: E402

SAMPLE_PAGES = [
    "Machine learning is a field of artificial intelligence in which computers learn patterns "
    "from data instead of following hand-written rules. A trained model can make predictions "
    "about new, unseen examples. Supervised learning uses labeled examples, while unsupervised "
    "learning discovers structure in unlabeled data.",
    "Overfitting happens when a model memorizes the training data and performs poorly on new "
    "data. Regularization, early stopping and cross-validation are common ways to reduce "
    "overfitting. A validation set is used to tune hyper-parameters of the model.",
    "Neural networks consist of layers of connected neurons that transform their inputs with "
    "weights and activation functions. Backpropagation adjusts the weights by computing "
    "gradients of the loss function, and gradient descent updates them step by step.",
]
BENCH_QUESTION = "What is machine learning?"
BENCH_EXPECTED = "Machine learning lets computers learn patterns from data to make predictions."
BENCH_STUDENT = "It is when computers learn from data to predict things."


def make_sample_pdf(path: Path) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    for text in SAMPLE_PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
    doc.save(str(path))
    doc.close()


class Recorder:
    """Collects one record per operation."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.failed: set[str] = set()

    def add(self, operation: str, elapsed_ms: float, success: bool,
            error: Optional[str] = None, **extra: Any) -> None:
        self.records.append({
            "operation": operation, "elapsed_ms": round(elapsed_ms, 2),
            "success": success, "error": error, **extra,
        })
        if not success:
            self.failed.add(operation)

    def skip(self, operation: str, because: str) -> None:
        self.add(operation, 0.0, False, error=f"skipped: {because} failed")

    async def step(
        self, operation: str, fn: Callable[[], Awaitable[Any]],
        extras: Optional[Callable[[Any], dict[str, Any]]] = None,
    ) -> tuple[bool, Any]:
        timer = PerformanceTimer()
        timer.start()
        try:
            value = await fn()
        except Exception as exc:
            timer.stop()
            self.add(operation, timer.elapsed_ms(), False, error=_describe(exc))
            return False, None
        timer.stop()
        self.add(operation, timer.elapsed_ms(), True, **(extras(value) if extras else {}))
        return True, value


def _describe(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    if code:
        return f"{getattr(code, 'value', code)}: {message}"[:300]
    return f"{type(exc).__name__}: {exc}"[:300]


async def _stream_metrics(service: EvaluationService, document_id: str) -> dict[str, Any]:
    """Run the SSE evaluation and measure first-token / total latency (events are not stored)."""
    timer = PerformanceTimer()
    timer.start()
    first_token_ms: Optional[float] = None
    tokens = 0
    types: list[str] = []
    error: Optional[str] = None
    async for frame in service.evaluate_answer_stream(
        document_id, BENCH_QUESTION, BENCH_STUDENT, BENCH_EXPECTED, 10
    ):
        payload = json.loads(frame[len("data: "):].strip())
        etype = payload.get("type")
        if etype != "token" or not types or types[-1] != "token":
            types.append(etype)
        if etype == "token":
            tokens += 1
            if first_token_ms is None:
                first_token_ms = timer.elapsed_ms()
        elif etype == "error":
            error = f"{payload.get('code')}: {payload.get('message')}"
    total = timer.elapsed_ms()
    if error:
        raise RuntimeError(error)
    return {"first_token_latency_ms": None if first_token_ms is None else round(first_token_ms, 2),
            "total_latency_ms": round(total, 2), "token_count": tokens, "event_order": types}


async def run_benchmark(
    document: Path,
    *,
    embedding: Any,
    llm: Any,
    repo: Any,
    skip_llm: bool = False,
    number_of_questions: int = 2,
    rag_options: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Run every stage against the supplied (normally real) services.

    ``rag_options`` are extra RAGService kwargs (tests lower the similarity threshold for the
    lexical fake embedder; the CLI always uses the configured values).
    """
    settings = get_settings()
    rec = Recorder()
    document_id = str(uuid.uuid4())
    pipeline = PerformanceTimer()
    pipeline.start()

    # 1. PDF processing (validate + extract + clean + chunk) and its parts
    ok, processed = await rec.step(
        "pdf_processing",
        lambda: asyncio.to_thread(process_pdf, document, document_id, source_filename=document.name),
        lambda r: {"pages": r.pages, "chunks": len(r.chunks)},
    )
    _, pages = await rec.step("text_extraction", lambda: asyncio.to_thread(extract_pages, document),
                              lambda p: {"pages": len(p)})
    if pages is not None:
        await rec.step("chunking",
                       lambda: asyncio.to_thread(create_chunks, pages, document_id, document.name),
                       lambda c: {"chunks": len(c)})
    if not ok:
        for op in ("embedding", "chroma_insertion", "rag_retrieval_total", "llm_generation",
                   "question_generation", "evaluation", "streaming_evaluation"):
            rec.skip(op, "pdf_processing")
        return _finish(rec, pipeline)

    # 2. Embedding + Chroma insertion (VectorService reports both timings)
    vector = VectorService(embedding_service=embedding, chroma_repo=repo)
    ok, stats = await rec.step(
        "indexing_total",
        lambda: vector.index_chunks(processed.chunks, document_id, document.name),
        lambda s: {"chunks": s["chunk_count"]},
    )
    if ok:
        rec.add("embedding", stats["embedding_time_ms"], True, chunks=stats["chunk_count"],
                model=stats.get("model_name"))
        rec.add("chroma_insertion", stats["chroma_insert_time_ms"], True, chunks=stats["chunk_count"])
    else:
        rec.failed.add("embedding")
        for op in ("rag_retrieval_total", "llm_generation", "question_generation", "evaluation",
                   "streaming_evaluation"):
            rec.skip(op, "indexing_total")
        return _finish(rec, pipeline)

    rag = RAGService(embedding_service=embedding, chroma_repo=repo, llm_service=llm, **(rag_options or {}))

    # 3. Retrieval, broken into its components, then the full RAG call
    ok, vec = await rec.step("rag_embedding", lambda: embedding.embed_text(BENCH_QUESTION),
                             lambda v: {"dimensions": len(v)})
    if ok:
        async def _query() -> Any:
            return await asyncio.to_thread(
                repo.query, vec, document_id, settings.top_k, settings.similarity_threshold)
        ok_q, hits = await rec.step("rag_chroma_query", _query, lambda h: {"hits": len(h)})
        if ok_q:
            async def _ctx() -> Any:
                return build_context(hits, settings.max_context_chars)
            await rec.step("rag_context_build", _ctx,
                           lambda r: {"context_chars": len(r[0]), "included_chunks": len(r[1])})
    await rec.step(
        "rag_retrieval_total",
        lambda: rag.retrieve(document_id, BENCH_QUESTION),
        lambda r: {"retrieved_chunks": len(r["chunks"]), "context_chars": len(r["context"]),
                   "insufficient_context": r["insufficient_context"]},
    )

    if skip_llm:
        return _finish(rec, pipeline)

    # 4. LLM
    ok, _ = await rec.step(
        "llm_generation",
        lambda: llm.generate("You are a concise assistant.", "Reply with the single word: OK"),
        lambda out: {"output_chars": len(out)},
    )
    if not ok:
        for op in ("question_generation", "evaluation", "streaming_evaluation"):
            rec.skip(op, "llm_generation")
        return _finish(rec, pipeline)

    questions = QuestionGenerationService(rag_service=rag, llm_service=llm)
    evaluator = EvaluationService(rag_service=rag, llm_service=llm)
    await rec.step(
        "question_generation",
        lambda: questions.generate_questions(document_id, number_of_questions),
        lambda r: {"questions": len(r.questions)},
    )
    await rec.step(
        "evaluation",
        lambda: evaluator.evaluate_answer(document_id, BENCH_QUESTION, BENCH_STUDENT, BENCH_EXPECTED, 10),
        lambda r: {"score": r.score, "sources": len(r.sources)},
    )
    ok, metrics = await rec.step("streaming_evaluation", lambda: _stream_metrics(evaluator, document_id))
    if ok:  # promote the interesting streaming numbers onto the record
        rec.records[-1].update(metrics)
    return _finish(rec, pipeline)


def _finish(rec: Recorder, pipeline: PerformanceTimer) -> list[dict[str, Any]]:
    pipeline.stop()
    rec.add("total_pipeline", pipeline.elapsed_ms(), not rec.failed,
            error=None if not rec.failed else f"failed operations: {', '.join(sorted(rec.failed))}")
    return rec.records


def format_table(records: list[dict[str, Any]]) -> str:
    lines = [f"{'operation':<24}{'elapsed_ms':>12}  {'success':<8}details"]
    for r in records:
        extras = {k: v for k, v in r.items() if k not in ("operation", "elapsed_ms", "success", "error")}
        detail = r["error"] if r["error"] else ", ".join(f"{k}={v}" for k, v in extras.items())
        lines.append(f"{r['operation']:<24}{r['elapsed_ms']:>12.1f}  {str(r['success']):<8}{detail}")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app.repositories.chroma_repository import ChromaRepository
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_service import OllamaLLMService

    tmp_dir = tempfile.mkdtemp(prefix="m3_bench_")
    embedding, llm = EmbeddingService(), OllamaLLMService()
    try:
        document = Path(args.document) if args.document else Path(tmp_dir) / "benchmark_sample.pdf"
        if not args.document:
            make_sample_pdf(document)
        client = chromadb.PersistentClient(
            path=str(Path(tmp_dir) / "chroma"), settings=ChromaSettings(anonymized_telemetry=False))
        repo = ChromaRepository(client=client, collection_name="benchmark")
        return await run_benchmark(document, embedding=embedding, llm=llm, repo=repo,
                                   skip_llm=args.skip_llm, number_of_questions=args.questions)
    finally:
        await embedding.close()
        await llm.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the Module 3 pipeline against real services.")
    parser.add_argument("--document", help="PDF to benchmark (default: a generated sample PDF)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip all LLM stages (RAG-only)")
    parser.add_argument("--questions", type=int, default=2, help="Questions to generate (default 2)")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON results file")
    args = parser.parse_args(argv)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    records = asyncio.run(_run(args))
    print(format_table(records))
    settings = get_settings()
    payload = {"started_at": started, "model": settings.ollama_model,
               "embedding_model": settings.ollama_embedding_model, "results": records}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")
    return 0 if all(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
