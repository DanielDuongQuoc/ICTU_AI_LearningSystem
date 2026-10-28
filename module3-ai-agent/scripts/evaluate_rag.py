"""Basic, deterministic RAG retrieval evaluation (Hit@K, Precision@K, Recall@K).

Flow: load dataset -> index its corpus (real embeddings + a throw-away ChromaDB) -> send each
question to ``RAGService.retrieve`` -> compare retrieved chunk ids with the ground truth ->
average the metrics.  No LLM is involved.

Dataset format (see ``tests/data/rag_eval.json``)::

    {"documents": [{"name": "doc1", "pages": ["page 1 text", ...]}],
     "samples":   [{"id": "q1", "document": "doc1", "question": "...",
                    "relevant_phrases": ["..."]}]}

Ground truth for a sample = chunks of its document whose text contains one of
``relevant_phrases`` (case-insensitive)  UNION  ids in ``relevant_chunk_ids``.  A sample can
instead point at an already indexed document with ``"document_id"`` + ``relevant_chunk_ids``
(use ``--use-app-store`` to read the application's own ChromaDB).

Usage::

    python scripts/evaluate_rag.py [--dataset tests/data/rag_eval.json] [--k 5] [--output rag_eval_results.json]

Requires a running Ollama with the embedding model pulled (the app's real configuration is used).

Phase 17 suite (a ``.jsonl`` dataset - the default; implemented in ``scripts/rag_quality.py``)::

    python scripts/evaluate_rag.py [--mode offline|ollama] [--k 5] [--out-dir evaluation/reports]
                                   [--answers] [--min hit@5=0.9 ...] [--validate-only]

``--mode offline`` (default) uses a lexical test embedder and never calls a model: it measures the
retrieval pipeline (document filtering, thresholding, metadata), not production embedding quality.
Exit codes: 0 ok, 1 evaluation failure, 2 dataset invalid, 3 requested threshold not met.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.document import Chunk, ExtractedPage  # noqa: E402
from app.services.chunk_service import create_chunks  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402
from app.services.vector_service import VectorService  # noqa: E402
from app.utils.rag_metrics import average, score_sample  # noqa: E402

DEFAULT_DATASET = ROOT / "tests" / "data" / "rag_eval.json"
DEFAULT_JSONL_DATASET = ROOT / "evaluation" / "datasets" / "rag_eval.jsonl"


# -- dataset ------------------------------------------------------------


def load_dataset(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("samples"):
        raise ValueError("Dataset must be an object with a non-empty 'samples' list.")
    names = {d["name"] for d in data.get("documents", [])}
    for s in data["samples"]:
        if not s.get("question", "").strip():
            raise ValueError(f"Sample {s.get('id')!r}: empty question.")
        if "document" in s:
            if s["document"] not in names:
                raise ValueError(f"Sample {s.get('id')!r}: unknown document {s['document']!r}.")
        elif "document_id" not in s:
            raise ValueError(f"Sample {s.get('id')!r}: needs 'document' or 'document_id'.")
        if not s.get("relevant_phrases") and not s.get("relevant_chunk_ids"):
            raise ValueError(f"Sample {s.get('id')!r}: needs relevant_phrases or relevant_chunk_ids.")
    return data


# -- indexing -----------------------------------------------------------


async def ingest_corpus(
    dataset: dict[str, Any], vector_service: VectorService
) -> dict[str, dict[str, Any]]:
    """Chunk + embed + index every corpus document. Returns name -> {document_id, chunks}."""
    corpus: dict[str, dict[str, Any]] = {}
    for doc in dataset.get("documents", []):
        document_id = str(uuid.uuid4())
        pages = [ExtractedPage(page=i + 1, text=t) for i, t in enumerate(doc["pages"])]
        chunks = create_chunks(pages, document_id, f"{doc['name']}.pdf")
        await vector_service.index_chunks(chunks=chunks, document_id=document_id, source=f"{doc['name']}.pdf")
        corpus[doc["name"]] = {"document_id": document_id, "chunks": chunks}
    return corpus


def relevant_chunk_ids(sample: dict[str, Any], chunks: list[Chunk]) -> set[str]:
    phrases = [p.lower() for p in sample.get("relevant_phrases", [])]
    ids = set(sample.get("relevant_chunk_ids", []))
    for chunk in chunks:
        text = chunk.text.lower()
        if any(p in text for p in phrases):
            ids.add(chunk.chunk_id)
    return ids


# -- evaluation ---------------------------------------------------------


async def evaluate(
    dataset: dict[str, Any],
    rag: RAGService,
    corpus: dict[str, dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    skipped = 0
    for sample in dataset["samples"]:
        if "document" in sample:
            entry = corpus[sample["document"]]
            document_id, chunks = entry["document_id"], entry["chunks"]
        else:
            document_id, chunks = sample["document_id"], []
        relevant = relevant_chunk_ids(sample, chunks)
        if not relevant:
            skipped += 1  # ground truth could not be resolved -> cannot be scored
            per_sample.append({"id": sample.get("id"), "skipped": "no relevant chunks resolved"})
            continue
        result = await rag.retrieve(document_id=document_id, question=sample["question"], top_k=k)
        retrieved = [c["chunk_id"] for c in result["chunks"]]
        score = score_sample(retrieved, relevant, k)
        scores.append(score)
        per_sample.append({
            "id": sample.get("id"), "retrieved": len(retrieved),
            "relevant": len(relevant), **score,
        })
    return {"samples": len(scores), "skipped": skipped, "k": k,
            "metrics": average(scores), "per_sample": per_sample}


def format_report(report: dict[str, Any]) -> str:
    k, m = report["k"], report["metrics"]
    lines = [
        "RAG Evaluation",
        f"Samples: {report['samples']}" + (f" (skipped: {report['skipped']})" if report["skipped"] else ""),
        f"K: {k}",
        f"Hit@{k}: {m['hit']:.2f}",
        f"Precision@{k}: {m['precision']:.2f}",
        f"Recall@{k}: {m['recall']:.2f}",
    ]
    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app.core.config import get_settings
    from app.repositories.chroma_repository import ChromaRepository, init_chroma
    from app.services.embedding_service import EmbeddingService

    dataset = load_dataset(args.dataset)
    k = args.k or get_settings().top_k
    embedding = EmbeddingService()
    tmp_dir: Optional[str] = None
    try:
        if args.use_app_store:
            init_chroma()
            repo = ChromaRepository()
        else:
            tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")
            client = chromadb.PersistentClient(
                path=tmp_dir, settings=ChromaSettings(anonymized_telemetry=False)
            )
            repo = ChromaRepository(client=client, collection_name="rag_eval")
        rag = RAGService(embedding_service=embedding, chroma_repo=repo)
        corpus: dict[str, dict[str, Any]] = {}
        if dataset.get("documents"):
            corpus = await ingest_corpus(dataset, VectorService(embedding_service=embedding, chroma_repo=repo))
        return await evaluate(dataset, rag, corpus, k)
    finally:
        await embedding.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality (Hit/Precision/Recall@K).")
    parser.add_argument("--dataset", default=str(DEFAULT_JSONL_DATASET),
                        help=".jsonl = Phase 17 suite (default); .json = legacy Phase 10 dataset "
                             f"(e.g. {DEFAULT_DATASET})")
    parser.add_argument("--k", type=int, default=None, help="Top-K (default: TOP_K from config)")
    parser.add_argument("--output", default=None, help="Legacy (.json dataset): write the report as JSON")
    parser.add_argument("--use-app-store", action="store_true",
                        help="Legacy (.json dataset): use the application's ChromaDB instead of a temporary one")
    parser.add_argument("--mode", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--out-dir", default=str(ROOT / "evaluation" / "reports"),
                        help="Phase 17: where JSON/Markdown reports are written (never overwritten)")
    parser.add_argument("--answers", action="store_true", help="Phase 17: also generate/score answers (ollama mode)")
    parser.add_argument("--min", action="append", metavar="METRIC=VALUE",
                        help="Phase 17: exit 3 if a metric is below VALUE (repeatable)")
    parser.add_argument("--validate-only", action="store_true", help="Phase 17: only validate the dataset")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Phase 17: override SIMILARITY_THRESHOLD for this run (recorded in the report)")
    args = parser.parse_args(argv)
    if str(args.dataset).endswith(".jsonl"):
        from scripts.rag_quality import main_jsonl

        return main_jsonl(args)
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:  # operator-facing CLI: concise message, no traceback
        print(f"RAG evaluation failed: {type(exc).__name__}: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 1
    print(format_report(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

