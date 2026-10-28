"""Disaster-recovery drill on DISPOSABLE test data (Phase 21).

    python -m scripts.dr_drill [--workdir PATH]

create test document -> backup -> DESTROY the (temporary) persistence -> verify it is really gone ->
restore -> query again -> compare. Each phase runs in its own Python process, so "restart" is real (no
open SQLite handles survive) and the restore cannot be helped by in-process state.

Safety: it only ever touches a freshly created work directory (default: a new temp dir) and refuses a
directory that already exists with content or that contains the configured CHROMA_PERSIST_DIRECTORY.
It never reads or writes production data.

What the numbers mean: durations are for a tiny test collection, fake (lexical) embeddings and a local
disk. They are an OBSERVATION of this drill, not an RTO/RPO measurement for production
(docs/DISASTER_RECOVERY.md keeps Target / Observed / Verified separate).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC_ID = "dr-drill-doc"
TEXT = ("Zebracorn systems explain machine learning. Zebracorn models learn patterns from data. "
        "Quokkaplex indexes speed up database queries. " * 6)


# ----------------------------------------------------------------------------- worker (separate process)
def _worker(phase: str, directory: str, doc_id: str) -> dict[str, Any]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app.repositories.chroma_repository import ChromaRepository
    from app.services.document_service import process_pdf
    from app.services.vector_service import VectorService
    from tests.fakes import LexicalEmbedding, sample_pdf_bytes

    client = chromadb.PersistentClient(path=directory, settings=ChromaSettings(anonymized_telemetry=False))
    repo = ChromaRepository(client=client, collection_name="dr_drill")
    vector = VectorService(embedding_service=LexicalEmbedding(), chroma_repo=repo)
    if phase == "ingest":
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "drill.pdf"
            pdf.write_bytes(sample_pdf_bytes([TEXT]))
            res = process_pdf(pdf, doc_id, source_filename="drill.pdf", content_type="application/pdf", max_bytes=10_000_000)
        asyncio.run(vector.index_chunks(chunks=res.chunks, document_id=doc_id, source=res.filename))
        return {"chunks": repo.count_by_document(doc_id)}
    if phase == "query":
        found = asyncio.run(vector.search("zebracorn data", doc_id, top_k=5, similarity_threshold=0.0))["results"]
        first = found[0] if found else {}
        meta = first.get("metadata", first) if isinstance(first, dict) else {}
        return {
            "chunks": repo.count_by_document(doc_id),
            "results": len(found),
            "source": meta.get("source"),
            "embedding_models": sorted(repo.stored_embedding_models()),
            "top_text_sha256": hashlib.sha256(str(first.get("text", first.get("document", ""))).encode()).hexdigest() if first else None,
        }
    raise SystemExit(f"unknown phase {phase}")


def _spawn(phase: str, directory: Path, doc_id: str = DOC_ID) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-m", "scripts.dr_drill", "--worker", phase, str(directory), doc_id],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"worker {phase} failed: {proc.stderr[-400:]}")
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{"))
    return json.loads(line)


# ----------------------------------------------------------------------------- the drill
def run_drill(workdir: Path) -> dict[str, Any]:
    from app.core.config import get_settings
    from scripts.backup import create_backup
    from scripts.restore import restore_backup

    workdir = Path(workdir).resolve()
    configured = Path(get_settings().chroma_persist_directory).resolve()
    if workdir == configured or configured in workdir.parents or workdir in configured.parents:
        raise RuntimeError("refusing to run the drill on/around the configured CHROMA_PERSIST_DIRECTORY")
    if workdir.exists() and any(workdir.iterdir()):
        raise RuntimeError(f"work directory must be new/empty: {workdir}")
    data, backups = workdir / "chroma", workdir / "backups"
    data.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    steps: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        steps.append((name, ok, detail))

    t = time.perf_counter()
    before = _spawn("ingest", data)
    timings["ingest_s"] = time.perf_counter() - t
    first = _spawn("query", data)
    check("ingest + query before backup", before["chunks"] > 0 and first["results"] > 0, f"chunks={before['chunks']}")

    t = time.perf_counter()
    archive = create_backup(data, backups)
    timings["backup_s"] = time.perf_counter() - t
    check("backup created with checksum", archive.is_file() and archive.with_name(archive.name + ".sha256").is_file(), archive.name)

    shutil.rmtree(data)                                                       # destroy TEST persistence only
    check("test persistence destroyed", not data.exists())

    t = time.perf_counter()
    moved = restore_backup(archive, data, assume_yes=True)
    timings["restore_s"] = time.perf_counter() - t
    after = _spawn("query", data)
    timings["rto_observed_s"] = time.perf_counter() - t                       # restore + first successful query
    check("restore did not need to move anything aside", moved is None)
    check("document retrievable after restore", after["results"] > 0 and after["chunks"] == before["chunks"],
          f"chunks {before['chunks']} -> {after['chunks']}")
    check("source/metadata identical", after["source"] == first["source"] and after["embedding_models"] == first["embedding_models"],
          f"source={after['source']} models={after['embedding_models']}")
    check("top chunk identical", after["top_text_sha256"] == first["top_text_sha256"])
    return {
        "ok": all(ok for _, ok, _ in steps),
        "steps": steps,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "scope": "TEST DATA, fake embeddings, local disk: not a production RPO/RTO measurement",
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--worker", nargs=3, metavar=("PHASE", "DIR", "DOC_ID"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker:
        print(json.dumps(_worker(*a.worker)))
        return 0
    workdir = a.workdir or Path(tempfile.mkdtemp(prefix="m3-dr-drill-"))
    result = run_drill(workdir)
    for name, ok, detail in result["steps"]:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" - {detail}" if detail else ""))
    print("timings (s):", result["timings_s"])
    print(result["scope"])
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
