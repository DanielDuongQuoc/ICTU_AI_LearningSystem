"""Shared helpers for the Phase 17 AI-quality evaluation scripts.

* dataset loading + schema validation (``DatasetError`` -> exit code 2),
* run metadata / reproducibility fingerprint (never contains secrets),
* an offline or Ollama-backed service stack over a throw-away ChromaDB,
* JSON + Markdown report writing that refuses to overwrite an existing report.

Exit codes used by every ``scripts/evaluate_*.py`` script::

    0  evaluation ran; (no thresholds requested, or all met)
    1  evaluation failure (infrastructure / unexpected error)  -> NOT a quality verdict
    2  dataset invalid                                         -> data-integrity failure
    3  evaluation ran but a requested quality threshold was not met
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "evaluation" / "datasets"
DEFAULT_REPORT_DIR = ROOT / "evaluation" / "reports"

EXIT_OK, EXIT_FAILURE, EXIT_DATASET, EXIT_THRESHOLD = 0, 1, 2, 3

LABEL_SOURCES = {"synthetic_author", "human_reviewed"}
QUESTION_TYPES = {"short_answer", "multiple_choice", "true_false", "essay"}
DIFFICULTIES = {"easy", "medium", "hard"}
RAG_CATEGORIES = {
    "answerable", "paraphrased", "partial", "false_premise", "out_of_scope",
    "outside_knowledge_request", "fabricate_source_request",
}
ABSTAIN_PHRASES = (
    "does not contain", "do not contain", "insufficient", "not enough information",
    "no information", "cannot answer", "can't answer", "not mentioned", "not provided",
)

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "the and for are was were that this with from into than then them they their there what which "
    "who whom when where why how does did has have had not but can could would should its it's a an "
    "of to in on at by or as is be been being you your our".split()
)


class DatasetError(ValueError):
    """The dataset is malformed (maps to exit code 2)."""


# -- text helpers ----------------------------------------------------------


def content_tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 3 and w not in STOPWORDS}


def token_coverage(needle: str, haystack: str) -> float:
    """Fraction of ``needle`` content tokens that occur in ``haystack`` (1.0 if needle empty)."""
    n = content_tokens(needle)
    if not n:
        return 1.0
    return len(n & content_tokens(haystack)) / len(n)


def looks_like_abstention(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in ABSTAIN_PHRASES)


def mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# -- dataset loading / validation -------------------------------------------


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"Dataset file not found: {p}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p.name}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{p.name}:{lineno}: each line must be a JSON object")
        rows.append(row)
    if not rows:
        raise DatasetError(f"{p.name}: dataset is empty")
    return rows


def load_corpus(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DATASET_DIR / "corpus.json"
    if not p.is_file():
        raise DatasetError(f"Corpus file not found: {p}")
    try:
        corpus = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{p.name}: invalid JSON ({exc.msg})") from exc
    for key in ("dataset_version", "provenance", "documents"):
        if key not in corpus:
            raise DatasetError(f"{p.name}: missing '{key}'")
    seen_chunks: set[str] = set()
    for doc in corpus["documents"]:
        if not doc.get("document_id") or not doc.get("passages"):
            raise DatasetError(f"{p.name}: every document needs document_id and passages")
        for ps in doc["passages"]:
            cid = ps.get("chunk_id")
            if not cid or not str(ps.get("text", "")).strip() or not isinstance(ps.get("page"), int):
                raise DatasetError(f"{p.name}: bad passage in {doc['document_id']}: {ps}")
            if cid in seen_chunks:
                raise DatasetError(f"{p.name}: duplicate chunk_id {cid}")
            seen_chunks.add(cid)
    return corpus


def corpus_index(corpus: dict[str, Any]) -> dict[str, dict[str, str]]:
    """document_id -> {chunk_id: text}."""
    return {d["document_id"]: {p["chunk_id"]: p["text"] for p in d["passages"]} for d in corpus["documents"]}


def _require(row: dict[str, Any], name: str, key: str, types: tuple[type, ...]) -> Any:
    if key not in row:
        raise DatasetError(f"{name}: missing field '{key}'")
    val = row[key]
    if not isinstance(val, types) or (isinstance(val, bool) and bool not in types):
        raise DatasetError(f"{name}: field '{key}' must be {'/'.join(t.__name__ for t in types)}")
    return val


def _common(row: dict[str, Any], idx: dict[str, dict[str, str]], seen: set[str]) -> str:
    cid = _require(row, "row", "case_id", (str,))
    if cid in seen:
        raise DatasetError(f"{cid}: duplicate case_id")
    seen.add(cid)
    doc = _require(row, cid, "document_id", (str,))
    if doc not in idx:
        raise DatasetError(f"{cid}: unknown document_id {doc!r} (not in corpus.json)")
    if _require(row, cid, "label_source", (str,)) not in LABEL_SOURCES:
        raise DatasetError(f"{cid}: label_source must be one of {sorted(LABEL_SOURCES)}")
    return cid


def validate_rag_cases(rows: list[dict[str, Any]], corpus: dict[str, Any]) -> None:
    idx, seen = corpus_index(corpus), set()
    for row in rows:
        cid = _common(row, idx, seen)
        if not _require(row, cid, "query", (str,)).strip():
            raise DatasetError(f"{cid}: empty query")
        rel = _require(row, cid, "relevant_chunk_ids", (list,))
        _require(row, cid, "expected_key_points", (list,))
        abstain = _require(row, cid, "expect_abstain", (bool,))
        if _require(row, cid, "category", (str,)) not in RAG_CATEGORIES:
            raise DatasetError(f"{cid}: unknown category {row['category']!r}")
        if abstain and rel:
            raise DatasetError(f"{cid}: an abstention case must have no relevant_chunk_ids")
        if not abstain and not rel:
            raise DatasetError(f"{cid}: an answerable case needs relevant_chunk_ids")
        for c in rel:
            if c not in idx[row["document_id"]]:
                raise DatasetError(f"{cid}: relevant chunk {c} is not in document {row['document_id']}")


def validate_qgen_cases(rows: list[dict[str, Any]], corpus: dict[str, Any]) -> None:
    idx, seen = corpus_index(corpus), set()
    for row in rows:
        cid = _common(row, idx, seen)
        if _require(row, cid, "question_type", (str,)) not in QUESTION_TYPES:
            raise DatasetError(f"{cid}: invalid question_type")
        if _require(row, cid, "difficulty", (str,)) not in DIFFICULTIES:
            raise DatasetError(f"{cid}: invalid difficulty")
        if _require(row, cid, "number_of_questions", (int,)) < 1:
            raise DatasetError(f"{cid}: number_of_questions must be >= 1")
        if not _require(row, cid, "required_key_points", (list,)):
            raise DatasetError(f"{cid}: required_key_points must not be empty")
        refs = _require(row, cid, "reference_questions", (list,))
        if not refs:
            raise DatasetError(f"{cid}: reference_questions must not be empty")
        for ref in refs:
            for key in ("question", "expected_answer", "difficulty", "type"):
                if not isinstance(ref.get(key), str) or not ref[key].strip():
                    raise DatasetError(f"{cid}: reference question needs '{key}'")
            if ref["type"] == "multiple_choice":
                opts = ref.get("options")
                if not isinstance(opts, list) or len(opts) != 4 or len(set(opts)) != 4:
                    raise DatasetError(f"{cid}: multiple_choice reference needs 4 unique options")
                if ref.get("correct_answer") not in opts:
                    raise DatasetError(f"{cid}: correct_answer must be one of the options")
        _require(row, cid, "expected_properties", (dict,))


def validate_eval_cases(rows: list[dict[str, Any]], corpus: dict[str, Any]) -> None:
    idx, seen = corpus_index(corpus), set()
    for row in rows:
        cid = _common(row, idx, seen)
        for key in ("question", "expected_answer", "student_answer"):
            _require(row, cid, key, (str,))
        max_score = _require(row, cid, "max_score", (int, float))
        if max_score <= 0:
            raise DatasetError(f"{cid}: max_score must be > 0")
        if row.get("expected_error_code"):  # error-contract case: no score band
            if row.get("expected_score_min") is not None or row.get("expected_score_max") is not None:
                raise DatasetError(f"{cid}: an expected_error_code case must not define a score band")
        else:
            lo = _require(row, cid, "expected_score_min", (int, float))
            hi = _require(row, cid, "expected_score_max", (int, float))
            if not 0 <= lo <= hi <= max_score:
                raise DatasetError(f"{cid}: need 0 <= expected_score_min <= expected_score_max <= max_score")
        if "expected_correct" not in row or row["expected_correct"] not in (True, False, None):
            raise DatasetError(f"{cid}: expected_correct must be true, false or null")
        _require(row, cid, "category", (str,))
        for c in row.get("context_chunk_ids") or []:
            if c not in idx[row["document_id"]]:
                raise DatasetError(f"{cid}: context chunk {c} not in document {row['document_id']}")
        if row.get("consistency_group") and not isinstance(row.get("rank"), int):
            raise DatasetError(f"{cid}: consistency_group requires an integer rank")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- reproducibility ----------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unavailable"
    except Exception:
        return "unavailable"


def fingerprint(config: dict[str, Any]) -> str:
    """SHA-256 over the non-secret configuration (keys sorted)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_metadata(
    *, suite: str, mode: str, dataset_version: str, dataset_files: dict[str, Path],
    top_k: int, similarity_threshold: float, model_name: str, embedding_model: str,
    prompt_versions: dict[str, str],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    config = {
        "suite": suite, "mode": mode, "dataset_version": dataset_version, "top_k": top_k,
        "similarity_threshold": similarity_threshold, "model_name": model_name,
        "embedding_model": embedding_model, "prompt_versions": prompt_versions,
        "dataset_sha256": {n: file_sha256(p) for n, p in sorted(dataset_files.items())},
    }
    notes = []
    if mode == "offline":
        notes.append(
            "OFFLINE mode: lexical test embedder and a scripted (non-model) LLM. Numbers describe "
            "pipeline/harness behaviour, NOT the quality of the production model or embeddings."
        )
    else:
        notes.append(
            "OLLAMA mode: model tag is recorded but Ollama does not expose an immutable revision, and "
            "generation is not bit-for-bit reproducible; repeat runs can differ."
        )
    return {
        "evaluation_run_id": f"{suite}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}",
        "timestamp": now.isoformat(),
        "git_commit": git_commit(),
        "dataset_version": dataset_version,
        "prompt_version": prompt_versions,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "configuration_fingerprint": fingerprint(config),
        "mode": mode,
        "evaluation_type": "deterministic_offline" if mode == "offline" else "model_dependent",
        "human_review": "HUMAN REVIEW NOT PERFORMED",
        "reproducibility_notes": notes,
    }


# -- service stack ---------------------------------------------------------------


class EvalStack:
    """RAG + question + evaluation services over a temporary ChromaDB."""

    def __init__(self, mode: str, threshold: Optional[float] = None) -> None:
        from app.core.config import get_settings

        self.settings = get_settings()
        self.mode = mode
        self.threshold = self.settings.similarity_threshold if threshold is None else threshold
        self._tmp: Optional[str] = None
        if mode == "offline":
            from tests.fakes import LexicalEmbedding, RecordingLLM, make_repo

            self.embedding = LexicalEmbedding()
            self.llm = RecordingLLM()
            self.repo = make_repo(f"eval_{uuid.uuid4().hex[:8]}")
            self.embedding_model = self.embedding.model_name
            self.model_name = self.llm.model_name
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            from app.repositories.chroma_repository import ChromaRepository
            from app.services.embedding_service import EmbeddingService
            from app.services.llm_service import LLMService

            self.embedding = EmbeddingService()
            self.llm = LLMService()
            self._tmp = tempfile.mkdtemp(prefix="quality_eval_")
            client = chromadb.PersistentClient(
                path=self._tmp, settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.repo = ChromaRepository(client=client, collection_name="quality_eval")
            self.embedding_model = self.settings.ollama_embedding_model
            self.model_name = self.settings.ollama_model

        from app.services.evaluation_service import EvaluationService
        from app.services.question_service import QuestionGenerationService
        from app.services.rag_service import RAGService
        from app.services.vector_service import VectorService

        self.vector = VectorService(embedding_service=self.embedding, chroma_repo=self.repo)
        self.rag = RAGService(
            embedding_service=self.embedding, chroma_repo=self.repo, llm_service=self.llm,
            similarity_threshold=self.threshold,
        )
        self.questions = QuestionGenerationService(rag_service=self.rag, llm_service=self.llm)
        self.evaluator = EvaluationService(rag_service=self.rag, llm_service=self.llm)
        self.doc_map: dict[str, str] = {}  # fixture document_id -> temp UUID

    async def index_corpus(self, corpus: dict[str, Any]) -> None:
        """Index fixed fixture chunk ids under fresh UUID document ids (never production ids)."""
        from app.schemas.document import Chunk, ChunkMetadata

        for doc in corpus["documents"]:
            real_id = str(uuid.uuid4())
            self.doc_map[doc["document_id"]] = real_id
            chunks = [
                Chunk(chunk_id=p["chunk_id"], document_id=real_id, text=p["text"],
                      metadata=ChunkMetadata(source=f"{doc['document_id']}.pdf", page=p["page"]))
                for p in doc["passages"]
            ]
            await self.vector.index_chunks(chunks=chunks, document_id=real_id, source=f"{doc['document_id']}.pdf")

    async def close(self) -> None:
        await self.embedding.close()
        close_llm = getattr(self.llm, "close", None)
        if close_llm is not None and self.mode != "offline":
            await close_llm()
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)


# -- reports ----------------------------------------------------------------------


def _md_value(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.4f}"
    return "n/a" if v is None else str(v)


def render_markdown(title: str, report: dict[str, Any]) -> str:
    meta = report["metadata"]
    lines = [f"# {title}", "", "## Run metadata", ""]
    for key, val in meta.items():
        if isinstance(val, list):
            val = "; ".join(map(str, val))
        lines.append(f"- **{key}**: {val if not isinstance(val, dict) else json.dumps(val)}")
    lines += ["", "## Metrics", ""]
    for section, values in report.get("sections", {}).items():
        lines += [f"### {section}", ""]
        if isinstance(values, dict):
            lines += ["| metric | value |", "|---|---|"]
            lines += [f"| {k} | {_md_value(v) if not isinstance(v, (dict, list)) else json.dumps(v)} |"
                      for k, v in values.items()]
        else:
            lines.append(str(values))
        lines.append("")
    if report.get("findings"):
        lines += ["## Findings", ""] + [f"- {f}" for f in report["findings"]] + [""]
    if report.get("threshold_failures"):
        lines += ["## Threshold failures", ""] + [f"- {f}" for f in report["threshold_failures"]] + [""]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: Path, name: str, title: str) -> list[Path]:
    """Write ``<name>_<run_id>.json/.md``; mode 'x' guarantees nothing is silently overwritten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = report["metadata"]["evaluation_run_id"]
    paths = []
    for ext, text in (("json", json.dumps(report, indent=2, default=str)), ("md", render_markdown(title, report))):
        path = out_dir / f"{name}_{run_id}.{ext}"
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(text)
        paths.append(path)
    return paths


def parse_thresholds(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise DatasetError(f"--min expects metric=value, got {item!r}")
        out[key.strip()] = float(val)
    return out

