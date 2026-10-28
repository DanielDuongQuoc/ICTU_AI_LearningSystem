"""Phase 10 - retrieval metrics, evaluation dataset and the offline evaluation flow."""

from __future__ import annotations

import json
import uuid

import pytest

from app.services.rag_service import RAGService
from app.services.vector_service import VectorService
from app.utils.rag_metrics import average, hit_at_k, precision_at_k, recall_at_k, score_sample
from scripts.evaluate_rag import (
    DEFAULT_DATASET, evaluate, format_report, ingest_corpus, load_dataset, relevant_chunk_ids,
)
from tests.fakes import LexicalEmbedding, make_repo

# -- metrics ------------------------------------------------------------


def test_recall_precision_hit_definitions():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "c", "z"]
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert hit_at_k(retrieved, relevant, 5) == 1.0


def test_metrics_respect_k_cutoff():
    retrieved = ["x", "y", "a"]
    assert hit_at_k(retrieved, ["a"], 2) == 0.0
    assert recall_at_k(retrieved, ["a"], 2) == 0.0
    assert hit_at_k(retrieved, ["a"], 3) == 1.0
    assert precision_at_k(retrieved, ["a"], 3) == pytest.approx(1 / 3)


def test_precision_denominator_is_k_even_with_fewer_results():
    assert precision_at_k(["a"], ["a"], 5) == pytest.approx(1 / 5)


def test_duplicates_counted_once():
    assert recall_at_k(["a", "a", "a"], ["a", "b"], 3) == 0.5


def test_empty_retrieval_scores_zero():
    assert score_sample([], ["a"], 5) == {"hit": 0.0, "precision": 0.0, "recall": 0.0}


def test_invalid_arguments():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 5)
    with pytest.raises(ValueError):
        hit_at_k(["a"], ["a"], 0)


def test_average():
    avg = average([{"hit": 1, "precision": 0.4, "recall": 1}, {"hit": 0, "precision": 0.0, "recall": 0.5}])
    assert avg == {"hit": 0.5, "precision": 0.2, "recall": 0.75}
    assert average([]) == {"hit": 0.0, "precision": 0.0, "recall": 0.0}


# -- dataset ------------------------------------------------------------


def test_dataset_loads_and_is_wellformed():
    data = load_dataset(DEFAULT_DATASET)
    assert len(data["samples"]) >= 5
    names = {d["name"] for d in data["documents"]}
    assert len(names) >= 2  # needed to exercise document isolation
    for sample in data["samples"]:
        assert sample["document"] in names and sample["question"].strip()


def test_every_phrase_exists_in_the_corpus():
    data = load_dataset(DEFAULT_DATASET)
    corpus_text = {d["name"]: " ".join(d["pages"]).lower() for d in data["documents"]}
    for sample in data["samples"]:
        for phrase in sample["relevant_phrases"]:
            assert phrase.lower() in corpus_text[sample["document"]], sample["id"]


@pytest.mark.parametrize("bad", [
    {"samples": []},
    {"documents": [], "samples": [{"question": "q", "relevant_phrases": ["x"]}]},
    {"documents": [{"name": "d", "pages": ["t"]}], "samples": [{"document": "zzz", "question": "q", "relevant_phrases": ["x"]}]},
    {"documents": [{"name": "d", "pages": ["t"]}], "samples": [{"document": "d", "question": " ", "relevant_phrases": ["x"]}]},
    {"documents": [{"name": "d", "pages": ["t"]}], "samples": [{"document": "d", "question": "q"}]},
])
def test_invalid_datasets_rejected(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


# -- evaluation flow (real chunking + real ChromaDB, lexical embeddings) --


@pytest.mark.asyncio
async def test_evaluation_flow_end_to_end_offline():
    data = load_dataset(DEFAULT_DATASET)
    embedding = LexicalEmbedding()
    repo = make_repo(f"rag_eval_{uuid.uuid4().hex[:8]}")
    vector = VectorService(embedding_service=embedding, chroma_repo=repo)
    corpus = await ingest_corpus(data, vector)
    rag = RAGService(embedding_service=embedding, chroma_repo=repo, similarity_threshold=0.0,
                     max_context_chars=12000, max_top_k=20)

    report = await evaluate(data, rag, corpus, k=3)

    assert report["samples"] == len(data["samples"]) and report["skipped"] == 0
    metrics = report["metrics"]
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # lexical retrieval on this tiny corpus must find the relevant chunk almost always
    assert metrics["hit"] >= 0.7 and metrics["recall"] >= 0.7
    text = format_report(report)
    assert "Hit@3:" in text and "Precision@3:" in text and "Recall@3:" in text and "Samples: 10" in text


@pytest.mark.asyncio
async def test_ground_truth_resolution_uses_chunk_text():
    data = load_dataset(DEFAULT_DATASET)
    embedding = LexicalEmbedding()
    repo = make_repo(f"rag_eval_{uuid.uuid4().hex[:8]}")
    corpus = await ingest_corpus(data, VectorService(embedding_service=embedding, chroma_repo=repo))
    sample = data["samples"][0]
    ids = relevant_chunk_ids(sample, corpus[sample["document"]]["chunks"])
    assert ids
    other = relevant_chunk_ids(sample, corpus["databases_basics"]["chunks"])
    assert not other  # phrases of one document never match another document's chunks
