"""Deterministic retrieval-quality metrics (no LLM involved).

Definitions (``retrieved`` is the ranked list of chunk ids, ``relevant`` the ground truth):

* **Hit@K**       - 1 if at least one relevant chunk is in the top K, else 0.
* **Precision@K** - (relevant chunks in the top K) / K.  The denominator is always K, so a
  retriever that returns fewer than K chunks is penalised.
* **Recall@K**    - (relevant chunks in the top K) / (total relevant chunks).

Duplicate ids in ``retrieved`` are counted once.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _top_k_unique(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be > 0")
    seen: set[str] = set()
    unique: list[str] = []
    for chunk_id in retrieved:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique[:k]


def _relevant_in_top_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> int:
    rel = set(relevant)
    return sum(1 for c in _top_k_unique(retrieved, k) if c in rel)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if _relevant_in_top_k(retrieved, relevant, k) > 0 else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return _relevant_in_top_k(retrieved, relevant, k) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    return _relevant_in_top_k(retrieved, rel, k) / len(rel)


def score_sample(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> dict[str, float]:
    rel = list(relevant)
    return {
        "hit": hit_at_k(retrieved, rel, k),
        "precision": precision_at_k(retrieved, rel, k),
        "recall": recall_at_k(retrieved, rel, k),
    }


def average(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric over all samples (empty input -> zeros)."""
    if not scores:
        return {"hit": 0.0, "precision": 0.0, "recall": 0.0}
    n = len(scores)
    return {key: sum(s[key] for s in scores) / n for key in ("hit", "precision", "recall")}


# -- Phase 17 additions (rank-aware metrics; the functions above are unchanged) ----------


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None) -> float:
    """1 / rank of the first relevant chunk (rank starts at 1, duplicates collapsed); 0 if none."""
    rel = set(relevant)
    unique = _top_k_unique(retrieved, k if k is not None else max(len(retrieved), 1))
    for rank_, chunk_id in enumerate(unique, start=1):
        if chunk_id in rel:
            return 1.0 / rank_
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@K: DCG of the ranking divided by the ideal DCG for len(relevant) hits."""
    import math

    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    dcg = sum(
        1.0 / math.log2(i + 2) for i, c in enumerate(_top_k_unique(retrieved, k)) if c in rel
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0
