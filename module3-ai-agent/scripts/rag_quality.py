"""Phase 17 RAG quality suite (JSONL datasets). Used by ``scripts/evaluate_rag.py``.

Retrieval metrics (Hit/Precision/Recall@K, MRR, nDCG@K), abstention / insufficient-context
detection, document-isolation and source-integrity invariants, and (ollama mode only) lexical
answer-quality proxies.  Not part of the application runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from app.utils.rag_metrics import ndcg_at_k, reciprocal_rank, score_sample
from scripts import eval_common as ec


async def evaluate_cases(
    cases: list[dict[str, Any]], stack: Any, k: int, corpus: dict[str, Any], with_answers: bool = False
) -> dict[str, Any]:
    """Score every case against the fixture corpus indexed in ``stack``.

    Retrieval metrics use only cases with ground-truth chunks; abstention cases measure whether the
    pipeline stops before generation.  A chunk from another document is an isolation violation and
    a source that does not match retrieved metadata is a source-integrity violation; both are
    counted separately and never averaged away.
    """
    index = ec.corpus_index(corpus)
    rows: list[dict[str, Any]] = []
    isolation_violations = source_violations = 0
    for case in cases:
        real_id = stack.doc_map[case["document_id"]]
        t0 = time.perf_counter()
        result = await stack.rag.retrieve(document_id=real_id, question=case["query"], top_k=k)
        latency_ms = (time.perf_counter() - t0) * 1000
        chunks = result["chunks"]
        retrieved = [c["chunk_id"] for c in chunks]
        foreign = [
            c for c in chunks
            if c.get("document_id") != real_id or c["chunk_id"] not in index[case["document_id"]]
        ]
        isolation_violations += len(foreign)
        by_id = {c["chunk_id"]: c for c in chunks}
        bad_sources = [
            s for s in result.get("sources", [])
            if s.get("chunk_id") not in by_id or s.get("page") != by_id[s["chunk_id"]].get("page")
        ]
        source_violations += len(bad_sources)
        abstained = bool(result["insufficient_context"] or not chunks)
        row: dict[str, Any] = {
            "case_id": case["case_id"], "category": case["category"],
            "expect_abstain": case["expect_abstain"], "retrieved": retrieved,
            "retrieval_abstained": abstained, "latency_ms": round(latency_ms, 2),
            "isolation_violations": len(foreign), "source_violations": len(bad_sources),
        }
        if not case["expect_abstain"]:
            relevant = set(case["relevant_chunk_ids"])
            row.update(score_sample(retrieved, relevant, k))
            row["mrr"] = reciprocal_rank(retrieved, relevant, k)
            row["ndcg"] = ndcg_at_k(retrieved, relevant, k)
            row["incorrect_abstention"] = abstained
        else:
            row["correct_abstention"] = abstained
        if with_answers:
            answer = await stack.rag.answer_question(document_id=real_id, question=case["query"], top_k=k)
            text = answer["answer"]
            ctx = " ".join(c["text"] for c in chunks)
            answered = not (answer["insufficient_context"] or ec.looks_like_abstention(text))
            row["final_abstained"] = not answered
            if case["expect_abstain"]:
                row["unsupported_answer"] = answered
            else:
                kp = case["expected_key_points"]
                row["completeness"] = (
                    ec.mean([1.0 if ec.token_coverage(p, text) >= 0.6 else 0.0 for p in kp]) if kp else None
                )
                row["faithfulness_proxy"] = ec.token_coverage(text, ctx) if answered else None
                row["answer_relevance_proxy"] = (
                    ec.token_coverage(case["query"], text + " " + ctx) if answered else None
                )
            retrieved_set = set(retrieved)
            row["answer_source_violations"] = sum(
                1 for s in answer["sources"] if s["chunk_id"] not in retrieved_set
            )
            source_violations += row["answer_source_violations"]
        rows.append(row)

    answerable = [r for r in rows if not r["expect_abstain"]]
    abstain = [r for r in rows if r["expect_abstain"]]
    metrics: dict[str, Any] = {
        f"hit@{k}": ec.mean([r["hit"] for r in answerable]),
        f"precision@{k}": ec.mean([r["precision"] for r in answerable]),
        f"recall@{k}": ec.mean([r["recall"] for r in answerable]),
        "mrr": ec.mean([r["mrr"] for r in answerable]),
        f"ndcg@{k}": ec.mean([r["ndcg"] for r in answerable]),
        "empty_retrieval_rate_answerable": ec.mean([1.0 if not r["retrieved"] else 0.0 for r in answerable]),
        "latency_ms_mean": ec.mean([r["latency_ms"] for r in rows]),
        "latency_ms_max": max(r["latency_ms"] for r in rows),
    }
    abstention = {
        "abstain_cases": len(abstain),
        "insufficient_context_detection_rate": ec.mean([1.0 if r["correct_abstention"] else 0.0 for r in abstain]),
        "unsupported_context_pass_rate": ec.mean([0.0 if r["correct_abstention"] else 1.0 for r in abstain]),
        "incorrect_abstention_rate": ec.mean([1.0 if r["incorrect_abstention"] else 0.0 for r in answerable]),
    }
    by_category: dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in answerable if r["category"] == cat]
        by_category[cat] = {
            "n": sum(1 for r in rows if r["category"] == cat),
            f"hit@{k}": ec.mean([r["hit"] for r in sub]),
            "mrr": ec.mean([r["mrr"] for r in sub]),
        }
    sections: dict[str, Any] = {
        f"Retrieval (K={k}, answerable cases n={len(answerable)})": metrics,
        f"Abstention / insufficient context (n={len(abstain)} abstain cases)": abstention,
        "Security invariants": {
            "document_isolation_violations": isolation_violations,
            "source_integrity_violations": source_violations,
        },
        "Per category": by_category,
    }
    findings: list[str] = []
    if with_answers:
        scored = [r for r in answerable if r.get("completeness") is not None]
        answered_rows = [r for r in answerable if r.get("faithfulness_proxy") is not None]
        sections["Answer quality (lexical proxies; NOT ground truth)"] = {
            "completeness": ec.mean([r["completeness"] for r in scored]),
            "faithfulness_proxy": ec.mean([r["faithfulness_proxy"] for r in answered_rows]),
            "answer_relevance_proxy": ec.mean([r["answer_relevance_proxy"] for r in answered_rows]),
            "correct_abstention_rate": ec.mean([0.0 if r["unsupported_answer"] else 1.0 for r in abstain]),
            "unsupported_answer_rate": ec.mean([1.0 if r["unsupported_answer"] else 0.0 for r in abstain]),
            "incorrect_final_abstention_rate": ec.mean([1.0 if r["final_abstained"] else 0.0 for r in answerable]),
        }
        findings.append(
            "Faithfulness/relevance/completeness are token-overlap proxies, not an LLM judge or human "
            "verdict; they miss paraphrase and negation errors."
        )
    else:
        findings.append("Answer-quality metrics were not computed (needs --answers with --mode ollama).")
    return {
        "k": k, "cases": len(rows), "sections": sections, "per_case": rows, "findings": findings,
        "invariants": {
            "document_isolation_violations": isolation_violations,
            "source_integrity_violations": source_violations,
        },
        "metrics_flat": {**metrics, **abstention},
    }


async def run_jsonl_suite(args: argparse.Namespace) -> dict[str, Any]:
    from app.core.prompts import PROMPT_VERSION

    dataset_path = Path(args.dataset)
    cases = ec.load_jsonl(dataset_path)
    corpus = ec.load_corpus()
    ec.validate_rag_cases(cases, corpus)
    if args.validate_only:
        return {"validated": len(cases)}
    if args.answers and args.mode != "ollama":
        raise ec.DatasetError("--answers needs --mode ollama (offline mode has no real LLM).")
    stack = ec.EvalStack(args.mode, getattr(args, "threshold", None))
    k = args.k or stack.settings.top_k
    try:
        await stack.index_corpus(corpus)
        report = await evaluate_cases(cases, stack, k, corpus, with_answers=args.answers)
    finally:
        await stack.close()
    report["metadata"] = ec.run_metadata(
        suite="rag", mode=args.mode, dataset_version=corpus["dataset_version"],
        dataset_files={"rag_eval.jsonl": dataset_path, "corpus.json": ec.DATASET_DIR / "corpus.json"},
        top_k=k, similarity_threshold=stack.threshold, model_name=stack.model_name,
        embedding_model=stack.embedding_model, prompt_versions={"rag": PROMPT_VERSION},
    )
    report["metadata"]["label_sources"] = sorted({c["label_source"] for c in cases})
    failures = []
    for key, floor in ec.parse_thresholds(args.min).items():
        value = report["metrics_flat"].get(key)
        if value is None or value < floor:
            failures.append(f"{key}={value} is below the requested minimum {floor}")
    report["threshold_failures"] = failures
    return report


def main_jsonl(args: argparse.Namespace) -> int:
    try:
        report = asyncio.run(run_jsonl_suite(args))
        if args.validate_only:
            print(f"Dataset OK: {report['validated']} RAG cases validated.")
            return ec.EXIT_OK
        paths = ec.write_reports(report, Path(args.out_dir), "rag_eval", "RAG Evaluation Report")
    except ec.DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return ec.EXIT_DATASET
    except Exception as exc:  # evaluation failure, distinct from low quality
        print(f"RAG evaluation failed: {type(exc).__name__}: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return ec.EXIT_FAILURE
    meta = report["metadata"]
    print(
        f"RAG evaluation ({meta['mode']} mode, K={report['k']}, dataset {meta['dataset_version']}, "
        f"run {meta['evaluation_run_id']})"
    )
    for name, values in list(report["sections"].items())[:3]:
        print(f"  {name}")
        for key, val in values.items():
            print(f"    {key}: {round(val, 4) if isinstance(val, float) else val}")
    print("Reports:", ", ".join(str(p) for p in paths))
    for f in report["threshold_failures"]:
        print(f"THRESHOLD NOT MET: {f}", file=sys.stderr)
    return ec.EXIT_THRESHOLD if report["threshold_failures"] else ec.EXIT_OK

