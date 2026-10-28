"""Phase 17 - Answer-evaluation (grading) quality evaluation.

Every dataset case is sent through the real ``EvaluationService`` (validation, retrieval, prompt,
LLM-output validation, backend percentage + sources).  Results are compared with a rubric band
(``expected_score_min..max`` in points of ``max_score``) written BEFORE any run.

Metrics: MAE / RMSE against the band midpoint (normalised to a 0..1 fraction of ``max_score`` so
cases with different ``max_score`` are comparable), mean distance outside the band, within-band
rate, ``correct``-label accuracy (only cases with a non-null ``expected_correct``), over-scoring of
wrong answers, under-scoring of paraphrases, rank-order consistency inside ``consistency_group``s,
range / percentage invariants, and deterministic error-code cases (empty answer, too long,
insufficient context) which must stop BEFORE the LLM is called.

Tolerance: the band IS the tolerance.  Bands are 2-3 points wide on a 10-point scale because two
careful human graders of short answers routinely differ by that much; the bands were chosen
a-priori by the dataset author and have not been calibrated against human graders
(HUMAN REVIEW NOT PERFORMED).  They are NOT an academic standard.

``--mode offline`` makes the "LLM" return the band midpoint (and the expected label), so the run
is a HARNESS SELF-CHECK of the pipeline and metrics; it says nothing about model grading quality.
``--mode ollama`` uses the real model.

Usage::

    python scripts/evaluate_answer_evaluation.py [--mode offline|ollama] [--out-dir DIR]
        [--min within_band_rate=0.7 ...] [--validate-only]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_common as ec  # noqa: E402

DEFAULT_DATASET = ec.DATASET_DIR / "answer_evaluation_eval.jsonl"
WRONG_PREFIXES = ("wrong", "contradictory", "irrelevant", "prompt_injection")  # expected-incorrect answers
PARAPHRASE_PREFIX = "paraphrase"
PERCENT_TOLERANCE = 0.01


def band_midpoint(case: dict[str, Any]) -> float:
    return (case["expected_score_min"] + case["expected_score_max"]) / 2


def distance_outside_band(score: float, case: dict[str, Any]) -> float:
    if score < case["expected_score_min"]:
        return case["expected_score_min"] - score
    if score > case["expected_score_max"]:
        return score - case["expected_score_max"]
    return 0.0


def consistency_violations(rows: list[dict[str, Any]]) -> list[str]:
    """Within a group a lower ``rank`` (better answer) must not score lower than a higher rank."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("group") and r.get("normalized") is not None:
            groups.setdefault(r["group"], []).append(r)
    out: list[str] = []
    for name, members in groups.items():
        for a in members:
            for b in members:
                if a["rank"] < b["rank"] and a["normalized"] < b["normalized"]:
                    out.append(f"{name}: {a['case_id']}(rank {a['rank']}) scored below {b['case_id']}(rank {b['rank']})")
    return out


def _expected_llm_payload(case: dict[str, Any]) -> dict[str, Any]:
    score = float(band_midpoint(case))
    correct = case["expected_correct"] if case["expected_correct"] is not None else score >= case["max_score"] / 2
    return {
        "score": score, "correct": bool(correct), "feedback": "Scripted harness feedback.",
        "strengths": [], "missing_points": [], "errors": [],
    }


async def evaluate(cases: list[dict[str, Any]], stack: Any, corpus: dict[str, Any], offline: bool) -> dict[str, Any]:
    from app.core.exceptions import AppError

    index = ec.corpus_index(corpus)
    rows: list[dict[str, Any]] = []
    for case in cases:
        real_id = stack.doc_map[case["document_id"]]
        student = case["student_answer"]
        if case.get("pad_student_answer_to_chars"):
            student = (student + " ") * (case["pad_student_answer_to_chars"] // (len(student) + 1) + 1)
            student = student[: case["pad_student_answer_to_chars"]]
        if offline:
            stack.llm.eval_payload = _expected_llm_payload(case) if not case.get("expected_error_code") else dict(stack.llm.eval_payload)
        calls_before = len(stack.llm.calls) if offline else None
        row: dict[str, Any] = {
            "case_id": case["case_id"], "category": case["category"], "group": case.get("consistency_group"),
            "rank": case.get("rank"), "max_score": case["max_score"],
            "expected_band": None if case.get("expected_error_code") else [case["expected_score_min"], case["expected_score_max"]],
            "expected_correct": case["expected_correct"],
        }
        try:
            res = await stack.evaluator.evaluate_answer(
                document_id=real_id, question=case["question"], student_answer=student,
                expected_answer=case["expected_answer"], max_score=case["max_score"],
            )
        except AppError as exc:
            row["error_code"] = str(exc.code)
            row["expected_error_code"] = case.get("expected_error_code")
            if offline:
                row["llm_called"] = len(stack.llm.calls) > calls_before
            rows.append(row)
            continue
        score = float(res.score)
        valid_ids = set(index[case["document_id"]])
        if case.get("expected_error_code"):  # should have failed; no band to compare with
            row.update({"score": score, "expected_error_code": case["expected_error_code"],
                        "sources_valid": all(s.chunk_id in valid_ids for s in res.sources)})
            rows.append(row)
            continue
        row.update({
            "score": score, "correct": res.correct, "percentage": res.percentage,
            "normalized": score / case["max_score"],
            "in_range": 0 <= score <= case["max_score"] and math.isfinite(score),
            "percentage_backend_ok": abs(res.percentage - round(score / case["max_score"] * 100, 2)) <= PERCENT_TOLERANCE,
            "sources_valid": all(s.chunk_id in valid_ids for s in res.sources),
            "within_band": case["expected_score_min"] <= score <= case["expected_score_max"],
            "distance_outside_band": distance_outside_band(score, case),
            "midpoint_error_norm": abs(score - band_midpoint(case)) / case["max_score"],
            "expected_error_code": case.get("expected_error_code"),
        })
        rows.append(row)

    all_scored = [r for r in rows if "score" in r]
    scored = [r for r in all_scored if not r.get("expected_error_code")]
    expected_errors = [r for r in rows if r.get("expected_error_code")]
    unexpected_errors = [r for r in rows if "score" not in r and not r.get("expected_error_code")]
    missed_errors = [r for r in all_scored if r.get("expected_error_code")]
    error_ok = [r for r in expected_errors if r.get("error_code") == r["expected_error_code"]]
    labelled = [r for r in scored if r["expected_correct"] is not None]
    wrong_rows = [r for r in scored if r["category"].startswith(WRONG_PREFIXES)]
    para_rows = [r for r in scored if r["category"].startswith(PARAPHRASE_PREFIX)]
    mid_errs = [r["midpoint_error_norm"] for r in scored]
    violations = consistency_violations(scored)
    label_score_incons = [
        r["case_id"] for r in scored
        if (r["correct"] and r["percentage"] < 50) or ((not r["correct"]) and r["percentage"] >= 70)
    ]
    metrics = {
        "cases": len(rows), "graded": len(scored),
        "mae_normalized_vs_band_midpoint": ec.mean(mid_errs),
        "rmse_normalized_vs_band_midpoint": math.sqrt(sum(e * e for e in mid_errs) / len(mid_errs)) if mid_errs else None,
        "mean_distance_outside_band_points": ec.mean([r["distance_outside_band"] for r in scored]),
        "within_band_rate": ec.mean([1.0 if r["within_band"] else 0.0 for r in scored]),
        "correct_label_accuracy": ec.mean([1.0 if r["correct"] == r["expected_correct"] else 0.0 for r in labelled]),
        "correct_label_cases": len(labelled),
        "over_scoring_rate_wrong_answers": ec.mean([1.0 if r["score"] > r["expected_band"][1] else 0.0 for r in wrong_rows]),
        "under_scoring_rate_paraphrases": ec.mean([1.0 if r["score"] < r["expected_band"][0] else 0.0 for r in para_rows]),
        "score_in_range_rate": ec.mean([1.0 if r["in_range"] else 0.0 for r in scored]),
        "backend_percentage_ok_rate": ec.mean([1.0 if r["percentage_backend_ok"] else 0.0 for r in scored]),
        "rank_consistency_violations": len(violations),
    }
    errors = {
        "error_code_cases": len(expected_errors), "error_code_correct": len(error_ok),
        "error_cases_that_returned_a_score": len(missed_errors),
        "unexpected_errors": [{"case_id": r["case_id"], "code": r["error_code"]} for r in unexpected_errors],
        "llm_called_on_error_case": sum(1 for r in expected_errors if r.get("llm_called")),
    }
    invalid_sources = sum(1 for r in all_scored if not r["sources_valid"])
    findings = [
        "Bands were written by the dataset author before any run and are not calibrated against human "
        "graders. HUMAN REVIEW NOT PERFORMED.",
        "The `correct` flag is produced by the LLM; the backend only rejects correct+score 0 and "
        "incorrect+full score. No score threshold defines `correct`, so label accuracy and score "
        "bands can legitimately disagree. This contract was NOT changed.",
    ]
    if unexpected_errors or missed_errors or len(error_ok) != len(expected_errors):
        findings.append(
            "INSUFFICIENT_CONTEXT outcomes depend on the embedding model and SIMILARITY_THRESHOLD; in "
            "offline mode they come from the lexical test embedder and are NOT representative of the "
            "production embedding model (nomic-embed-text). Re-run with --mode ollama to verify."
        )
    if label_score_incons:
        findings.append(
            "`correct` disagreed with percentage (correct and <50%, or incorrect and >=70%) in: "
            + ", ".join(label_score_incons)
        )
    if violations:
        findings += [f"Rank-order violation - {v}" for v in violations]
    if offline:
        findings.insert(0, "OFFLINE harness self-check: the 'LLM' returns the band midpoint, so score metrics "
                           "validate the pipeline/metrics, NOT grading quality. Error-code and insufficient-"
                           "context cases exercise real service behaviour.")
    return {
        "sections": {"Grading quality": metrics, "Deterministic error / abstention cases": errors,
                     "Security invariants": {"sources_outside_document": invalid_sources}},
        "per_case": rows, "findings": findings,
        "metrics_flat": {**{k: v for k, v in metrics.items()}, "unexpected_errors": len(unexpected_errors),
                         "error_code_failures": len(expected_errors) - len(error_ok) + len(missed_errors)},
        "invariants": {"source_integrity_violations": invalid_sources, "document_isolation_violations": invalid_sources},
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from app.core.prompts import EVALUATION_PROMPT_VERSION

    cases = ec.load_jsonl(args.dataset)
    corpus = ec.load_corpus()
    ec.validate_eval_cases(cases, corpus)
    if args.validate_only:
        return {"validated": len(cases)}
    stack = ec.EvalStack(args.mode, args.threshold)
    try:
        await stack.index_corpus(corpus)
        report = await evaluate(cases, stack, corpus, offline=args.mode == "offline")
        if not any("score" in r for r in report["per_case"]):
            raise RuntimeError("no case produced a score; evaluator did not run")
    finally:
        await stack.close()
    report["metadata"] = ec.run_metadata(
        suite="answer_eval", mode=args.mode, dataset_version=corpus["dataset_version"],
        dataset_files={"answer_evaluation_eval.jsonl": Path(args.dataset), "corpus.json": ec.DATASET_DIR / "corpus.json"},
        top_k=stack.settings.top_k, similarity_threshold=stack.threshold, model_name=stack.model_name,
        embedding_model=stack.embedding_model, prompt_versions={"evaluation": EVALUATION_PROMPT_VERSION},
    )
    report["metadata"]["label_sources"] = sorted({c["label_source"] for c in cases})
    failures = []
    for key, floor in ec.parse_thresholds(args.min).items():
        value = report["metrics_flat"].get(key)
        if value is None or value < floor:
            failures.append(f"{key}={value} is below the requested minimum {floor}")
    report["threshold_failures"] = failures
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate answer-evaluation (grading) quality.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--mode", choices=["offline", "ollama"], default="offline")
    p.add_argument("--out-dir", default=str(ec.DEFAULT_REPORT_DIR))
    p.add_argument("--threshold", type=float, default=None, help="Override SIMILARITY_THRESHOLD (recorded)")
    p.add_argument("--min", action="append", metavar="METRIC=VALUE")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    try:
        report = asyncio.run(run(args))
        if args.validate_only:
            print(f"Dataset OK: {report['validated']} answer-evaluation cases validated.")
            return ec.EXIT_OK
        paths = ec.write_reports(report, Path(args.out_dir), "answer_eval", "Answer Evaluation Quality Report")
    except ec.DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return ec.EXIT_DATASET
    except Exception as exc:
        print(f"Answer-evaluation run failed: {type(exc).__name__}: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return ec.EXIT_FAILURE
    meta = report["metadata"]
    print(f"Answer evaluation ({meta['mode']} mode, dataset {meta['dataset_version']}, run {meta['evaluation_run_id']})")
    for key, val in report["metrics_flat"].items():
        print(f"  {key}: {round(val, 4) if isinstance(val, float) else val}")
    print("Reports:", ", ".join(str(x) for x in paths))
    for f in report["threshold_failures"]:
        print(f"THRESHOLD NOT MET: {f}", file=sys.stderr)
    return ec.EXIT_THRESHOLD if report["threshold_failures"] else ec.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())


