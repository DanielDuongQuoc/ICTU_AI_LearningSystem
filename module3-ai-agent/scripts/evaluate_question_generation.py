"""Phase 17 - Question Generation quality evaluation.

For every dataset case the real ``QuestionGenerationService`` is run (retrieval, prompt, validation,
dedup, backend source attribution) and each generated question is checked against the fixture
passages of its document:

* structure   - type/difficulty equal the request; multiple_choice has exactly 4 unique options and
                a correct_answer that is one of them (the service enforces this; we re-check it);
* answerability / groundedness - lexical coverage of the question / expected answer by the document
                text (a PROXY: it cannot detect a fluent but wrong answer);
* key points  - every ``required_key_points`` term appears in the question, answer or options;
* duplication - near-duplicate questions among everything generated for the same document;
* source integrity - every ``source_chunks`` id exists in the case's own document;
* difficulty  - only label consistency is automatic; semantic difficulty needs human review.

``--mode offline`` replays the dataset's *reference* questions as the LLM output.  That is a
harness self-check (it proves the checks and the service validation work end-to-end); it says
nothing about model quality.  ``--mode ollama`` uses the real model.

Usage::

    python scripts/evaluate_question_generation.py [--mode offline|ollama] [--out-dir DIR]
        [--min groundedness_rate=0.8 ...] [--validate-only]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_common as ec  # noqa: E402

DEFAULT_DATASET = ec.DATASET_DIR / "question_generation_eval.jsonl"
ANSWERABLE_MIN = 0.5      # share of question content words found in the document
GROUNDED_MIN = 0.6        # share of expected-answer content words found in the document
BOOLEAN_ANSWERS = {"true", "false"}


def check_question(
    q: dict[str, Any], case: dict[str, Any], passages: dict[str, str]
) -> dict[str, Any]:
    """Deterministic checks for ONE generated question (dict form of ``GeneratedQuestion``)."""
    doc_text = " ".join(passages.values())
    answer = str(q.get("expected_answer", ""))
    is_bool = answer.strip().lower() in BOOLEAN_ANSWERS or q.get("type") == "true_false"
    options = q.get("options") or []
    mcq_ok: Optional[bool] = None
    mcq_supported: Optional[bool] = None
    if q.get("type") == "multiple_choice":
        correct = q.get("correct_answer")
        mcq_ok = len(options) == 4 and len(set(o.strip().lower() for o in options)) == 4 and correct in options
        mcq_supported = bool(correct) and ec.token_coverage(str(correct), doc_text) >= ANSWERABLE_MIN
    searchable = " ".join([q.get("question", ""), answer, *map(str, options)])
    key_points = case["required_key_points"]
    src = list(q.get("source_chunks") or [])
    return {
        "type_matches_request": q.get("type") == case["question_type"],
        "difficulty_matches_request": q.get("difficulty") == case["difficulty"],
        "answerable": ec.token_coverage(q.get("question", ""), doc_text) >= ANSWERABLE_MIN,
        "grounded": (
            ec.token_coverage(q.get("question", ""), doc_text) >= GROUNDED_MIN if is_bool
            else ec.token_coverage(answer, doc_text) >= GROUNDED_MIN
        ),
        "key_point_coverage": sum(1 for p in key_points if ec.token_coverage(p, searchable) >= 1.0) / len(key_points),
        "mcq_structure_ok": mcq_ok,
        "mcq_correct_supported": mcq_supported,
        "source_ids_valid": all(s in passages for s in src),
        "has_sources": bool(src),
        "foreign_source_ids": [s for s in src if s not in passages],
    }


def duplicate_pairs(questions: list[str]) -> int:
    from app.services.question_service import is_duplicate, normalize_text

    seen: list[str] = []
    dups = 0
    for text in questions:
        norm = normalize_text(text)
        if is_duplicate(norm, seen):
            dups += 1
        else:
            seen.append(norm)
    return dups


def _rate(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return ec.mean([1.0 if v else 0.0 for v in vals])


async def evaluate(cases: list[dict[str, Any]], stack: Any, corpus: dict[str, Any], offline: bool) -> dict[str, Any]:
    from app.core.exceptions import AppError

    index = ec.corpus_index(corpus)
    rows: list[dict[str, Any]] = []
    per_doc_questions: dict[str, list[str]] = {}
    errors: list[dict[str, str]] = []
    for case in cases:
        real_id = stack.doc_map[case["document_id"]]
        if offline:
            stack.llm.questions_payload = {"questions": [dict(r) for r in case["reference_questions"]]}
        try:
            result = await stack.questions.generate_questions(
                document_id=real_id, number_of_questions=case["number_of_questions"],
                difficulty=case["difficulty"], question_type=case["question_type"],
            )
        except AppError as exc:
            errors.append({"case_id": case["case_id"], "code": str(exc.code)})
            continue
        for q in result.questions:
            qd = q.model_dump(mode="json")
            row = {"case_id": case["case_id"], "question_type": case["question_type"],
                   "difficulty": case["difficulty"], **check_question(qd, case, index[case["document_id"]])}
            rows.append(row)
            per_doc_questions.setdefault(case["document_id"], []).append(qd["question"])
    total = sum(len(v) for v in per_doc_questions.values())
    dups = sum(duplicate_pairs(v) for v in per_doc_questions.values())
    mcq_rows = [r for r in rows if r["question_type"] == "multiple_choice"]
    foreign = sum(len(r["foreign_source_ids"]) for r in rows)
    metrics = {
        "cases": len(cases), "cases_generated": len({r["case_id"] for r in rows}),
        "generation_errors": len(errors), "questions_checked": len(rows),
        "answerability_rate": _rate(rows, "answerable"),
        "groundedness_rate": _rate(rows, "grounded"),
        "key_point_coverage_mean": ec.mean([r["key_point_coverage"] for r in rows]),
        "duplication_rate": (dups / total) if total else None,
        "type_match_rate": _rate(rows, "type_matches_request"),
        "difficulty_label_match_rate": _rate(rows, "difficulty_matches_request"),
        "mcq_structure_ok_rate": _rate(mcq_rows, "mcq_structure_ok"),
        "mcq_correct_option_supported_rate": _rate(mcq_rows, "mcq_correct_supported"),
        "source_integrity_rate": _rate(rows, "source_ids_valid"),
        "questions_with_sources_rate": _rate(rows, "has_sources"),
    }
    by_type: dict[str, Any] = {}
    for t in sorted({r["question_type"] for r in rows}):
        sub = [r for r in rows if r["question_type"] == t]
        by_type[t] = {"n": len(sub), "grounded": _rate(sub, "grounded"), "answerable": _rate(sub, "answerable")}
    findings = [
        "Answerability/groundedness/expected-answer correctness are lexical-overlap proxies against the "
        "fixture passages, not human or LLM judgements. A fluent but factually wrong answer built from "
        "document words would pass.",
        "Semantic difficulty consistency is NOT measured automatically (only label equality). "
        "HUMAN REVIEW NOT PERFORMED.",
    ]
    if offline:
        findings.insert(0, "OFFLINE harness self-check: the 'LLM' replays the dataset's reference questions, "
                           "so these numbers validate the checker and service validation, not a model.")
    return {
        "sections": {"Question generation": metrics, "By question type": by_type,
                     "Security invariants": {"foreign_source_ids": foreign},
                     "Generation errors": errors or "none"},
        "per_question": rows, "findings": findings, "metrics_flat": metrics,
        "invariants": {"source_integrity_violations": foreign, "document_isolation_violations": foreign},
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from app.core.prompts import QUESTION_GENERATION_PROMPT_VERSION

    cases = ec.load_jsonl(args.dataset)
    corpus = ec.load_corpus()
    ec.validate_qgen_cases(cases, corpus)
    if args.validate_only:
        return {"validated": len(cases)}
    # The lexical test embedder scores much lower than a real embedding model, so the harness
    # self-check uses threshold 0.0 (the generic retrieval query has no lexical overlap with any passage);
    # production uses 0.3. The value is recorded in the report metadata.
    threshold = args.threshold if args.threshold is not None else (0.0 if args.mode == "offline" else None)
    stack = ec.EvalStack(args.mode, threshold)
    try:
        await stack.index_corpus(corpus)
        report = await evaluate(cases, stack, corpus, offline=args.mode == "offline")
        if not report["per_question"]:
            raise RuntimeError(f"no question was generated for any case: {report['sections']['Generation errors']}")
    finally:
        await stack.close()
    report["metadata"] = ec.run_metadata(
        suite="qgen", mode=args.mode, dataset_version=corpus["dataset_version"],
        dataset_files={"question_generation_eval.jsonl": Path(args.dataset), "corpus.json": ec.DATASET_DIR / "corpus.json"},
        top_k=stack.settings.top_k, similarity_threshold=stack.threshold, model_name=stack.model_name,
        embedding_model=stack.embedding_model, prompt_versions={"question_generation": QUESTION_GENERATION_PROMPT_VERSION},
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
    p = argparse.ArgumentParser(description="Evaluate question-generation quality.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--mode", choices=["offline", "ollama"], default="offline")
    p.add_argument("--out-dir", default=str(ec.DEFAULT_REPORT_DIR))
    p.add_argument("--min", action="append", metavar="METRIC=VALUE")
    p.add_argument("--threshold", type=float, default=None, help="Override SIMILARITY_THRESHOLD (recorded)")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    try:
        report = asyncio.run(run(args))
        if args.validate_only:
            print(f"Dataset OK: {report['validated']} question-generation cases validated.")
            return ec.EXIT_OK
        paths = ec.write_reports(report, Path(args.out_dir), "qgen_eval", "Question Generation Evaluation Report")
    except ec.DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return ec.EXIT_DATASET
    except Exception as exc:
        print(f"Question-generation evaluation failed: {type(exc).__name__}: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return ec.EXIT_FAILURE
    meta = report["metadata"]
    print(f"Question generation evaluation ({meta['mode']} mode, dataset {meta['dataset_version']}, run {meta['evaluation_run_id']})")
    for key, val in report["metrics_flat"].items():
        print(f"  {key}: {round(val, 4) if isinstance(val, float) else val}")
    print("Reports:", ", ".join(str(x) for x in paths))
    for f in report["threshold_failures"]:
        print(f"THRESHOLD NOT MET: {f}", file=sys.stderr)
    return ec.EXIT_THRESHOLD if report["threshold_failures"] else ec.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())


