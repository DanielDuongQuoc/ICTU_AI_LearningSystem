"""Phase 20 - the SLO report never invents a PASS."""

from __future__ import annotations

from scripts.check_reliability import evaluate_slos, main, parse_metrics, quantile_upper_bound, render

SRC = "unit-test"


def _metrics(n_ok=100, n_err=0, fast=100, slow=0) -> str:
    lines = [f'module3_http_requests_total{{method="POST",route="/api/rag/retrieve",status="200"}} {n_ok}']
    if n_err:
        lines.append(f'module3_http_requests_total{{method="POST",route="/api/rag/retrieve",status="503"}} {n_err}')
    r = "/api/rag/retrieve"
    total = fast + slow
    for le, cum in (("0.05", fast), ("0.1", fast), ("0.25", fast), ("0.5", fast), ("1.0", fast), ("2.5", total),
                    ("5.0", total), ("10.0", total), ("30.0", total), ("60.0", total), ("120.0", total)):
        lines.append(f'module3_http_request_duration_seconds_bucket{{method="POST",route="{r}",le="{cum and le}"}} {cum}')
    lines.append(f'module3_http_request_duration_seconds_bucket{{method="POST",route="{r}",le="+Inf"}} {total}')
    return "\n".join(lines) + "\n"


def _row(rows, name):
    return next(r for r in rows if r.slo.startswith(name))


def test_no_data_is_inconclusive_never_pass():
    rows = evaluate_slos(parse_metrics(""), SRC)
    assert rows and all(r.status.startswith("INCONCLUSIVE") for r in rows)


def test_too_few_samples_is_inconclusive():
    rows = evaluate_slos(parse_metrics(_metrics(n_ok=5, fast=5)), SRC)
    assert _row(rows, "Availability").status == "INCONCLUSIVE"
    assert _row(rows, "Retrieve latency").status == "INCONCLUSIVE"


def test_enough_good_data_passes_and_bad_data_fails():
    good = evaluate_slos(parse_metrics(_metrics()), SRC)
    assert _row(good, "Availability").status == "PASS" and _row(good, "Retrieve latency").status == "PASS"
    bad = evaluate_slos(parse_metrics(_metrics(n_ok=80, n_err=20, fast=50, slow=50)), SRC)
    assert _row(bad, "Availability").status == "FAIL" and _row(bad, "Retrieve latency").status == "FAIL"


def test_llm_slos_inconclusive_in_harness_mode():
    rows = evaluate_slos(parse_metrics(_metrics()), SRC, harness=True)
    for name in ("Question generation", "Evaluation latency", "LLM success"):
        assert _row(rows, name).status == "INCONCLUSIVE"


def test_regression_needs_baseline():
    rows = evaluate_slos(parse_metrics(_metrics()), SRC)
    assert _row(rows, "Retrieve p95 vs baseline").status == "INCONCLUSIVE"


def test_quantile_helper():
    assert quantile_upper_bound(0, [], 0.95) is None
    assert quantile_upper_bound(100, [(0.1, 50), (1.0, 96), (float("inf"), 100)], 0.95) == 1.0


def test_render_has_required_columns():
    out = render(evaluate_slos(parse_metrics(""), SRC))
    for col in ("SLO", "Target", "Observed", "Status", "Data source"):
        assert col in out.splitlines()[0]


def test_cli_exit_codes(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text(_metrics(), encoding="utf-8")
    assert main(["--metrics-file", str(f)]) == 0
    assert main(["--metrics-file", str(f), "--strict"]) == 1          # some SLOs inconclusive -> strict fails
    f.write_text(_metrics(n_ok=80, n_err=20), encoding="utf-8")
    assert main(["--metrics-file", str(f)]) == 1
