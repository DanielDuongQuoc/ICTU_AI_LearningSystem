"""Phase 9 — PerformanceTimer. No exact-millisecond assertions (would be flaky)."""

import pytest

from app.utils.performance import PerformanceTimer


def test_timer_measures_non_negative_elapsed():
    timer = PerformanceTimer()
    timer.start()
    timer.stop()
    assert timer.elapsed_ms() >= 0


def test_timer_context_manager():
    with PerformanceTimer() as timer:
        sum(range(1000))
    assert timer.elapsed_ms() >= 0


def test_timer_elapsed_while_running_is_non_negative():
    timer = PerformanceTimer()
    timer.start()
    assert timer.elapsed_ms() >= 0


def test_measure_records_named_laps():
    timer = PerformanceTimer()
    with timer.measure("rag"):
        pass
    with timer.measure("llm"):
        pass
    assert set(timer.laps) == {"rag", "llm"}
    assert all(v >= 0 for v in timer.laps.values())


def test_measure_records_lap_even_on_exception():
    timer = PerformanceTimer()
    with pytest.raises(ValueError):
        with timer.measure("boom"):
            raise ValueError("x")
    assert "boom" in timer.laps


def test_stop_before_start_raises():
    with pytest.raises(Exception):
        PerformanceTimer().stop()


# ── Streaming metrics (Phase 9) ────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_logs_duration_and_token_metrics(caplog):
    import logging

    from tests.test_evaluation import DOC_A, default_hits, make_llm, make_stack
    from tests.test_evaluation_stream import EXPECTED, QUESTION, STUDENT, good_stream, set_stream

    llm = make_llm()
    set_stream(llm, good_stream())
    svc, _ = make_stack(default_hits(), llm)
    with caplog.at_level(logging.INFO):
        _ = [f async for f in svc.evaluate_answer_stream(DOC_A, QUESTION, STUDENT, EXPECTED, 10)]
    line = next(r.message for r in caplog.records if r.message.startswith("Evaluation stream |"))
    for key in ("first_token_latency_ms=", "last_token_ms=", "stream_duration_ms=",
                "token_count=", "total_latency_ms="):
        assert key in line
    assert "stream_duration_ms=null" not in line
    assert STUDENT not in line
