"""Small timing helper so ``time.perf_counter()`` is not copied all over the code base."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional


class PerformanceTimer:
    """Stopwatch with named laps.

    >>> timer = PerformanceTimer().start()
    >>> with timer.measure("rag"):
    ...     pass
    >>> timer.laps["rag"] >= 0
    True
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._stop: Optional[float] = None
        self.laps: dict[str, float] = {}

    def start(self) -> "PerformanceTimer":
        self._start = time.perf_counter()
        self._stop = None
        return self

    def stop(self) -> float:
        """Stop the timer and return the elapsed time in milliseconds."""
        if self._start is None:
            raise RuntimeError("PerformanceTimer.stop() called before start().")
        self._stop = time.perf_counter()
        return self.elapsed_ms()

    def elapsed_ms(self) -> float:
        """Milliseconds since ``start()`` (frozen after ``stop()``).  0.0 if never started."""
        if self._start is None:
            return 0.0
        end = self._stop if self._stop is not None else time.perf_counter()
        return max(0.0, (end - self._start) * 1000)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Record the duration (ms) of the ``with`` block in ``laps[name]``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.laps[name] = max(0.0, (time.perf_counter() - t0) * 1000)

    def __enter__(self) -> "PerformanceTimer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
