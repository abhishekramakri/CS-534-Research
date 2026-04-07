"""
profiler.py
Lightweight timing utilities for measuring per-module pipeline latency.
Every module in the pipeline is timed through this so we have a baseline
before offloading experiments.
"""

import time
from collections import OrderedDict


class Profiler:
    """Collects named timing measurements across pipeline stages."""

    def __init__(self):
        self._timings: OrderedDict[str, float] = OrderedDict()
        self._starts: dict[str, float] = {}

    def start(self, name: str):
        """Begin timing a named stage."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record elapsed ms. Returns elapsed ms."""
        elapsed_ms = (time.perf_counter() - self._starts.pop(name)) * 1000
        self._timings[name] = elapsed_ms
        return elapsed_ms

    def timer(self, name: str):
        """Context manager for timing a block."""
        return _TimerCtx(self, name)

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

    @property
    def total_ms(self) -> float:
        return sum(self._timings.values())

    def summary(self) -> str:
        lines = [
            "",
            "=" * 52,
            "  Pipeline Latency Breakdown",
            "=" * 52,
        ]
        for name, ms in self._timings.items():
            bar = "#" * int(ms / max(self._timings.values()) * 20)
            lines.append(f"  {name:28s} {ms:8.1f} ms  {bar}")
        lines.append("-" * 52)
        lines.append(f"  {'TOTAL':28s} {self.total_ms:8.1f} ms")
        lines.append("=" * 52)
        return "\n".join(lines)


class _TimerCtx:
    def __init__(self, profiler: Profiler, name: str):
        self._p = profiler
        self._n = name

    def __enter__(self):
        self._p.start(self._n)
        return self

    def __exit__(self, *args):
        self._p.stop(self._n)
