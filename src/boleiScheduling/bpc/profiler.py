from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class PerformanceProfiler:
    """Lightweight cumulative profiler used by the BPC implementation.

    The profiler deliberately records only coarse-grained sections.  It is
    safe to keep enabled during normal runs and avoids per-label wall-clock
    calls inside the pricing hot loop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def timeBlock(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] += perf_counter() - start

    def addTime(self, name, value):
        self.times[name] += float(value)

    def increment(self, name, amount=1):
        self.counts[name] += int(amount)

    def snapshot(self):
        return {
            'times': dict(self.times),
            'counts': dict(self.counts),
        }

    def time(self, name):
        return float(self.times.get(name, 0.0))

    def count(self, name):
        return int(self.counts.get(name, 0))
