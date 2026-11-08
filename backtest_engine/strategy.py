"""Strategies: see one closed bar at a time and may emit a SignalEvent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from itertools import islice

from .events import Bar, SignalEvent


class Strategy(ABC):
    def prepare(self, df) -> None:
        """Called once with all the bars before the loop starts (signal strategies
        precompute here). Default: nothing."""

    @abstractmethod
    def on_bar(self, bar: Bar) -> SignalEvent | None:
        """Called once per closed bar. Return a signal to change position, else None."""


class MovingAverageCrossover(Strategy):
    """Long when the fast SMA of closes is above the slow SMA, short when below.

    Only acts on a cross, so the first trade waits for the first cross after both
    averages exist. long_only=True goes flat instead of short.
    """

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = False):
        if not 0 < fast < slow:
            raise ValueError(f"Need 0 < fast < slow, got fast={fast}, slow={slow}")
        self.fast, self.slow, self.long_only = fast, slow, long_only
        self._closes: deque[float] = deque(maxlen=slow)
        self._state: int | None = None  # +1 fast above slow, -1 below
        self.fast_ma = self.slow_ma = float("nan")

    def on_bar(self, bar: Bar) -> SignalEvent | None:
        self._closes.append(bar.close)
        if len(self._closes) < self.slow:
            return None

        self.slow_ma = sum(self._closes) / self.slow
        self.fast_ma = sum(islice(self._closes, self.slow - self.fast, None)) / self.fast
        if self.fast_ma == self.slow_ma:
            return None  # touching isn't crossing
        state = 1 if self.fast_ma > self.slow_ma else -1

        previous, self._state = self._state, state
        if previous is None or state == previous:
            return None
        target = max(state, 0) if self.long_only else state
        return SignalEvent(bar.time, target)
