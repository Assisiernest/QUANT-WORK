"""Signal strategies: a function of the price data returns the wanted position on every bar.

    def signals(df, fast=10, slow=30):           # df: open, high, low, close, volume (UTC index)
        f, s = df.close.rolling(fast).mean(), df.close.rolling(slow).mean()
        return np.sign(f - s)                     # +1 long, -1 short, 0 flat (NaN = flat)

SignalStrategy runs such a function once on all the bars, then the event loop trades it
bar by bar: a change in the wanted position on bar t becomes an order at bar t+1's open,
exactly like the built-in event strategies. The function must only use data up to each
bar (rolling windows, shift(1), ...). check_lookahead() tests that by re-running it on
truncated data and comparing.

BUILTINS holds ready-made strategies with their parameters; load_custom() turns Python
source code (pasted or uploaded) into a strategy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .events import Bar, SignalEvent
from .strategy import MovingAverageCrossover, Strategy

SignalFn = Callable[..., pd.Series]


class SignalStrategy(Strategy):
    """Wraps a signal function as an event-loop Strategy."""

    def __init__(self, fn: SignalFn, long_only: bool = False, **params):
        self.fn, self.long_only, self.params = fn, long_only, params
        self._targets: dict[pd.Timestamp, int] = {}
        self._last: int | None = None

    def prepare(self, df: pd.DataFrame) -> None:
        self._targets = clean_signals(self.fn(df, **self.params), df, long_only=self.long_only).to_dict()

    def on_bar(self, bar: Bar) -> SignalEvent | None:
        target = self._targets.get(bar.time, 0)
        previous, self._last = self._last, target
        # First bar: only act if the function already wants a position.
        if target == previous or (previous is None and target == 0):
            return None
        return SignalEvent(bar.time, target)


def clean_signals(raw, df: pd.DataFrame, long_only: bool = False) -> pd.Series:
    """Coerce a signal function's output to a Series of -1/0/+1 on df's index."""
    if not isinstance(raw, pd.Series):
        raw = pd.Series(np.asarray(raw, dtype=float), index=df.index) if len(raw) == len(df) else None
        if raw is None:
            raise ValueError("signals() must return a Series (or array) with one value per bar")
    s = pd.to_numeric(raw, errors="coerce").astype(float).reindex(df.index).fillna(0)  # True/False -> 1/0
    s = np.sign(s).astype(int)
    return s.clip(lower=0) if long_only else s


# --- look-ahead check ------------------------------------------------------------
def check_lookahead(fn: SignalFn, df: pd.DataFrame, params: dict | None = None, cuts: int = 4) -> list[str]:
    """Problems found (empty list = passed): re-run fn on the data cut off at a few points
    and check every signal before each cut is unchanged. If it changes, fn used later bars."""
    params = params or {}
    full = clean_signals(fn(df, **params), df)
    problems = []
    n = len(df)
    for frac in np.linspace(0.3, 0.9, cuts):
        k = int(n * frac)
        part = clean_signals(fn(df.iloc[:k], **params), df.iloc[:k])
        diff = part.ne(full.iloc[:k])
        if diff.any():
            first = diff.idxmax()
            problems.append(f"signal on {first:%Y-%m-%d %H:%M} changes when later bars are added "
                            f"(cut at bar {k:,}); the function looks ahead")
    return problems


# --- built-in strategies ------------------------------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _hold_between(enter_long, exit_long, enter_short, exit_short, index) -> pd.Series:
    """Stateful position from entry/exit conditions (each a boolean Series)."""
    pos, out = 0, np.zeros(len(index), dtype=int)
    el, xl, es, xs = (np.asarray(c, dtype=bool) for c in (enter_long, exit_long, enter_short, exit_short))
    for i in range(len(index)):
        if pos == 1 and xl[i]:
            pos = 0
        elif pos == -1 and xs[i]:
            pos = 0
        if pos == 0:
            pos = 1 if el[i] else -1 if es[i] else 0
        out[i] = pos
    return pd.Series(out, index=index)


def ema_cross(df, fast: int = 12, slow: int = 26):
    """Long when the fast EMA is above the slow EMA, short when below."""
    return np.sign(_ema(df.close, fast) - _ema(df.close, slow))


def macd(df, fast: int = 12, slow: int = 26, signal: int = 9):
    """Long when the MACD line is above its signal line, short when below."""
    line = _ema(df.close, fast) - _ema(df.close, slow)
    return np.sign(line - _ema(line, signal))


def rsi_reversion(df, period: int = 14, lower: float = 30, upper: float = 70):
    """Buy when RSI drops below `lower`, sell short above `upper`; exit both when RSI crosses 50."""
    r = _rsi(df.close, period)
    return _hold_between(r < lower, r > 50, r > upper, r < 50, df.index)


def bollinger_reversion(df, period: int = 20, width: float = 2.0):
    """Buy below the lower band, short above the upper band; exit at the middle band."""
    mid = df.close.rolling(period).mean()
    sd = df.close.rolling(period).std()
    c = df.close
    return _hold_between(c < mid - width * sd, c >= mid, c > mid + width * sd, c <= mid, df.index)


def donchian_breakout(df, period: int = 20, exit_period: int = 10):
    """Turtle-style: buy a close above the prior `period`-bar high, short below the prior low;
    exit on the opposite `exit_period`-bar extreme."""
    hi, lo = df.high.rolling(period).max().shift(1), df.low.rolling(period).min().shift(1)
    xhi, xlo = df.high.rolling(exit_period).max().shift(1), df.low.rolling(exit_period).min().shift(1)
    c = df.close
    return _hold_between(c > hi, c < xlo, c < lo, c > xhi, df.index)


def buy_and_hold(df):
    """Long from the first bar to the last."""
    return pd.Series(1, index=df.index)


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    low: float
    high: float
    step: float = 1
    help: str = ""

    @property
    def is_int(self) -> bool:
        return all(float(v).is_integer() for v in (self.default, self.low, self.high, self.step))


@dataclass(frozen=True)
class StrategySpec:
    """A strategy the app can offer: name, description, parameters and how to build it."""
    key: str
    name: str
    description: str
    params: tuple[Param, ...] = ()
    fn: SignalFn | None = None  # signal function, or None for an event strategy (below)
    event_class: type | None = None
    constraints: tuple[str, ...] = field(default=())  # e.g. ("fast < slow",)

    def make(self, long_only: bool = False, **params) -> Strategy:
        params = {p.name: (int(params[p.name]) if p.is_int else float(params[p.name]))
                  for p in self.params if p.name in params}
        for rule in self.constraints:
            a, op, b = rule.split()
            if op == "<" and not params[a] < params[b]:
                raise ValueError(f"{self.name}: needs {rule} (got {a}={params[a]}, {b}={params[b]})")
        if self.event_class is not None:
            return self.event_class(long_only=long_only, **params)
        return SignalStrategy(self.fn, long_only=long_only, **params)

    def factory(self, long_only: bool = False) -> Callable[..., Strategy]:
        """For RunConfig.make_strategy / optimisation: build with keyword parameters."""
        return lambda **params: self.make(long_only=long_only, **params)


BUILTINS: dict[str, StrategySpec] = {s.key: s for s in [
    StrategySpec("sma", "SMA crossover", "Long when the fast simple moving average is above the slow one, "
                 "short when below.", (Param("fast", 20, 2, 500), Param("slow", 50, 3, 1000)),
                 event_class=MovingAverageCrossover, constraints=("fast < slow",)),
    StrategySpec("ema", "EMA crossover", ema_cross.__doc__, (Param("fast", 12, 2, 500), Param("slow", 26, 3, 1000)),
                 fn=ema_cross, constraints=("fast < slow",)),
    StrategySpec("macd", "MACD", macd.__doc__, (Param("fast", 12, 2, 200), Param("slow", 26, 3, 400),
                                                Param("signal", 9, 2, 100)), fn=macd, constraints=("fast < slow",)),
    StrategySpec("rsi", "RSI mean reversion", rsi_reversion.__doc__,
                 (Param("period", 14, 2, 200), Param("lower", 30, 1, 49), Param("upper", 70, 51, 99)), fn=rsi_reversion),
    StrategySpec("bollinger", "Bollinger band reversion", bollinger_reversion.__doc__,
                 (Param("period", 20, 2, 500), Param("width", 2.0, 0.5, 5.0, 0.1)), fn=bollinger_reversion),
    StrategySpec("donchian", "Donchian breakout", donchian_breakout.__doc__,
                 (Param("period", 20, 2, 500), Param("exit_period", 10, 2, 500)), fn=donchian_breakout),
    StrategySpec("hold", "Buy and hold", buy_and_hold.__doc__, fn=buy_and_hold),
]}


# --- custom strategies from source code -----------------------------------------
CUSTOM_TEMPLATE = '''\
# Your strategy: a function named `signals` that gets the price data and returns the
# position you want on every bar: +1 = long, -1 = short, 0 = flat (NaN counts as flat).
#
#   df has columns open, high, low, close, volume (and spread for MT5 data), indexed by
#   bar time in UTC. pd (pandas) and np (numpy) are available.
#
# Use only data up to each bar (rolling windows, .shift(1), ...). The position you return
# for a bar is traded at the NEXT bar's open, so you can use that bar's close.
# Parameters with defaults show up as inputs in the app.

def signals(df, lookback=50, threshold=0.0):
    momentum = df.close.pct_change(lookback) * 100      # % change over `lookback` bars
    position = np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))
    return pd.Series(position, index=df.index)
'''


def load_custom(source: str) -> tuple[SignalFn, dict]:
    """Run strategy source code; return its signals() function and its parameters' defaults.

    The code runs on your machine with full Python access, like any script you run.
    """
    namespace: dict = {"pd": pd, "np": np}
    exec(compile(source, "<your strategy>", "exec"), namespace)  # noqa: S102 - the user's own code
    fn = namespace.get("signals")
    if not callable(fn):
        raise ValueError("Define a function named `signals(df, ...)` that returns the wanted position per bar")
    sig = inspect.signature(fn)
    params = {name: p.default for name, p in list(sig.parameters.items())[1:]
              if p.default is not inspect.Parameter.empty}
    return fn, params
