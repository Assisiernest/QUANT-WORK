"""Parameter search, out-of-sample testing and walk-forward analysis.

    config = RunConfig(symbol="USDJPY", point=0.001)
    grid = {"fast": [10, 20, 30], "slow": [50, 100, 200]}

    table = optimize(df, grid, config, objective="sharpe")           # every combination, best first
    oos = out_of_sample(df, grid, config, split=0.7)                  # tune on 70%, test on the last 30%
    wf = walk_forward(df, grid, config, windows=4, train_frac=0.5)    # re-tune before each test window

Out-of-sample and walk-forward runs warm the strategy up on the bars before each test
window (moving averages are already filled) but only trade inside the window. Walk-forward
chains the test windows: each starts with the equity the previous one ended with, and the
stitched result is a normal BacktestResult (metrics, trade log, chart all work on it).
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .broker import Slippage, SimulatedBroker
from .data import BarFeed
from .engine import BacktestResult, run_backtest
from .orders import OrderPolicy
from .portfolio import Portfolio
from .sizing import Sizer
from .strategy import MovingAverageCrossover, Strategy

# Objectives: any key of BacktestResult.stats(); higher is better for all of these.
OBJECTIVES = ("sharpe", "sortino", "net_pnl", "return_pct", "profit_factor", "calmar", "expectancy",
              "recovery_factor", "max_drawdown_pct")


@dataclass
class RunConfig:
    """Everything about a backtest except the strategy's parameters."""
    symbol: str = "USDJPY"
    point: float = 0.001
    cash: float = 10_000.0
    units: float = 10_000.0
    sizer: Sizer | None = None
    orders: OrderPolicy | None = None
    slippage: Slippage | None = None
    make_strategy: Callable[..., Strategy] = MovingAverageCrossover

    def run(self, df: pd.DataFrame, params: dict, trade_from: pd.Timestamp | None = None,
            cash: float | None = None, title: str | None = None) -> BacktestResult:
        portfolio = Portfolio(self.symbol, cash=self.cash if cash is None else cash, units=self.units,
                              point=self.point, sizer=self.sizer, orders=self.orders)
        broker = SimulatedBroker(self.point, copy.deepcopy(self.slippage))  # fresh random state per run
        result = run_backtest(BarFeed(df), self.make_strategy(**params), portfolio, broker,
                              title=title, trade_from=trade_from)
        result.meta["params"] = dict(params)
        return result


def param_grid(grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid's values."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _score(result: BacktestResult, objective: str, min_trades: int) -> float:
    s = result.stats()
    value = s[objective]
    if s["trades"] < min_trades or value is None or (isinstance(value, float) and math.isnan(value)):
        return -math.inf
    return float(value)


def optimize(df: pd.DataFrame, grid: dict[str, list], config: RunConfig, objective: str = "sharpe",
             min_trades: int = 10, trade_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Backtest every combination in `grid`; one row each, best `objective` first.

    Combinations the strategy rejects (e.g. fast >= slow) are skipped. Runs with fewer
    than min_trades trades rank last (score -inf), so a lucky handful of trades can't win.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    rows = []
    for params in param_grid(grid):
        try:
            result = config.run(df, params, trade_from=trade_from)
        except ValueError:
            continue  # invalid combination
        s = result.stats()
        rows.append({**params, "score": _score(result, objective, min_trades), "trades": s["trades"],
                     "net_pnl": s["net_pnl"], "return_pct": s["return_pct"], "sharpe": s["sharpe"],
                     "profit_factor": s["profit_factor"], "max_drawdown_pct": s["max_drawdown_pct"],
                     "win_rate_pct": s["win_rate_pct"]})
    if not rows:
        raise ValueError("No valid parameter combinations in the grid")
    table = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    table.attrs["objective"] = objective
    table.attrs["params"] = {k: type(grid[k][0]) for k in grid}  # name -> type, to undo pandas' upcasting
    return table


def best_params(table: pd.DataFrame) -> dict:
    """The top row's parameters, as the same Python types as in the grid (ints stay ints)."""
    row = table.iloc[0]
    return {k: kind(row[k]) for k, kind in table.attrs["params"].items()}


def _split_time(df: pd.DataFrame, split) -> pd.Timestamp:
    if isinstance(split, float):
        if not 0 < split < 1:
            raise ValueError("split must be between 0 and 1 (fraction of bars in sample)")
        return df.index[int(len(df) * split)]
    return pd.Timestamp(split)


# --- out-of-sample -------------------------------------------------------------
@dataclass
class OutOfSampleResult:
    split: pd.Timestamp
    objective: str
    table: pd.DataFrame  # in-sample optimisation, best first
    params: dict  # chosen on in-sample
    in_sample: BacktestResult  # best params, in-sample bars only
    out_of_sample: BacktestResult  # same params, out-of-sample bars only (warmed up on in-sample)
    full: BacktestResult  # same params over all bars, for the chart

    def comparison(self) -> pd.DataFrame:
        """In-sample vs out-of-sample, side by side."""
        keys = ["return_pct", "cagr_pct", "net_pnl", "max_drawdown_pct", "sharpe", "profit_factor",
                "win_rate_pct", "trades", "trades_per_day", "expectancy"]
        a, b = self.in_sample.stats(), self.out_of_sample.stats()
        return pd.DataFrame({"in_sample": [a[k] for k in keys], "out_of_sample": [b[k] for k in keys]}, index=keys)


def out_of_sample(df: pd.DataFrame, grid: dict[str, list], config: RunConfig, split: float | str = 0.7,
                  objective: str = "sharpe", min_trades: int = 10) -> OutOfSampleResult:
    """Pick parameters on the bars before `split` (a fraction, or a date), then test them
    untouched on the bars after it."""
    cut = _split_time(df, split)
    train = df[df.index < cut]
    table = optimize(train, grid, config, objective, min_trades)
    params = best_params(table)
    label = ", ".join(f"{k}={v}" for k, v in params.items())
    return OutOfSampleResult(
        split=cut, objective=objective, table=table, params=params,
        in_sample=config.run(train, params, title=f"In-sample ({label})"),
        out_of_sample=config.run(df, params, trade_from=cut, title=f"Out-of-sample ({label})"),
        full=_mark_split(config.run(df, params, title=f"Full period ({label}), out-of-sample from {cut:%Y-%m-%d}"), cut),
    )


def _mark_split(result: BacktestResult, cut: pd.Timestamp) -> BacktestResult:
    result.meta["split"] = cut  # the chart draws the in-sample / out-of-sample line here
    return result


# --- walk-forward --------------------------------------------------------------
@dataclass
class WalkForwardResult:
    objective: str
    anchored: bool
    windows: pd.DataFrame  # one row per window: dates, chosen params, in-sample and out-of-sample results
    results: list[BacktestResult] = field(repr=False)  # each window's out-of-sample run
    combined: BacktestResult = field(repr=False)  # all out-of-sample windows chained together

    @property
    def efficiency(self) -> float:
        """Walk-forward efficiency: out-of-sample return per day / in-sample return per day.
        Around 0.5 or more is usually taken as the in-sample edge carrying over."""
        w = self.windows
        is_rate = (w.is_return_pct / w.train_days).mean()
        oos_rate = (w.oos_return_pct / w.test_days).mean()
        return oos_rate / is_rate if is_rate > 0 else float("nan")


def walk_forward(df: pd.DataFrame, grid: dict[str, list], config: RunConfig, windows: int = 4,
                 train_frac: float = 0.5, anchored: bool = False, objective: str = "sharpe",
                 min_trades: int = 10) -> WalkForwardResult:
    """Walk-forward analysis.

    The first train_frac of the bars is the first training window; the rest is cut into
    `windows` equal test windows. Before each test window the grid is re-optimised on the
    training window just before it: the same length each time (rolling), or from the start
    of the data (anchored). Each test window is then traded with the chosen parameters.
    """
    if windows < 1 or not 0 < train_frac < 1:
        raise ValueError("need windows >= 1 and 0 < train_frac < 1")
    n = len(df)
    train_len = int(n * train_frac)
    test_len = (n - train_len) // windows
    if test_len < 2:
        raise ValueError("not enough bars for that many test windows")

    rows, results = [], []
    equity = config.cash
    for w in range(windows):
        test_start = train_len + w * test_len
        test_end = n if w == windows - 1 else test_start + test_len
        train_start = 0 if anchored else test_start - train_len
        train = df.iloc[train_start:test_start]
        cut = df.index[test_start]

        table = optimize(train, grid, config, objective, min_trades)
        params = best_params(table)
        is_result = config.run(train, params)
        # Test: warm up on the training bars, trade only from the window's first bar.
        oos = config.run(df.iloc[train_start:test_end], params, trade_from=cut, cash=equity,
                         title=f"Window {w + 1} out-of-sample")
        equity = oos.equity.iloc[-1]
        results.append(oos)
        a, b = is_result.stats(), oos.stats()
        rows.append({
            "window": w + 1,
            "train_start": train.index[0], "train_end": train.index[-1],
            "test_start": cut, "test_end": df.index[test_end - 1],
            **{f"param_{k}": v for k, v in params.items()},
            "is_score": table.score.iloc[0], "is_return_pct": a["return_pct"], "is_trades": a["trades"],
            "oos_return_pct": b["return_pct"], "oos_net_pnl": b["net_pnl"], "oos_sharpe": b["sharpe"],
            "oos_max_drawdown_pct": b["max_drawdown_pct"], "oos_trades": b["trades"],
            "oos_end_equity": equity,
            "train_days": (train.index[-1] - train.index[0]) / pd.Timedelta(days=1),
            "test_days": (df.index[test_end - 1] - cut) / pd.Timedelta(days=1),
        })
    table = pd.DataFrame(rows).set_index("window")
    combined = _stitch(results, config.cash, title=f"Walk-forward out-of-sample ({windows} windows, "
                                                   f"{'anchored' if anchored else 'rolling'}, {objective})")
    combined.meta["windows"] = [(r.test_start, r.test_end, f"W{i}") for i, r in table.iterrows()]
    return WalkForwardResult(objective, anchored, table, results, combined)


def _stitch(results: list[BacktestResult], initial_cash: float, title: str) -> BacktestResult:
    """Chain out-of-sample windows into one result (each window started on the previous one's equity)."""
    history = pd.concat([r.history for r in results])
    history["drawdown_pct"] = (history.equity / history.equity.cummax() - 1) * 100
    trades = pd.concat([r.trades for r in results], ignore_index=True)
    fills = pd.concat([r.fills for r in results], ignore_index=True)
    meta = {k: v for k, v in results[0].meta.items() if k != "params"}
    meta["skipped_signals"] = sum(r.meta.get("skipped_signals", 0) for r in results)
    return BacktestResult(trades, history, fills, initial_cash, title=title, meta=meta)
