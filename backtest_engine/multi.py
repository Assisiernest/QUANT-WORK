"""Multi-asset backtests: one strategy traded on several assets from one pot of capital.

Capital is split between the assets by weight; each asset is traded in its own
sub-account by the same strategy, orders and sizing (risk sizing uses that asset's
sub-account equity). The sub-accounts are then added up bar by bar on a common
timeline, so the combined result has true portfolio equity, drawdowns and metrics.
Sub-accounts don't share margin, and one asset's losses don't shrink another's size.

    assets = [Asset("USDJPY", df_jpy), Asset("EURUSD", df_eur, weight=2)]
    multi = run_multi(assets, config, {"fast": 20, "slow": 50})
    multi.combined.stats()["sharpe"], multi.per_asset["EURUSD"].trade_log()
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from .engine import BacktestResult, point_size
from .optimize import RunConfig


@dataclass
class Asset:
    symbol: str
    df: pd.DataFrame
    weight: float = 1.0
    point: float | None = None  # price of one spread point; None = guess from the price level
    lot_step: float | None = None  # size rounding for this asset (1,000 FX micro lot, 1 share, 0.01 oz...)
    units: float | None = None  # fixed-size override for this asset


@dataclass
class MultiAssetResult:
    combined: BacktestResult
    per_asset: dict[str, BacktestResult]
    weights: dict[str, float]  # normalised, sum to 1
    benchmark: pd.Series  # buy-and-hold of the same assets and weights, USD, on the combined timeline
    asset_benchmarks: dict[str, pd.Series] = field(default_factory=dict)

    def asset_table(self) -> pd.DataFrame:
        """One row per asset: allocation and headline results."""
        rows = {}
        for sym, r in self.per_asset.items():
            s = r.stats()
            bh = self.asset_benchmarks[sym]
            rows[sym] = {
                "weight_pct": self.weights[sym] * 100, "start": r.initial_cash, "final_equity": s["final_equity"],
                "net_pnl": s["net_pnl"], "return_pct": s["return_pct"],
                "buy_hold_pct": (bh.iloc[-1] / bh.iloc[0] - 1) * 100, "sharpe": s["sharpe"],
                "sortino": s["sortino"], "max_drawdown_pct": s["max_drawdown_pct"], "trades": s["trades"],
                "win_rate_pct": s["win_rate_pct"], "profit_factor": s["profit_factor"],
                "fees": s["spread_paid"] + s["slippage_paid"],
            }
        return pd.DataFrame(rows).T

    def return_correlation(self) -> pd.DataFrame:
        """Correlation of the assets' daily strategy returns."""
        from .metrics import daily_returns
        return pd.DataFrame({sym: daily_returns(r).set_axis(_days(r)) for sym, r in self.per_asset.items()}).corr()


def _days(r: BacktestResult) -> pd.Index:
    from .data_loader import trading_day
    return pd.Index(sorted(set(trading_day(r.equity.index))))


def run_multi(assets: list[Asset], config: RunConfig, params: dict, title: str | None = None,
              trade_from: pd.Timestamp | None = None) -> MultiAssetResult:
    if not assets:
        raise ValueError("Add at least one asset")
    symbols = [a.symbol for a in assets]
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"Asset names must be unique: {symbols}")
    total = sum(a.weight for a in assets)
    if total <= 0 or any(a.weight < 0 for a in assets):
        raise ValueError("Weights must be >= 0 and add up to more than 0")
    weights = {a.symbol: a.weight / total for a in assets}

    per_asset, benchmarks = {}, {}
    for a in assets:
        if weights[a.symbol] == 0:
            continue
        cash = config.cash * weights[a.symbol]
        point = a.point or point_size(a.df["close"].median())
        cfg = replace(config, symbol=a.symbol, point=point, cash=cash, units=a.units or config.units)
        if cfg.sizer is not None and a.lot_step:
            cfg = replace(cfg, sizer=replace(cfg.sizer, lot_step=a.lot_step, min_units=a.lot_step))
        label = f"{a.symbol}: {title}" if title else a.symbol
        per_asset[a.symbol] = cfg.run(a.df, params, trade_from=trade_from, title=label)
        closes = a.df.close if trade_from is None else a.df.close[a.df.index >= trade_from]
        benchmarks[a.symbol] = cash * closes / closes.iloc[0]

    weights = {s: w for s, w in weights.items() if s in per_asset}
    if len(per_asset) == 1:
        (sym, only), = per_asset.items()
        only.title = title or only.title
        return MultiAssetResult(only, per_asset, weights, benchmarks[sym], benchmarks)

    combined = _combine(per_asset, config.cash, title or f"{len(per_asset)} assets")
    bench = _sum_on(combined.history.index, {s: (b, config.cash * weights[s]) for s, b in benchmarks.items()})
    return MultiAssetResult(combined, per_asset, weights, bench, benchmarks)


def _sum_on(index: pd.DatetimeIndex, series: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    """Add series up on `index`, carrying each forward and using its starting value before it begins."""
    total = pd.Series(0.0, index=index)
    for s, start in series.values():
        total += s.reindex(index.union(s.index)).ffill().reindex(index).fillna(start)
    return total


def _combine(per_asset: dict[str, BacktestResult], cash: float, title: str) -> BacktestResult:
    index = pd.DatetimeIndex(sorted(set().union(*(r.history.index for r in per_asset.values()))), name="time")
    cols = {"balance": [], "open_pnl": [], "equity": [], "long": [], "short": []}
    for r in per_asset.values():
        h = r.history.reindex(index.union(r.history.index)).ffill().reindex(index)
        h["balance"] = h.balance.fillna(r.initial_cash)
        h["equity"] = h.equity.fillna(r.initial_cash)
        h["open_pnl"] = h.open_pnl.fillna(0.0)
        pos = h.position.fillna(0.0)
        cols["balance"].append(h.balance)
        cols["open_pnl"].append(h.open_pnl)
        cols["equity"].append(h.equity)
        cols["long"].append((pos > 0).astype(int))
        cols["short"].append((pos < 0).astype(int))
    history = pd.DataFrame({
        "close": float("nan"),
        # For a portfolio, "position" is the number of assets with an open position.
        "position": sum(cols["long"]) + sum(cols["short"]),
        "avg_price": float("nan"),
        "balance": sum(cols["balance"]), "open_pnl": sum(cols["open_pnl"]), "equity": sum(cols["equity"]),
        "long_positions": sum(cols["long"]), "short_positions": sum(cols["short"]),
    }, index=index)
    history["drawdown_pct"] = (history.equity / history.equity.cummax() - 1) * 100

    def tagged(frame_of, time_col):
        parts = [frame_of(r).assign(symbol=s) for s, r in per_asset.items()]
        # Empty frames (an asset with no trades) have untyped columns that would turn times into objects.
        parts = [p for p in parts if len(p)] or parts[:1]
        out = pd.concat(parts, ignore_index=True).sort_values(time_col, kind="stable").reset_index(drop=True)
        for col in out.columns:
            if col.endswith("time"):
                out[col] = pd.to_datetime(out[col], utc=True)
        return out

    trades = tagged(lambda r: r.trades, "exit_time")
    fills = tagged(lambda r: r.fills, "time")
    first = next(iter(per_asset.values()))
    meta = {k: v for k, v in first.meta.items() if k not in ("symbol", "skipped_signals")}
    meta.update(assets=list(per_asset), skipped_signals=sum(r.meta.get("skipped_signals", 0) for r in per_asset.values()))
    return BacktestResult(trades, history, fills, cash, title=title, meta=meta)
