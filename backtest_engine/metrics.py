"""Performance metrics for a BacktestResult.

compute_metrics(result) returns sections of Metric(name, value, kind); kind says how to
show the value ("money", "pct", "change" (signed %), "ratio", "int", "pips", "duration", "price").
Annualised figures use 252 trading days a year and should be read with care on a
few months of data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data_loader import trading_day

if TYPE_CHECKING:
    from .engine import BacktestResult

TRADING_DAYS = 252
NAN = float("nan")


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: object
    kind: str
    key: str  # short id, for code and CSV


def daily_returns(result: "BacktestResult") -> pd.Series:
    """Equity return of each trading day (17:00 New York close); the first is against the start."""
    eq = result.equity
    closes = eq.groupby(trading_day(eq.index)).last()
    return pd.concat([pd.Series([result.initial_cash]), closes.reset_index(drop=True)]).pct_change().dropna()


def longest_drawdown(equity: pd.Series) -> pd.Timedelta:
    """Longest time from an equity peak until it's regained (or the data ends)."""
    peak = equity.cummax()
    underwater = equity < peak
    if not underwater.any():
        return pd.Timedelta(0)
    longest, start = pd.Timedelta(0), None
    times = equity.index
    for i, below in enumerate(underwater.to_numpy()):
        if below and start is None:
            start = times[i - 1] if i else times[0]  # the peak bar
        elif not below and start is not None:
            longest, start = max(longest, times[i] - start), None
    if start is not None:
        longest = max(longest, times[-1] - start)
    return longest


def _streak(flags: pd.Series) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def compute_metrics(result: "BacktestResult") -> dict[str, list[Metric]]:
    h, t = result.history, result.trades
    start, eq = result.initial_cash, result.equity
    final = eq.iloc[-1]
    net = final - start

    years = (h.index[-1] - h.index[0]) / pd.Timedelta(days=365)
    cagr = ((final / start) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else NAN
    dd_usd = (eq - eq.cummax()).min()
    dd_pct = h.drawdown_pct.min()

    r = daily_returns(result)
    sd = r.std()
    downside = np.sqrt((r.clip(upper=0) ** 2).mean())
    sharpe = r.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else NAN
    sortino = r.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else NAN

    wins, losses = t[t.pnl > 0], t[t.pnl < 0]
    gross_win, gross_loss = wins.pnl.sum(), losses.pnl.sum()
    avg_win = wins.pnl.mean() if len(wins) else 0.0
    avg_loss = losses.pnl.mean() if len(losses) else 0.0
    held = t.exit_time - t.entry_time
    n_days = len(r)

    def m(name, value, kind, key):
        return Metric(name, value, kind, key)

    return {
        "Returns": [
            m("Starting balance", start, "money", "start"),
            m("Final equity", final, "money", "final_equity"),
            m("Net P&L", net, "money", "net_pnl"),
            m("Return", net / start * 100, "change", "return_pct"),
            m("Annualised return (CAGR)", cagr, "change", "cagr_pct"),
            m("Gross profit", gross_win, "money", "gross_profit"),
            m("Gross loss", gross_loss, "money", "gross_loss"),
            m("Spread paid", t.spread_cost.sum(), "money", "spread_paid"),
            m("Slippage paid", t.slippage_cost.sum(), "money", "slippage_paid"),
            m("Total pips", t.pips.sum(), "pips", "total_pips"),
        ],
        "Risk": [
            m("Max drawdown", dd_pct, "pct", "max_drawdown_pct"),
            m("Max drawdown ($)", dd_usd, "money", "max_drawdown_usd"),
            m("Longest drawdown", longest_drawdown(eq), "duration", "longest_drawdown"),
            m("Volatility (annualised)", sd * np.sqrt(TRADING_DAYS) * 100, "pct", "volatility_pct"),
            m("Value at risk 95% (1 day)", r.quantile(0.05) * 100 if len(r) else NAN, "change", "var95_pct"),
            m("Value at risk 99% (1 day)", r.quantile(0.01) * 100 if len(r) else NAN, "change", "var99_pct"),
            m("Sharpe ratio", sharpe, "ratio", "sharpe"),
            m("Sortino ratio", sortino, "ratio", "sortino"),
            m("Calmar ratio (CAGR / max DD)", cagr / -dd_pct if dd_pct < 0 else NAN, "ratio", "calmar"),
            m("Recovery factor (net P&L / max DD)", net / -dd_usd if dd_usd < 0 else NAN, "ratio", "recovery_factor"),
            m("Best day", r.max() * 100 if n_days else NAN, "change", "best_day_pct"),
            m("Worst day", r.min() * 100 if n_days else NAN, "change", "worst_day_pct"),
        ],
        "Trades": [
            m("Total trades", len(t), "int", "trades"),
            m("Winners", len(wins), "int", "winners"),
            m("Losers", len(losses), "int", "losers"),
            m("Win rate", len(wins) / len(t) * 100 if len(t) else NAN, "pct", "win_rate_pct"),
            m("Profit factor", gross_win / -gross_loss if gross_loss < 0 else NAN, "ratio", "profit_factor"),
            m("Expectancy per trade", t.pnl.mean() if len(t) else NAN, "money", "expectancy"),
            m("Expectancy per trade (pips)", t.pips.mean() if len(t) else NAN, "pips", "expectancy_pips"),
            m("Average win", avg_win, "money", "avg_win"),
            m("Average loss", avg_loss, "money", "avg_loss"),
            m("Payoff ratio (avg win / avg loss)", avg_win / -avg_loss if avg_loss < 0 else NAN, "ratio", "payoff"),
            m("Largest win", t.pnl.max() if len(t) else NAN, "money", "largest_win"),
            m("Largest loss", t.pnl.min() if len(t) else NAN, "money", "largest_loss"),
            m("Max consecutive wins", _streak(t.pnl > 0), "int", "max_consec_wins"),
            m("Max consecutive losses", _streak(t.pnl < 0), "int", "max_consec_losses"),
            m("Exits on signal", int((t.exit_reason == "signal").sum()), "int", "exits_signal"),
            m("Exits at stop-loss", int((t.exit_reason == "stop loss").sum()), "int", "exits_stop_loss"),
            m("Exits at take-profit", int((t.exit_reason == "take profit").sum()), "int", "exits_take_profit"),
            m("Exits at trailing stop", int((t.exit_reason == "trailing stop").sum()), "int", "exits_trailing"),
            m("Margin calls", int((t.exit_reason == "margin call").sum()), "int", "margin_calls"),
            m("Signals skipped (size 0)", result.meta.get("skipped_signals", 0), "int", "skipped_signals"),
        ],
        "Exposure": [
            m("Time in market", (h.position != 0).mean() * 100, "pct", "exposure_pct"),
            m("Average holding time", held.mean() if len(t) else NAN, "duration", "avg_hold"),
            m("Median holding time", held.median() if len(t) else NAN, "duration", "median_hold"),
            m("Longest trade", held.max() if len(t) else NAN, "duration", "longest_trade"),
            m("Average bars held", t.bars_held.mean() if len(t) else NAN, "ratio", "avg_bars"),
            m("Trades per day", len(t) / n_days if n_days else NAN, "ratio", "trades_per_day"),
            m("Average MAE (pips)", t.mae_pips.mean() if len(t) else NAN, "pips", "avg_mae_pips"),
            m("Average MFE (pips)", t.mfe_pips.mean() if len(t) else NAN, "pips", "avg_mfe_pips"),
            m("Average size (units)", t.units.mean() if len(t) else NAN, "int", "avg_units"),
            m("Largest size (units)", t.units.max() if len(t) else NAN, "int", "max_units"),
        ],
    }


def by_side(result: "BacktestResult") -> pd.DataFrame:
    """Long vs short vs all: trades, win rate, P&L, pips, profit factor, average trade."""
    t = result.trades
    rows = {}
    for name, g in [("Long", t[t.side == "long"]), ("Short", t[t.side == "short"]), ("All", t)]:
        loss = g.pnl[g.pnl < 0].sum()
        rows[name] = {
            "trades": len(g),
            "win_rate_pct": (g.pnl > 0).mean() * 100 if len(g) else NAN,
            "net_pnl": g.pnl.sum(),
            "pips": g.pips.sum(),
            "profit_factor": g.pnl[g.pnl > 0].sum() / -loss if loss < 0 else NAN,
            "avg_trade": g.pnl.mean() if len(g) else NAN,
            "spread_paid": g.spread_cost.sum(),
            "slippage_paid": g.slippage_cost.sum(),
        }
    return pd.DataFrame(rows).T


def format_value(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return "-"
    if kind == "money":
        return f"{'-' if value < 0 else ''}${abs(value):,.2f}"
    if kind == "change":
        return f"{value:+.2f}%" if value else "0.00%"
    if kind == "pct":
        return f"{value:.2f}%"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "pips":
        return f"{value:+,.1f}"
    if kind == "duration":
        return format_duration(value)
    if kind == "datetime":
        return f"{value:%Y-%m-%d %H:%M}"
    return f"{value:,.2f}"


def format_duration(td: pd.Timedelta) -> str:
    if pd.isna(td):
        return "-"
    minutes = int(td.total_seconds() // 60)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def metrics_frame(result: "BacktestResult") -> pd.DataFrame:
    """All metrics as one table: section, metric, value (raw) and display (formatted)."""
    rows = [
        {"section": section, "metric": mt.name, "key": mt.key, "value": mt.value,
         "display": format_value(mt.value, mt.kind)}
        for section, items in compute_metrics(result).items() for mt in items
    ]
    return pd.DataFrame(rows)
