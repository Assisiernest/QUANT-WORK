"""QuantConnect-style statistics and the series behind the results charts.

qc_statistics(result, benchmark) returns the statistics QuantConnect prints at the end
of a backtest (Sharpe, Sortino, probabilistic Sharpe, alpha/beta, information ratio,
Treynor, turnover, ...), computed from daily returns at the 17:00 New York close.
The benchmark is an equity series, normally buy-and-hold of the same assets.

Conventions (as in QuantConnect): annualised with 252 trading days; Sharpe and Sortino
use an annual risk-free rate (default 0); alpha is annualised; beta and alpha come
from regressing daily strategy returns on daily benchmark returns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

from .data_loader import trading_day
from .metrics import Metric, compute_metrics, format_value

if TYPE_CHECKING:
    from .engine import BacktestResult

DAYS = 252
NAN = float("nan")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def daily_equity(equity: pd.Series, start: float | None = None, calendar: bool = False) -> pd.Series:
    """Equity at each trading day's close (17:00 New York), optionally with the start value first.

    calendar=True also includes weekends and holidays (equity carried forward, so those days
    return zero), which is how QuantConnect samples its daily returns."""
    days = equity.groupby(trading_day(equity.index)).last()
    days.index = pd.to_datetime(days.index)
    if calendar:
        days = days.reindex(pd.date_range(days.index[0], days.index[-1], freq="D")).ffill()
    if start is not None:
        days = pd.concat([pd.Series([start], index=[days.index[0] - pd.Timedelta(days=1)]), days])
    return days


def daily_returns(equity: pd.Series, start: float | None = None, calendar: bool = False) -> pd.Series:
    return daily_equity(equity, start, calendar).pct_change().dropna()


def probabilistic_sharpe(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """Probability that the true (per-period) Sharpe ratio is above benchmark_sharpe,
    allowing for skew and fat tails (Bailey & Lopez de Prado, 2012)."""
    n = len(returns)
    sd = returns.std()
    if n < 3 or not sd > 0:
        return NAN
    sr = returns.mean() / sd
    skew, kurt = returns.skew(), returns.kurt() + 3  # pandas kurt is excess kurtosis
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if not denom > 0:
        return NAN
    return _norm_cdf((sr - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom))


def drawdown_periods(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """The deepest drawdowns: peak, trough, recovery (NaT if not recovered), depth and length."""
    peak = equity.cummax()
    under = equity < peak
    rows, start = [], None
    times = equity.index
    for i, below in enumerate(under.to_numpy()):
        if below and start is None:
            start = i - 1 if i else 0
        if (not below or i == len(times) - 1) and start is not None:
            end = i if not below else None
            seg = equity.iloc[start:(i + 1)]
            trough = seg.idxmin()
            rows.append({
                "peak": times[start], "trough": trough, "recovery": times[end] if end is not None else pd.NaT,
                "depth_pct": (seg.min() / equity.iloc[start] - 1) * 100,
                "length": (times[end] if end is not None else times[-1]) - times[start],
                "recovered": end is not None,
            })
            start = None
    out = pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth_pct", "length", "recovered"])
    return out.sort_values("depth_pct").head(top).reset_index(drop=True)


def monthly_returns(equity: pd.Series, start: float) -> pd.DataFrame:
    """Return % of each calendar month (rows: year, columns: Jan..Dec), first month against the start."""
    eq = equity.copy()
    eq.index = eq.index.tz_convert("UTC").tz_localize(None)
    month_end = eq.resample("ME").last().dropna()
    rets = pd.concat([pd.Series([start]), month_end.reset_index(drop=True)]).pct_change().dropna() * 100
    rets.index = month_end.index
    table = pd.DataFrame({"year": rets.index.year, "month": rets.index.month, "ret": rets.to_numpy()})
    pivot = table.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    year_eq = month_end.groupby(month_end.index.year).last()
    pivot["Year"] = pd.concat([pd.Series([start]), year_eq.reset_index(drop=True)]).pct_change().dropna().to_numpy() * 100
    return pivot


def rolling_sharpe(equity: pd.Series, window: int = 20, risk_free: float = 0.0, calendar: bool = False) -> pd.Series:
    """Annualised Sharpe ratio over a rolling window of trading days."""
    r = daily_returns(equity, calendar=calendar) - risk_free / DAYS
    return (r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(DAYS)).dropna()


def qc_statistics(result: "BacktestResult", benchmark: pd.Series | None = None,
                  risk_free: float = 0.0, calendar_days: bool = False) -> list[Metric]:
    """QuantConnect's backtest statistics, in its order. calendar_days=True uses QuantConnect's
    daily-return sampling (weekends included as zero-return days)."""
    s = result.stats()
    t = result.trades
    start = result.initial_cash
    r = daily_returns(result.equity, start, calendar_days)
    rf_day = risk_free / DAYS
    excess = r - rf_day
    ann_sd = r.std() * math.sqrt(DAYS)
    # QuantConnect: downside deviation = sample std of the negative daily returns, annualised.
    neg = excess[excess < 0]
    downside = neg.std() * math.sqrt(DAYS) if len(neg) > 1 else NAN
    years = (result.history.index[-1] - result.history.index[0]).days / 365  # calendar years, as QuantConnect
    cagr = (s["final_equity"] / start) ** (1 / years) - 1 if years > 0 and s["final_equity"] > 0 else NAN

    alpha = beta = info = tracking = treynor = NAN
    if benchmark is not None and len(r) > 2:
        b = daily_returns(benchmark, benchmark.iloc[0], calendar_days).reindex(r.index)
        both = pd.concat([r, b], axis=1, keys=["s", "b"]).dropna()
        if len(both) > 2 and both.b.var() > 0:
            beta = both.s.cov(both.b) / both.b.var()
            alpha = ((both.s.mean() - rf_day) - beta * (both.b.mean() - rf_day)) * DAYS
            active = both.s - both.b
            tracking = active.std() * math.sqrt(DAYS)
            info = active.mean() * DAYS / tracking if tracking > 0 else NAN
            treynor = (cagr - risk_free) / beta if beta else NAN

    wins, losses = t[t.pnl > 0], t[t.pnl < 0]
    win_rate = len(wins) / len(t) if len(t) else NAN
    loss_rate = len(losses) / len(t) if len(t) else NAN
    before = start + t.pnl.cumsum() - t.pnl  # balance before each trade
    pct = t.pnl / before * 100
    avg_win_pct = pct[t.pnl > 0].mean() if len(wins) else 0.0
    avg_loss_pct = pct[t.pnl < 0].mean() if len(losses) else 0.0
    pl_ratio = avg_win_pct / -avg_loss_pct if avg_loss_pct < 0 else NAN
    expectancy = win_rate * pl_ratio - loss_rate if pl_ratio == pl_ratio else NAN

    fees = s["spread_paid"] + s["slippage_paid"]
    eq_days = daily_equity(result.equity)
    traded = result.fills.groupby(pd.to_datetime(trading_day(pd.DatetimeIndex(result.fills.time)))).notional.sum() \
        if len(result.fills) else pd.Series(dtype=float)
    turnover = (traded.reindex(eq_days.index, fill_value=0) / eq_days).mean() * 100 if len(eq_days) else NAN

    dd = drawdown_periods(result.equity, top=1)
    recovery_days = NAN
    if len(dd) and dd.recovered.iloc[0]:
        recovery_days = (dd.recovery.iloc[0] - dd.trough.iloc[0]) / pd.Timedelta(days=1)

    m = lambda name, value, kind, key: Metric(name, value, kind, key)  # noqa: E731
    return [
        m("Total Orders", len(result.fills), "int", "total_orders"),
        m("Average Win", avg_win_pct, "change", "avg_win_pct"),
        m("Average Loss", avg_loss_pct, "change", "avg_loss_pct"),
        m("Compounding Annual Return", cagr * 100, "change", "cagr_pct"),
        m("Drawdown", -s["max_drawdown_pct"], "pct", "drawdown_pct"),
        m("Expectancy", expectancy, "ratio", "qc_expectancy"),
        m("Start Equity", start, "money", "start"),
        m("End Equity", s["final_equity"], "money", "end"),
        m("Net Profit", s["return_pct"], "change", "net_profit_pct"),
        m("Sharpe Ratio", excess.mean() / r.std() * math.sqrt(DAYS) if r.std() > 0 else NAN, "ratio", "sharpe"),
        m("Sortino Ratio", excess.mean() * DAYS / downside if downside > 0 else NAN, "ratio", "sortino"),
        # QuantConnect's reference is a Sharpe ratio of 1 (annual): the chance the true Sharpe beats it.
        m("Probabilistic Sharpe Ratio", probabilistic_sharpe(excess, 1 / math.sqrt(DAYS)) * 100, "pct", "psr_pct"),
        m("Loss Rate", loss_rate * 100, "pct", "loss_rate_pct"),
        m("Win Rate", win_rate * 100, "pct", "win_rate_pct"),
        m("Profit-Loss Ratio", pl_ratio, "ratio", "profit_loss_ratio"),
        m("Alpha", alpha, "ratio", "alpha"),
        m("Beta", beta, "ratio", "beta"),
        m("Annual Standard Deviation", ann_sd, "ratio", "annual_sd"),
        m("Annual Variance", ann_sd ** 2, "ratio", "annual_variance"),
        m("Information Ratio", info, "ratio", "information_ratio"),
        m("Tracking Error", tracking, "ratio", "tracking_error"),
        m("Treynor Ratio", treynor, "ratio", "treynor"),
        m("Total Fees", fees, "money", "total_fees"),
        m("Portfolio Turnover", turnover, "pct", "turnover_pct"),
        m("Drawdown Recovery (days)", recovery_days, "ratio", "drawdown_recovery_days"),
    ]


def statistics_frame(result: "BacktestResult", benchmark: pd.Series | None = None,
                     risk_free: float = 0.0, calendar_days: bool = False) -> pd.DataFrame:
    """QuantConnect statistics first, then QuantConnect's trade statistics, then every engine metric."""
    rows = [{"section": "Overview (QuantConnect)", "metric": x.name, "key": x.key, "value": x.value,
             "display": format_value(x.value, x.kind)}
            for x in qc_statistics(result, benchmark, risk_free, calendar_days)]
    fees = result.fills.spread_cost.sum() + result.fills.slippage_cost.sum() if len(result.fills) else 0.0
    rows += [{"section": "Trade statistics (QuantConnect)", "metric": x.name, "key": x.key, "value": x.value,
              "display": format_value(x.value, x.kind)} for x in trade_statistics(result.trades, fees)]
    for section, items in compute_metrics(result).items():
        rows += [{"section": section, "metric": x.name, "key": x.key, "value": x.value,
                  "display": format_value(x.value, x.kind)} for x in items]
    return pd.DataFrame(rows)


# --- QuantConnect trade statistics ---------------------------------------------------
def _longest_run(flags: pd.Series) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def trade_statistics(trades: pd.DataFrame, fees: float = 0.0) -> list[Metric]:
    """QuantConnect's "trade statistics" for a table of closed trades with entry_time, exit_time,
    pnl (USD) and, optionally, mae_usd / mfe_usd.

    Per-trade Sharpe and Sortino are average P&L divided by the standard deviation / downside
    deviation of the trades' P&L (not daily returns), so they differ from the portfolio's."""
    t = trades
    m = lambda name, value, kind, key: Metric(name, value, kind, key)  # noqa: E731
    if t.empty:
        return [m("Total trades", 0, "int", "ts_trades")]
    pnl = t.pnl
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]  # as QuantConnect: a break-even trade is a loser
    held = t.exit_time - t.entry_time
    cum = pnl.cumsum()
    closed_dd = (cum - pd.concat([pd.Series([0.0]), cum.reset_index(drop=True)]).cummax().iloc[1:].to_numpy()).min()
    downside = losses.std() if len(losses) > 1 else NAN  # QuantConnect: sample std of the trades with P&L <= 0
    sd = pnl.std()
    avg_profit = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    has_exc = {"mae_usd", "mfe_usd"} <= set(t.columns)
    dur = lambda s: s.mean() if len(s) else pd.NaT  # noqa: E731
    med = lambda s: s.median() if len(s) else pd.NaT  # noqa: E731
    out = [
        m("Start", t.entry_time.min(), "datetime", "ts_start"),
        m("End", t.exit_time.max(), "datetime", "ts_end"),
        m("Total trades", len(t), "int", "ts_trades"),
        m("Winning trades", len(wins), "int", "ts_winners"),
        m("Losing trades", len(losses), "int", "ts_losers"),
        m("Total profit/loss", pnl.sum(), "money", "ts_total_pnl"),
        m("Total profit", wins.sum(), "money", "ts_total_profit"),
        m("Total loss", losses.sum(), "money", "ts_total_loss"),
        m("Largest profit", pnl.max(), "money", "ts_largest_profit"),
        m("Largest loss", pnl.min(), "money", "ts_largest_loss"),
        m("Average profit/loss", pnl.mean(), "money", "ts_avg_pnl"),
        m("Average profit", avg_profit, "money", "ts_avg_profit"),
        m("Average loss", avg_loss, "money", "ts_avg_loss"),
        m("Average trade duration", dur(held), "duration", "ts_avg_duration"),
        m("Average winning duration", dur(held[pnl > 0]), "duration", "ts_avg_win_duration"),
        m("Average losing duration", dur(held[pnl <= 0]), "duration", "ts_avg_loss_duration"),
        m("Median trade duration", med(held), "duration", "ts_median_duration"),
        m("Median winning duration", med(held[pnl > 0]), "duration", "ts_median_win_duration"),
        m("Median losing duration", med(held[pnl <= 0]), "duration", "ts_median_loss_duration"),
        m("Max consecutive wins", _longest_run(pnl > 0), "int", "ts_max_wins"),
        m("Max consecutive losses", _longest_run(pnl <= 0), "int", "ts_max_losses"),
        m("Profit-loss ratio", avg_profit / -avg_loss if avg_loss < 0 else NAN, "ratio", "ts_profit_loss_ratio"),
        m("Win-loss ratio", len(wins) / len(losses) if len(losses) else NAN, "ratio", "ts_win_loss_ratio"),
        m("Win rate", len(wins) / len(t) * 100, "pct", "ts_win_rate"),
        m("Loss rate", len(losses) / len(t) * 100, "pct", "ts_loss_rate"),
    ]
    if has_exc:
        out += [
            m("Average MAE", t.mae_usd.mean(), "money", "ts_avg_mae"),
            m("Average MFE", t.mfe_usd.mean(), "money", "ts_avg_mfe"),
            m("Largest MAE", t.mae_usd.min(), "money", "ts_largest_mae"),
            m("Largest MFE", t.mfe_usd.max(), "money", "ts_largest_mfe"),
        ]
    out += [
        m("Maximum closed-trade drawdown", closed_dd, "money", "ts_closed_dd"),
        m("Profit/loss standard deviation", sd, "money", "ts_pnl_sd"),
        m("Profit/loss downside deviation", downside, "money", "ts_pnl_downside"),
        m("Profit factor", wins.sum() / -losses.sum() if len(losses) else NAN, "ratio", "ts_profit_factor"),
        m("Sharpe ratio (per trade)", pnl.mean() / sd if sd > 0 else NAN, "ratio", "ts_sharpe"),
        m("Sortino ratio (per trade)", pnl.mean() / downside if downside > 0 else NAN, "ratio", "ts_sortino"),
        m("Profit to max drawdown", pnl.sum() / -closed_dd if closed_dd < 0 else NAN, "ratio", "ts_profit_to_dd"),
        m("Total fees", fees, "money", "ts_fees"),
    ]
    return out
