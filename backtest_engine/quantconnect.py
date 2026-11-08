"""Read a QuantConnect backtest export (the results .json) and check it for problems.

    qc = parse_qc("Smooth Blue Lion.json")
    qc.statistics            # the statistics QuantConnect printed, as it printed them
    qc.equity, qc.benchmark  # daily series; qc.closed_trades, qc.orders  # tables
    qc.findings              # things worth knowing, most serious first

The checks look for the usual ways a QuantConnect result misleads: orders the broker
model refused (so a position was never closed), profit that is unrealised on a position
still open at the end, closed trades that stop long before the backtest does, and a beta
so large that the result is mostly leveraged exposure to the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ORDER_TYPES = {0: "Market", 1: "Limit", 2: "Stop market", 3: "Stop limit", 4: "Market on open", 5: "Market on close",
               6: "Option exercise", 7: "Limit if touched", 8: "Combo market", 9: "Combo limit",
               10: "Combo leg limit", 11: "Trailing stop"}
ORDER_STATUS = {0: "New", 1: "Submitted", 2: "Partially filled", 3: "Filled", 4: "None", 5: "Canceled", 6: "None",
                7: "Invalid", 8: "Cancel pending", 9: "Update submitted"}


@dataclass
class Finding:
    severity: str  # "high", "medium" or "info"
    title: str
    detail: str


@dataclass
class QCResult:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    status: str
    currency: str
    statistics: dict[str, str]  # as printed, in QuantConnect's order
    portfolio: dict[str, float]  # totalPerformance.portfolioStatistics
    trade_stats: dict[str, str]  # totalPerformance.tradeStatistics
    runtime: dict[str, str]
    equity: pd.Series  # equity at each UTC day's last sample
    benchmark: pd.Series  # benchmark price, daily
    drawdown: pd.Series  # QuantConnect's drawdown chart, % (negative)
    exposure: pd.DataFrame  # long / short ratio per day
    turnover: pd.Series
    capacity: pd.Series
    orders: pd.DataFrame
    closed_trades: pd.DataFrame
    analysis: list[dict] = field(default_factory=list)  # QuantConnect's own analysis notes
    findings: list[Finding] = field(default_factory=list)

    @property
    def start_equity(self) -> float:
        return float(self.portfolio.get("startEquity", self.equity.iloc[0]))

    @property
    def end_equity(self) -> float:
        return float(self.portfolio.get("endEquity", self.equity.iloc[-1]))

    def benchmark_equity(self) -> pd.Series:
        """The benchmark rebased to the starting equity, to draw next to the strategy."""
        return self.benchmark / self.benchmark.iloc[0] * self.start_equity


def _num(x) -> float:
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").strip())
    except ValueError:
        return float("nan")


def _series(chart: dict, name: str, column: int = -1) -> pd.Series:
    s = chart.get("series", {}).get(name)
    if not s or not s.get("values"):
        return pd.Series(dtype=float)
    rows = s["values"]
    out = pd.Series([r[column] for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="s", utc=True))
    return out.groupby(out.index.normalize()).last().astype(float)  # last sample of each UTC day


def parse_qc(source) -> QCResult:
    """source: a path, bytes, an open file, or an already loaded dict."""
    if isinstance(source, dict):
        d = source
    elif isinstance(source, (str, Path)):
        d = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    else:
        raw = source.read() if hasattr(source, "read") else source
        d = json.loads(raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw)
    if "totalPerformance" not in d and "statistics" not in d:
        raise ValueError("This doesn't look like a QuantConnect backtest results file "
                         "(no totalPerformance or statistics section)")

    cfg, charts = d.get("algorithmConfiguration", {}), d.get("charts", {})
    perf = d.get("totalPerformance", {})
    portfolio = {k: _num(v) for k, v in perf.get("portfolioStatistics", {}).items()}
    equity = _series(charts.get("Strategy Equity", {}), "Equity")
    if equity.empty:
        raise ValueError("The file has no Strategy Equity chart, so there's nothing to plot")
    bm = _series(charts.get("Benchmark", {}), "Benchmark")
    exposure = pd.DataFrame({
        "long": _series(charts.get("Exposure", {}), next((n for n in charts.get("Exposure", {}).get("series", {})
                                                          if "Long" in n), "")),
        "short": _series(charts.get("Exposure", {}), next((n for n in charts.get("Exposure", {}).get("series", {})
                                                           if "Short" in n), "")),
    })

    orders = pd.DataFrame([{
        "id": o.get("id"), "time": pd.Timestamp(o["time"]), "symbol": o.get("symbol", {}).get("value", ""),
        "type": ORDER_TYPES.get(o.get("type"), str(o.get("type"))), "status": ORDER_STATUS.get(o.get("status"), str(o.get("status"))),
        "quantity": o.get("quantity"), "price": o.get("price"), "value": o.get("value"), "tag": o.get("tag", ""),
        "filled_at": pd.Timestamp(o["lastFillTime"]) if o.get("lastFillTime") else pd.NaT,
    } for o in d.get("orders", {}).values()], columns=["id", "time", "symbol", "type", "status", "quantity", "price",
                                                        "value", "tag", "filled_at"])
    orders = orders.sort_values("id").reset_index(drop=True) if len(orders) else orders

    closed = pd.DataFrame([{
        "symbol": (c.get("symbols") or [{}])[0].get("value", ""),
        "side": "short" if c.get("direction") == 1 else "long",  # QuantConnect: 0 = long, 1 = short
        "entry_time": pd.Timestamp(c["entryTime"]), "exit_time": pd.Timestamp(c["exitTime"]),
        "quantity": c.get("quantity"), "entry_price": c.get("entryPrice"), "exit_price": c.get("exitPrice"),
        "pnl": c.get("profitLoss"), "fees": c.get("totalFees", 0.0), "mae_usd": c.get("mae"), "mfe_usd": c.get("mfe"),
        "duration": pd.to_timedelta(c.get("duration", "00:00:00")),
    } for c in perf.get("closedTrades", [])])
    if len(closed):
        closed["balance"] = _start_equity(portfolio, equity) + closed.pnl.cumsum()

    result = QCResult(
        name=d.get("state", {}).get("Name") or cfg.get("name", "QuantConnect backtest"),
        start=pd.Timestamp(cfg["startDate"]) if cfg.get("startDate") else equity.index[0],
        end=pd.Timestamp(cfg["endDate"]) if cfg.get("endDate") else equity.index[-1],
        status=d.get("state", {}).get("Status", ""), currency=cfg.get("accountCurrency", "USD"),
        statistics=d.get("statistics", {}), portfolio=portfolio, trade_stats=perf.get("tradeStatistics", {}),
        runtime=d.get("runtimeStatistics", {}), equity=equity, benchmark=bm,
        drawdown=_series(charts.get("Drawdown", {}), "Equity Drawdown"), exposure=exposure,
        turnover=_series(charts.get("Portfolio Turnover", {}), "Portfolio Turnover"),
        capacity=_series(charts.get("Capacity", {}), "Strategy Capacity"),
        orders=orders, closed_trades=closed, analysis=d.get("analysis", []),
    )
    result.findings = check(result)
    return result


def _start_equity(portfolio: dict, equity: pd.Series) -> float:
    return float(portfolio.get("startEquity", equity.iloc[0]))


# --- the checks -------------------------------------------------------------------------
def net_position(orders: pd.DataFrame, upto: pd.Timestamp | None = None) -> dict[str, float]:
    """Units held per symbol from filled orders (optionally only those placed before `upto`)."""
    filled = orders[orders.status.isin(["Filled", "Partially filled"])]
    if upto is not None:
        filled = filled[filled.time < upto]
    return {s: q for s, q in filled.groupby("symbol").quantity.sum().items() if abs(q) > 1e-9}


def check(r: QCResult) -> list[Finding]:
    out: list[Finding] = []
    o, ct = r.orders, r.closed_trades
    equity_end, start = r.end_equity, r.start_equity
    net_profit = equity_end - start
    runtime = {k: _num(v) for k, v in r.runtime.items()}
    unrealised = runtime.get("Unrealized", float("nan"))
    holdings = runtime.get("Holdings", float("nan"))

    # 1. Orders the broker model refused: whatever they were meant to do never happened.
    bad = o[o.status.eq("Invalid")] if len(o) else o
    for _, row in bad.iterrows():
        held = net_position(o, upto=row.time)
        meant = "close a position" if held and (row.quantity * sum(held.values()) < 0) else "enter a position"
        note = next((a.get("sample") for a in r.analysis if isinstance(a.get("sample"), str)
                     and f"[{row.id}]" in a["sample"]), "")
        out.append(Finding(
            "high", f"Order {row.id} was {row.status.lower()} and never filled",
            f"On {row.time:%Y-%m-%d %H:%M} a {row.type.lower()} order for {row.quantity:+,.0f} {row.symbol} "
            f"(tag \"{row.tag}\") was meant to {meant}, but it was {row.status.lower()}. "
            + (f"QuantConnect says: {note} " if note else "")
            + ("The position it should have closed stayed open for the rest of the backtest." if meant == "close a position" else "")))

    # Pending orders the algorithm cancelled itself are normal; just say how many.
    cancelled = o[o.status.eq("Canceled")] if len(o) else o
    if len(cancelled):
        tags = ", ".join(f'"{t}"' for t in cancelled.tag.replace("", "(no tag)").value_counts().head(3).index)
        out.append(Finding("info", f"{len(cancelled)} pending orders were cancelled before filling",
                           f"Cancelled by the algorithm, usually normal. Tags: {tags}."))

    # 2. A position still open at the end, and how much of the result it is.
    final_pos = net_position(o)
    if final_pos and len(o):
        sym, units = next(iter(final_pos.items()))
        entries = o[(o.symbol == sym) & o.status.eq("Filled")]
        opened = entries.iloc[-1].time if len(entries) else r.start
        days_open = (r.end - opened).days
        lev = holdings / equity_end if holdings == holdings and equity_end else float("nan")
        share = unrealised / net_profit * 100 if unrealised == unrealised and net_profit else float("nan")
        text = (f"At the end the account still holds {units:+,.0f} {sym}, opened {opened:%Y-%m-%d} "
                f"({days_open} days before the end)")
        if lev == lev:
            text += f", worth {holdings:,.0f} on {equity_end:,.0f} of equity ({lev:.1f}x leverage)"
        text += "."
        if share == share:
            text += (f" Unrealised profit is {unrealised:,.0f}, which is {share:.0f}% of the whole "
                     f"{net_profit:,.0f} gain.")
        sev = "high" if (share == share and share > 60) or (lev == lev and lev > 5) else "medium"
        out.append(Finding(sev, "Most of the result is a position that was never closed" if sev == "high"
                           else "A position is still open at the end", text))

    # 3. Closed trades versus the headline.
    if len(ct):
        realised = float(ct.pnl.sum())
        if realised < 0 < net_profit:
            out.append(Finding(
                "high", "The closed trades lost money, but the account gained",
                f"The {len(ct)} closed trades made {realised:,.2f} in total ({int((ct.pnl > 0).sum())} winners), "
                f"yet the account grew from {start:,.0f} to {equity_end:,.0f}. "
                "The difference is unrealised profit on a position still open."))
        gap_days = (r.end - ct.exit_time.max()).days
        total_days = max((r.end - r.start).days, 1)
        if gap_days > 0.25 * total_days:
            out.append(Finding(
                "medium", f"No trade has closed since {ct.exit_time.max():%Y-%m-%d}",
                f"The backtest runs {total_days} days, but the last closed trade is {gap_days} days before the "
                f"end ({gap_days / total_days:.0%} of the period), so the statistics on closed trades "
                "(win rate, profit factor, trade Sharpe) describe only the first part."))
        if len(ct) < 100:
            out.append(Finding("info", f"Only {len(ct)} closed trades",
                               "Too few trades to say much about the win rate or expectancy with confidence."))

    # 4. Portfolio ratios versus the benchmark.
    beta = r.portfolio.get("beta", float("nan"))
    if beta == beta and abs(beta) > 2:
        bm_ret = (r.benchmark.iloc[-1] / r.benchmark.iloc[0] - 1) * 100 if len(r.benchmark) else float("nan")
        out.append(Finding(
            "medium", f"Beta is {beta:.1f}: the account moved about {abs(beta):.0f}x its benchmark",
            f"Day to day the strategy moved {abs(beta):.1f}x as much as its benchmark"
            + (f" (the benchmark itself returned {bm_ret:+.1f}% over the period)" if bm_ret == bm_ret else "")
            + ". Alpha and the ratios built on it mostly reflect leverage, not skill."))
    p = next((a.get("sample", {}).get("pValue") for a in r.analysis
              if "StatisticalSignificance" in a.get("name", "") and isinstance(a.get("sample"), dict)), None)
    if p is not None and p > 0.05:
        out.append(Finding("medium", f"The excess return over the benchmark isn't statistically significant (p = {p:g})",
                           "QuantConnect couldn't reject the idea that the strategy's daily returns, net of the "
                           "benchmark, average zero. With p above 0.05 the outperformance could be luck."))
    psr = r.portfolio.get("probabilisticSharpeRatio", float("nan"))
    if psr == psr and psr < 0.5 and r.portfolio.get("sharpeRatio", 0) > 0:
        out.append(Finding("info", f"Probabilistic Sharpe ratio is {psr:.0%}",
                           "Even with a positive Sharpe ratio, there's less than a 50% chance the true Sharpe ratio "
                           "is above zero given the length and shape of the returns."))

    # 5. Costs and shape of the trades.
    ts = {k: _num(v) for k, v in r.trade_stats.items()}
    if ts.get("totalFees", 0) == 0 and len(ct):
        out.append(Finding("info", "No fees or commissions were charged",
                           f"Total fees are $0. With {len(ct)} trades held about "
                           f"{r.trade_stats.get('averageTradeDuration', '?')} on average, costs beyond the bid/ask "
                           "spread (commissions, financing) could change the result."))
    avg_loss, avg_win = ts.get("averageLoss", 0), ts.get("averageProfit", 0)
    if avg_win and avg_loss and abs(avg_loss) > 1.5 * avg_win and ts.get("winRate", 1) < 0.5:
        out.append(Finding("info", "Losses are about twice the size of wins, and wins are rarer",
                           f"Average win {avg_win:,.2f} against average loss {avg_loss:,.2f} with a "
                           f"{ts.get('winRate', 0):.0%} win rate: expectancy is negative on the closed trades."))

    rank = {"high": 0, "medium": 1, "info": 2}
    return sorted(out, key=lambda f: rank[f.severity])
