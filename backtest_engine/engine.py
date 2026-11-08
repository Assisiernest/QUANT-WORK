"""The event loop and the results it produces."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .data_loader import trading_day

from .broker import SimulatedBroker
from .data import BarFeed
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .portfolio import Portfolio, Trade
from .strategy import Strategy


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    history: pd.DataFrame  # per-bar portfolio tracking, indexed by bar time (UTC)
    fills: pd.DataFrame
    initial_cash: float
    title: str = "Backtest"
    meta: dict = field(default_factory=dict)  # settings used: sizing, orders, slippage, ...

    @property
    def equity(self) -> pd.Series:
        return self.history["equity"]

    def portfolio_history(self) -> pd.DataFrame:
        """Per-bar tracking: close, position, avg_price, balance, open_pnl, equity, drawdown_pct."""
        return self.history

    def daily(self) -> pd.DataFrame:
        """Portfolio at the end of each trading day (17:00 New York close)."""
        h = self.history
        g = h.groupby(trading_day(h.index))
        out = g.agg(close=("close", "last"), position=("position", "last"), balance=("balance", "last"),
                    open_pnl=("open_pnl", "last"), equity=("equity", "last"),
                    drawdown_pct=("drawdown_pct", "min"))
        out.insert(5, "day_pnl", out.equity.diff().fillna(out.equity.iloc[0] - self.initial_cash))
        exits = self.trades.groupby(trading_day(pd.DatetimeIndex(self.trades.exit_time))).size()
        out.insert(6, "trades_closed", exits.reindex(out.index, fill_value=0))
        return out

    def plot_equity(self, path=None, show: bool = False):
        """Equity curve, drawdown and price with entries. Saves a PNG if path is given."""
        from .report import plot_equity
        return plot_equity(self, path=path, show=show)

    def trade_log(self) -> pd.DataFrame:
        """Every closed trade, numbered, with duration, return on balance and running balance."""
        t = self.trades
        balance_after = self.initial_cash + t.pnl.cumsum()
        balance_before = balance_after - t.pnl
        log = pd.DataFrame({
            "side": t.side,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "duration": t.exit_time - t.entry_time,
            "bars_held": t.bars_held,
            "units": t.units,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pips": t.pips,
            "pnl": t.pnl,
            "return_pct": t.pnl / balance_before * 100,
            "spread_cost": t.spread_cost,
            "slippage_cost": t.slippage_cost,
            "mae_pips": t.mae_pips,
            "mfe_pips": t.mfe_pips,
            "balance": balance_after,
            "cum_pnl": t.pnl.cumsum(),
            "exit_reason": t.exit_reason,
        })
        if "symbol" in t:  # multi-asset results
            log.insert(0, "symbol", t.symbol)
        log.index = pd.RangeIndex(1, len(log) + 1, name="trade")
        return log

    def metrics(self) -> pd.DataFrame:
        """All performance metrics (returns, risk, trades, exposure) as one table."""
        from .metrics import metrics_frame
        return metrics_frame(self)

    def by_side(self) -> pd.DataFrame:
        """Long vs short breakdown."""
        from .metrics import by_side
        return by_side(self)

    def stats(self) -> dict[str, object]:
        """Every metric as {key: value}, e.g. stats()["sharpe"]; see metrics() for names."""
        from .metrics import compute_metrics
        return {m.key: m.value for items in compute_metrics(self).values() for m in items}


def run_backtest(feed: BarFeed, strategy: Strategy, portfolio: Portfolio, broker: SimulatedBroker,
                 title: str | None = None, trade_from: pd.Timestamp | None = None) -> BacktestResult:
    """Run the loop. For each bar, in this order:

    1. the broker fills orders at this bar's open (market) or within it (limit, stop,
       stop-loss, take-profit, trailing stop);
    2. the portfolio marks its position to this bar's close;
    3. the strategy sees the closed bar and may signal;
    4. signal -> sized orders -> queued at the broker for the next bar.
    Any open position is closed at the last bar's close. If equity falls to zero or below,
    the position is closed at that bar's close ("margin call") and no new trades are taken.

    trade_from: bars before this time only warm up the strategy (its moving averages);
    no orders, and the result starts at trade_from. Used by out-of-sample testing.
    """
    queue: deque = deque()
    bar = None
    strategy.prepare(feed.df)
    for bar in feed:
        if trade_from is not None and bar.time < trade_from:
            strategy.on_bar(bar)  # warm-up only
            continue
        queue.append(MarketEvent(bar))
        while queue:
            event = queue.popleft()
            if isinstance(event, MarketEvent):
                for fill in broker.on_bar(event.bar):
                    portfolio.on_fill(fill)
                ask = broker.ask(event.bar.close, event.bar)
                portfolio.on_bar(event.bar, ask)
                if portfolio.position and portfolio.current_equity <= 0:
                    # Account wiped out: a broker would stop you out. Close now; no new trades after this.
                    portfolio.on_fill(broker.close_out(-portfolio.position, event.bar, reason="margin call"))
                    portfolio.history[-1] = portfolio.snapshot(event.bar, ask)
                signal = strategy.on_bar(event.bar)
                if signal:
                    queue.append(signal)
            elif isinstance(event, SignalEvent):
                queue.extend(portfolio.on_signal(event))
            elif isinstance(event, OrderEvent):
                broker.place(event)

    if bar is None or not portfolio.history:
        raise ValueError("No bars to backtest")
    broker.cancel_all()
    if portfolio.position:
        portfolio.on_fill(broker.close_out(-portfolio.position, bar))
        portfolio.history[-1] = portfolio.snapshot(bar, broker.ask(bar.close, bar))

    meta = {
        "symbol": portfolio.symbol,
        "sizing": portfolio.sizer.describe(),
        "orders": portfolio.orders.describe(),
        "slippage": broker.slippage.describe(),
        "skipped_signals": portfolio.skipped_signals,
    }
    return result_from_portfolio(portfolio, title or f"{portfolio.symbol} backtest", meta)


def result_from_portfolio(portfolio: Portfolio, title: str, meta: dict) -> BacktestResult:
    """Turn a portfolio's recorded trades, fills and snapshots into a BacktestResult."""
    trades = pd.DataFrame([_asdict(t) for t in portfolio.trades], columns=list(Trade.__slots__))
    fills = pd.DataFrame([_asdict(f) for f in portfolio.fills], columns=list(FillEvent.__slots__))
    fills["notional"] = [portfolio.notional_usd(abs(u), p) for u, p in zip(fills.units, fills.price)]
    history = pd.DataFrame([_asdict(s) for s in portfolio.history]).set_index("time")
    history["drawdown_pct"] = (history.equity / history.equity.cummax() - 1) * 100
    return BacktestResult(trades, history, fills, portfolio.initial_cash, title=title, meta=meta)


def _asdict(obj) -> dict:
    return {name: getattr(obj, name) for name in obj.__slots__}


def point_size(price: float) -> float:
    """Spread point: 0.001 for 3-digit quotes (JPY pairs), 0.00001 for 5-digit ones."""
    return 0.001 if price > 20 else 0.00001
