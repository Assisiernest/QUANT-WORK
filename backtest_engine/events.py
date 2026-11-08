"""Events passed around the backtest loop.

Flow per bar:  MarketEvent -> (strategy) SignalEvent -> (portfolio) OrderEvent
               -> (broker, at the NEXT bar's open) FillEvent -> (portfolio)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Bar:
    time: pd.Timestamp  # bar open time (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0  # in points, as MT5 exports it


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A bar has closed."""
    bar: Bar


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """The strategy's wanted position: +1 long, -1 short, 0 flat."""
    time: pd.Timestamp
    target: int


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An order for a signed number of units (+ buy, - sell).

    order_type: "market" fills at the next bar's open; "limit" and "stop" wait for `price`
    (compared with the ask for buys, the bid for sells) until filled, cancelled or expired.
    purpose: "entry" opens a position; "close" flattens it (and cancels its stop/target).
    stop_loss / take_profit / trailing_stop: price distances from the entry fill, attached
    to the position once the entry fills.
    """
    time: pd.Timestamp
    units: float
    order_type: str = "market"
    price: float | None = None
    purpose: str = "entry"
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    expiry_bars: int | None = None  # limit/stop entries: cancel if not filled within this many bars


@dataclass(frozen=True, slots=True)
class FillEvent:
    time: pd.Timestamp
    units: float  # signed
    price: float  # actual fill price, spread and slippage included
    spread_cost: float = 0.0  # spread paid on this fill, quote currency (buys only: they fill at the ask)
    slippage_cost: float = 0.0  # slippage paid on this fill, quote currency
    reason: str = "signal"  # signal, stop loss, take profit, trailing stop, end of data
