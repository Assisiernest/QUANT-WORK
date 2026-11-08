"""Portfolio: turns signals into sized orders, books fills, tracks cash, trades and equity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .events import Bar, FillEvent, OrderEvent, SignalEvent
from .orders import OrderPolicy
from .sizing import FixedUnits, Sizer


CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK", "SGD", "HKD",
              "MXN", "ZAR", "TRY", "CNH", "PLN", "HUF", "CZK"}


def is_fx_pair(symbol: str) -> bool:
    """True for six-letter currency pairs like EURGBP; False for AAPL, US500, BTC-USD, XAUUSD...
    (metals are priced in USD but aren't currencies: no micro-lot rounding, size in ounces.)"""
    s = symbol.upper()
    return len(s) == 6 and s[:3] in CURRENCIES and s[3:] in CURRENCIES


@dataclass(frozen=True, slots=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "long" / "short"
    units: float
    entry_price: float
    exit_price: float
    pips: float
    pnl: float  # account currency, spread already included (it's in the fill prices)
    spread_cost: float  # spread paid on entry + exit, USD
    slippage_cost: float  # slippage paid on entry + exit, USD
    mae_pips: float  # max adverse excursion: worst open loss while held (<= 0)
    mfe_pips: float  # max favourable excursion: best open profit while held (>= 0)
    mae_usd: float  # the same two excursions in USD
    mfe_usd: float
    bars_held: int
    exit_reason: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The account at one bar's close."""
    time: pd.Timestamp
    close: float  # bar close (bid)
    position: float  # signed units held
    avg_price: float  # entry price of the open position, NaN when flat
    balance: float  # cash: start + closed-trade P&L (USD)
    open_pnl: float  # unrealised P&L of the open position (USD)
    equity: float  # balance + open_pnl


class Portfolio:
    """One symbol, account in USD.

    sizer decides how many units each entry trades (default: `units` every time);
    orders (an OrderPolicy) decides the entry order type and the stop-loss /
    take-profit / trailing stop attached to it (default: market orders, no exits).

    P&L is earned in the quote currency (JPY for USDJPY) and converted to USD at
    the exit price: divided by the price for USD-base pairs (USDJPY), unchanged for
    USD-quote pairs (EURUSD).
    """

    def __init__(self, symbol: str, cash: float = 10_000.0, units: float = 10_000.0, point: float = 0.001,
                 sizer: Sizer | None = None, orders: OrderPolicy | None = None):
        symbol = symbol.upper()
        if symbol.endswith("USD") or not is_fx_pair(symbol):
            # USD-quoted pairs (EURUSD, XAUUSD) and anything that isn't a currency pair
            # (stocks, indices, crypto in USD): P&L is already in USD.
            self._to_usd = lambda pnl, price: pnl
            self._notional = lambda units, price: units * price
        elif symbol.startswith("USD"):
            self._to_usd = lambda pnl, price: pnl / price
            self._notional = lambda units, price: units
        else:
            raise ValueError(f"{symbol}: currency pairs need USD as base or quote (no crosses like EURGBP yet)")
        self.symbol, self.initial_cash, self.cash, self.units = symbol, cash, cash, units
        self.pip = point * 10  # 1 pip = 10 points on 3/5-digit quotes
        # Default: exactly `units` every trade, no rounding or leverage cap.
        self.sizer = sizer or FixedUnits(units=units, lot_step=1, min_units=1, max_leverage=math.inf)
        self.orders = orders or OrderPolicy()
        self._bid = self._ask = None  # last bar's close, for sizing and limit/stop prices
        self.skipped_signals = 0  # entries the sizer sized to zero

        self.position = 0.0
        self.avg_price = 0.0
        self.entry_time: pd.Timestamp | None = None
        # Open-trade bookkeeping for the trade log.
        self._entry_cost = 0.0  # spread paid entering the open position, quote currency
        self._entry_slip = 0.0  # slippage paid entering the open position, quote currency
        self._best = self._worst = 0.0  # best / worst open P&L seen, in price units per unit held
        self._bars_held = 0
        self.trades: list[Trade] = []
        self.fills: list[FillEvent] = []
        # One snapshot per bar, taken at the bar's close (see Snapshot).
        self.history: list[Snapshot] = []

    # --- events -----------------------------------------------------------
    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        """Close the current position if the signal disagrees with it, then place a sized entry."""
        side = signal.target
        held = (self.position > 0) - (self.position < 0)
        if held == side:
            return []
        orders = []
        if self.position:
            orders.append(OrderEvent(signal.time, -self.position, purpose="close"))
        if side == 0:
            return orders
        if self.current_equity <= 0:  # account wiped out: no new positions
            self.skipped_signals += 1
            return orders

        pol, pip = self.orders, self.pip
        dist = lambda pips: pips * pip if pips else None  # noqa: E731
        stop, target, trail = dist(pol.stop_loss_pips), dist(pol.take_profit_pips), dist(pol.trailing_stop_pips)
        ref = self._ask if side > 0 else self._bid  # price we'd enter at now
        units = self.sizer.size(self.current_equity, ref, stop or trail, self)
        if not units:
            self.skipped_signals += 1
            return orders
        price = None
        if pol.entry == "limit":
            price = ref - side * pol.entry_offset_pips * pip
        elif pol.entry == "stop":
            price = ref + side * pol.entry_offset_pips * pip
        orders.append(OrderEvent(signal.time, side * units, order_type=pol.entry, price=price, purpose="entry",
                                 stop_loss=stop, take_profit=target, trailing_stop=trail,
                                 expiry_bars=pol.expiry_bars))
        return orders

    def on_fill(self, fill: FillEvent) -> None:
        self.fills.append(fill)
        units = fill.units
        if self.position and (units > 0) != (self.position > 0):
            closing = min(abs(units), abs(self.position))
            self._book_close(closing, fill)
            units += closing if units < 0 else -closing
        if units:
            if not self.position:
                self.entry_time, self.avg_price = fill.time, fill.price
                self._best = self._worst = 0.0
                self._bars_held = 0
            else:
                total = abs(self.position) + abs(units)
                self.avg_price = (self.avg_price * abs(self.position) + fill.price * abs(units)) / total
            self._entry_cost += fill.spread_cost * abs(units) / abs(fill.units)
            self._entry_slip += fill.slippage_cost * abs(units) / abs(fill.units)
            self.position += units

    def on_bar(self, bar: Bar, ask: float) -> None:
        """Record where the account stands at this bar's close."""
        self._bid, self._ask = bar.close, ask
        if self.position:
            # Excursions over the bar's range, at the price we could close at (bid long, ask short).
            spread = ask - bar.close
            if self.position > 0:
                best, worst = bar.high - self.avg_price, bar.low - self.avg_price
            else:
                best, worst = self.avg_price - (bar.low + spread), self.avg_price - (bar.high + spread)
            self._best, self._worst = max(self._best, best), min(self._worst, worst)
            self._bars_held += 1
        self.history.append(self.snapshot(bar, ask))

    # --- helpers ----------------------------------------------------------
    @property
    def current_equity(self) -> float:
        return self.history[-1].equity if self.history else self.cash

    def usd_value(self, price_move: float, price: float) -> float:
        """USD value of a price move of `price_move` on one unit, at `price`."""
        return self._to_usd(price_move, price)

    def notional_usd(self, units: float, price: float) -> float:
        """USD size of a position of `units` at `price`."""
        return self._notional(units, price)

    def snapshot(self, bar: Bar, ask: float) -> "Snapshot":
        """Mark to market at the price we'd get closing now (bid for longs, ask for shorts)."""
        equity = self.equity_at(bar.close, ask)
        return Snapshot(
            time=bar.time, close=bar.close, position=self.position,
            avg_price=self.avg_price if self.position else float("nan"),
            balance=self.cash, open_pnl=equity - self.cash, equity=equity,
        )

    def equity_at(self, bid: float, ask: float) -> float:
        if not self.position:
            return self.cash
        price = bid if self.position > 0 else ask
        return self.cash + self._to_usd(self.position * (price - self.avg_price), price)

    def _book_close(self, qty: float, fill: FillEvent) -> None:
        direction = 1 if self.position > 0 else -1
        move = direction * (fill.price - self.avg_price)
        pnl = self._to_usd(qty * move, fill.price)
        self.cash += pnl
        entry_cost = self._entry_cost * qty / abs(self.position)
        exit_cost = fill.spread_cost * qty / abs(fill.units)
        entry_slip = self._entry_slip * qty / abs(self.position)
        exit_slip = fill.slippage_cost * qty / abs(fill.units)
        self.trades.append(Trade(
            entry_time=self.entry_time, exit_time=fill.time,
            side="long" if direction > 0 else "short", units=qty,
            entry_price=self.avg_price, exit_price=fill.price,
            pips=move / self.pip, pnl=pnl,
            spread_cost=self._to_usd(entry_cost + exit_cost, fill.price),
            slippage_cost=self._to_usd(entry_slip + exit_slip, fill.price),
            mae_pips=min(self._worst, move) / self.pip, mfe_pips=max(self._best, move) / self.pip,
            mae_usd=self._to_usd(qty * min(self._worst, move), fill.price),
            mfe_usd=self._to_usd(qty * max(self._best, move), fill.price),
            bars_held=self._bars_held, exit_reason=fill.reason,
        ))
        self._entry_cost -= entry_cost
        self._entry_slip -= entry_slip
        self.position -= direction * qty
        if not self.position:
            self.avg_price, self.entry_time, self._entry_cost, self._entry_slip = 0.0, None, 0.0, 0.0
