"""Simulated broker: order types, slippage, and stop-loss / take-profit / trailing-stop exits.

Prices: MT5 bar prices are bids. Sells fill at the bid, buys at the ask (bid + spread).
Limit and stop prices are on the side the order executes at (ask for buys, bid for sells).

Each bar, in this order:
1. market orders (from the previous bar's signal) fill at the open;
2. working limit / stop entries fill if the bar reaches their price (at the open if it
   gaps through), otherwise age and may expire;
3. the open position's stop-loss / trailing stop / take-profit are checked against the
   bar's range. If the stop and the target are both inside one bar, the stop is assumed
   to have hit first (the conservative choice). Exits for a position that opened
   inside this bar (limit/stop entry) start on the next bar;
4. a trailing stop moves up (long) / down (short) using this bar's extreme.

Slippage is always against you and applies to market orders, stop entries, stop-losses
and trailing stops; limit entries and take-profits are limit orders and fill at their price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import Bar, FillEvent, OrderEvent


# --- slippage ------------------------------------------------------------------
class Slippage:
    """No slippage. Subclasses return the slippage in points for one fill."""

    def points(self, bar: Bar) -> float:
        return 0.0

    def describe(self) -> str:
        return "none"


@dataclass
class FixedSlippage(Slippage):
    """The same slippage on every fill it applies to, e.g. 2 points."""
    pts: float = 2.0

    def points(self, bar: Bar) -> float:
        return self.pts

    def describe(self) -> str:
        return f"{self.pts:g} points per fill"


class RandomSlippage(Slippage):
    """Uniform random slippage between 0 and max_points; seeded, so runs repeat exactly."""

    def __init__(self, max_points: float = 4.0, seed: int = 0):
        self.max_points, self.seed = max_points, seed
        self._rng = np.random.default_rng(seed)

    def points(self, bar: Bar) -> float:
        return float(self._rng.uniform(0, self.max_points))

    def describe(self) -> str:
        return f"random 0-{self.max_points:g} points per fill (seed {self.seed})"


# --- broker --------------------------------------------------------------------
@dataclass
class _Working:
    order: OrderEvent
    age: int = 0


@dataclass
class _Protection:
    """Exits attached to the open position. Levels are on the exit side's price
    (bid for a long's sell, ask for a short's buy)."""
    units: float  # order that closes the position (opposite sign to it)
    stop: float | None
    target: float | None
    trail: float | None
    best: float  # best bid (long) / best ask (short) since entry, for the trailing stop
    trailing: bool = False  # True once the trailing stop has become the active stop
    wait: bool = False  # position opened inside this bar: start checking next bar

    @property
    def long(self) -> bool:
        return self.units < 0


class SimulatedBroker:
    def __init__(self, point: float, slippage: Slippage | None = None):
        self.point = point  # price of one point, e.g. 0.001 for USDJPY
        self.slippage = slippage or Slippage()
        self._market: list[OrderEvent] = []
        self._working: list[_Working] = []
        self._protection: _Protection | None = None

    # --- orders ---------------------------------------------------------------
    def place(self, order: OrderEvent) -> None:
        if order.purpose == "close":
            self._protection = None
            self._working.clear()
            self._market.append(order)
            return
        self._working.clear()  # a new entry replaces any unfilled one
        if order.order_type == "market":
            self._market.append(order)
        else:
            self._working.append(_Working(order))

    def cancel_all(self) -> None:
        self._market.clear()
        self._working.clear()
        self._protection = None

    @property
    def working_orders(self) -> list[OrderEvent]:
        return [w.order for w in self._working]

    @property
    def protection(self) -> _Protection | None:
        return self._protection

    # --- each bar -------------------------------------------------------------
    def on_bar(self, bar: Bar) -> list[FillEvent]:
        fills: list[FillEvent] = []
        s = bar.spread * self.point

        # 1. Market orders at the open.
        for order in self._market:
            level = bar.open + s if order.units > 0 else bar.open
            fill = self._execute(order.units, level, bar, slip=True, reason="signal")
            fills.append(fill)
            if order.purpose == "entry":
                self._attach(order, fill, bar, inside_bar=False)
        self._market.clear()

        # 2. Working limit / stop entries.
        still: list[_Working] = []
        for w in self._working:
            hit = self._entry_trigger(w.order, bar, s)
            if hit is None:
                w.age += 1
                if w.order.expiry_bars is None or w.age < w.order.expiry_bars:
                    still.append(w)
                continue
            level, at_open = hit
            fill = self._execute(w.order.units, level, bar, slip=w.order.order_type == "stop", reason="signal")
            fills.append(fill)
            self._attach(w.order, fill, bar, inside_bar=not at_open)
        self._working = still

        # 3. Exits for the open position.
        p = self._protection
        if p is not None:
            if p.wait:
                p.wait = False
            else:
                exit_fill = self._check_exits(p, bar, s)
                if exit_fill is not None:
                    fills.append(exit_fill)
                    self._protection = None
                    return fills
            # 4. Trail the stop with this bar's extreme.
            if p.trail:
                if p.long:
                    p.best = max(p.best, bar.high)
                    self._tighten(p, p.best - p.trail)
                else:
                    p.best = min(p.best, bar.low + s)
                    self._tighten(p, p.best + p.trail)
        return fills

    def close_out(self, units: float, bar: Bar, reason: str = "end of data") -> FillEvent:
        """Fill at this bar's close; used to flatten at the end of the data (or on a margin call)."""
        self.cancel_all()
        level = bar.close + bar.spread * self.point if units > 0 else bar.close
        return self._execute(units, level, bar, slip=False, reason=reason)

    def ask(self, bid: float, bar: Bar) -> float:
        return bid + bar.spread * self.point

    # --- internals ------------------------------------------------------------
    def _execute(self, units: float, level: float, bar: Bar, slip: bool, reason: str) -> FillEvent:
        """Fill at `level` (already ask for buys / bid for sells), moved against us by slippage."""
        buy = units > 0
        slip_price = self.slippage.points(bar) * self.point if slip else 0.0
        price = level + slip_price if buy else level - slip_price
        return FillEvent(
            bar.time, units, price,
            spread_cost=abs(units) * bar.spread * self.point if buy else 0.0,
            slippage_cost=abs(units) * slip_price,
            reason=reason,
        )

    def _entry_trigger(self, order: OrderEvent, bar: Bar, s: float) -> tuple[float, bool] | None:
        """(fill level, filled at the open?) if this bar reaches the order's price."""
        p = order.price
        if order.units > 0:  # buy: compare with the ask
            o, lo, hi = bar.open + s, bar.low + s, bar.high + s
            if order.order_type == "limit":
                return (o, True) if o <= p else (p, False) if lo <= p else None
            return (o, True) if o >= p else (p, False) if hi >= p else None
        o, lo, hi = bar.open, bar.low, bar.high  # sell: compare with the bid
        if order.order_type == "limit":
            return (o, True) if o >= p else (p, False) if hi >= p else None
        return (o, True) if o <= p else (p, False) if lo <= p else None

    def _attach(self, order: OrderEvent, fill: FillEvent, bar: Bar, inside_bar: bool) -> None:
        if not (order.stop_loss or order.take_profit or order.trailing_stop):
            self._protection = None
            return
        long = fill.units > 0
        sign = 1 if long else -1
        s = bar.spread * self.point
        # Exit-side price at entry: the bid for a long (we bought at the ask), the ask for a short.
        exit_side_now = fill.price - s if long else fill.price + s
        p = _Protection(
            units=-fill.units,
            stop=fill.price - sign * order.stop_loss if order.stop_loss else None,
            target=fill.price + sign * order.take_profit if order.take_profit else None,
            trail=order.trailing_stop,
            best=exit_side_now,
            wait=inside_bar,
        )
        if p.trail:
            self._tighten(p, p.best - sign * p.trail)
        self._protection = p

    @staticmethod
    def _tighten(p: _Protection, level: float) -> None:
        """Move the stop to `level` if that's tighter; a trailing stop never loosens."""
        if p.stop is None or (level > p.stop if p.long else level < p.stop):
            p.stop, p.trailing = level, True

    def _check_exits(self, p: _Protection, bar: Bar, s: float) -> FillEvent | None:
        stop_reason = "trailing stop" if p.trailing else "stop loss"
        if p.long:  # sell at the bid
            o, lo, hi = bar.open, bar.low, bar.high
            if p.stop is not None and o <= p.stop:
                return self._execute(p.units, o, bar, slip=True, reason=stop_reason)
            if p.target is not None and o >= p.target:
                return self._execute(p.units, o, bar, slip=False, reason="take profit")
            if p.stop is not None and lo <= p.stop:
                return self._execute(p.units, p.stop, bar, slip=True, reason=stop_reason)
            if p.target is not None and hi >= p.target:
                return self._execute(p.units, p.target, bar, slip=False, reason="take profit")
            return None
        o, lo, hi = bar.open + s, bar.low + s, bar.high + s  # short: buy at the ask
        if p.stop is not None and o >= p.stop:
            return self._execute(p.units, o, bar, slip=True, reason=stop_reason)
        if p.target is not None and o <= p.target:
            return self._execute(p.units, o, bar, slip=False, reason="take profit")
        if p.stop is not None and hi >= p.stop:
            return self._execute(p.units, p.stop, bar, slip=True, reason=stop_reason)
        if p.target is not None and lo <= p.target:
            return self._execute(p.units, p.target, bar, slip=False, reason="take profit")
        return None
