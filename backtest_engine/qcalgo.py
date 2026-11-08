"""Run QuantConnect-style Python algorithms on your own CSV data.

Paste an algorithm written for QuantConnect (LEAN) and run it here:

    from AlgorithmImports import *

    class EmaCross(QCAlgorithm):
        def initialize(self):
            self.set_start_date(2025, 12, 18)
            self.set_cash(10000)
            self.pair = self.add_forex("USDJPY", Resolution.MINUTE).symbol
            self.fast = self.ema(self.pair, 20)
            self.slow = self.ema(self.pair, 50)

        def on_data(self, data):
            if not self.slow.is_ready:
                return
            if self.fast.current.value > self.slow.current.value and not self.portfolio[self.pair].is_long:
                self.set_holdings(self.pair, 1)
            elif self.fast.current.value < self.slow.current.value and not self.portfolio[self.pair].is_short:
                self.set_holdings(self.pair, -1)

    run = run_algorithm(code, assets)        # assets: the uploaded CSVs, matched by symbol
    run.result.combined.stats()["sharpe"]    # the same results as any other backtest here

What's covered: initialize / on_data (snake_case or PascalCase), add_forex / add_equity /
add_crypto / add_cfd, set_start_date / set_end_date / set_cash / set_warm_up / set_time_zone,
set_holdings, liquidate, market / limit / stop market / stop limit / market-on-open orders,
the portfolio and securities objects, transactions (open orders, cancel), history, consolidate,
plot, debug / log, on_order_event, on_end_of_algorithm, RollingWindow and the indicators
SMA, EMA, RSI, BB, ATR, MACD, STD, MAX, MIN, ROC, MOM. Calls that only configure QuantConnect's
cloud (brokerage model, benchmark, fee models...) are accepted and ignored. Anything else (scheduled
events, universe selection, options, the algorithm framework) raises a clear "not supported" error.

How it fills (like a QuantConnect backtest): market orders fill immediately at the current bar's
close, buys at the ask (close + spread) and sells at the bid; limit / stop orders fill on later
bars when the price reaches them (at the open if it gaps through); market-on-open orders fill at
the next bar's open. One shared account for every asset. self.time is in the algorithm's time zone
(New York unless set_time_zone says otherwise), as on QuantConnect. Positions still open at the end
are closed at the last price so they appear in the trade log ("end of data").
"""

from __future__ import annotations

import math
import re
import sys
import traceback
import types
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from .broker import Slippage
from .engine import point_size, result_from_portfolio
from .events import Bar, FillEvent
from .multi import Asset, MultiAssetResult, _combine
from .portfolio import Portfolio

SOURCE_NAME = "<your algorithm>"
DEFAULT_CASH = 100_000.0  # QuantConnect's default when set_cash isn't called
MAX_LOG_LINES = 5_000


class AlgorithmError(RuntimeError):
    """A problem in the pasted algorithm, with the line number in its source when known."""


def _snake(name: str) -> str:
    if name.isupper():
        return name.lower()  # SMA -> sma
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()  # SetHoldings -> set_holdings


class _Pascal:
    """Lets QuantConnect's older PascalCase names reach the snake_case attributes."""

    def __getattr__(self, name: str):
        snake = _snake(name)
        if snake != name:
            try:
                return object.__getattribute__(self, snake)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")


# --- enum stand-ins ----------------------------------------------------------------------
class _Const(str):
    """A named constant that compares case-insensitively, so Resolution.Minute == Resolution.MINUTE."""

    def __eq__(self, other):
        return isinstance(other, str) and self.upper() == other.upper()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.upper())


class _Names:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item: str) -> _Const:
        if item.startswith("__"):
            raise AttributeError(item)
        return _Const(item)

    def __repr__(self):
        return self._name


ENUMS = ["Resolution", "Market", "OrderStatus", "OrderDirection", "OrderType", "MovingAverageType",
         "BrokerageName", "AccountType", "DataNormalizationMode", "SecurityType", "Field", "InsightDirection"]
_enums = {name: _Names(name) for name in ENUMS}
Resolution, OrderStatus = _enums["Resolution"], _enums["OrderStatus"]
RESOLUTION_SPAN = {"SECOND": timedelta(seconds=1), "MINUTE": timedelta(minutes=1), "HOUR": timedelta(hours=1),
                   "DAILY": timedelta(days=1)}


class Symbol(str):
    @property
    def value(self) -> str:
        return str(self)

    Value = value

    def __repr__(self):
        return str(self)


# --- data objects ---------------------------------------------------------------------------
@dataclass
class TradeBar(_Pascal):
    symbol: Symbol
    time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    period: timedelta = timedelta(minutes=1)

    @property
    def price(self) -> float:
        return self.close

    @property
    def value(self) -> float:
        return self.close


class Slice(_Pascal):
    """The bars that closed at this time step, by symbol: data[symbol].close, symbol in data."""

    def __init__(self, time: datetime, bars: dict[str, TradeBar]):
        self.time = time
        self.bars = bars
        self.quote_bars = bars

    def _key(self, s) -> str:
        return str(s).upper()

    def __getitem__(self, s) -> TradeBar:
        return self.bars[self._key(s)]

    def __contains__(self, s) -> bool:
        return self._key(s) in self.bars

    def contains_key(self, s) -> bool:
        return s in self

    def get(self, s, default=None):
        return self.bars.get(self._key(s), default)

    def keys(self):
        return [Symbol(k) for k in self.bars]

    def values(self):
        return list(self.bars.values())

    def items(self):
        return [(Symbol(k), v) for k, v in self.bars.items()]

    def __len__(self):
        return len(self.bars)


class _Anything(_Pascal):
    """Stand-in for configuration objects (settings, universe_settings): accepts any attribute."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class RollingWindow(_Pascal):
    """Most recent first: window[0] is the latest value added."""

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, size: int):
        self.size = int(size)
        self._items: deque = deque(maxlen=self.size)
        self.samples = 0

    def add(self, item) -> None:
        self._items.appendleft(item)
        self.samples += 1

    def __getitem__(self, i):
        if i >= len(self._items):
            raise IndexError(f"RollingWindow has {len(self._items)} items, asked for [{i}]")
        return self._items[i]

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def is_ready(self) -> bool:
        return len(self._items) == self.size

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def reset(self) -> None:
        self._items.clear()


# --- indicators -------------------------------------------------------------------------------
class IndicatorDataPoint(_Pascal):
    def __init__(self, value: float = 0.0, time=None):
        self.value, self.time = value, time

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"{self.value}"


class Indicator(_Pascal):
    uses_bars = False  # True: updated with whole bars (ATR); False: with one number (the close)

    def __init__(self, *args):
        nums = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        names = [a for a in args if isinstance(a, str)]
        self.name = names[0] if names else type(self).__name__
        self.period = int(nums[0]) if nums else 1
        self.samples = 0
        self.current = IndicatorDataPoint()
        self.window: RollingWindow = RollingWindow(max(self.period, 2))

    @property
    def is_ready(self) -> bool:
        return self.samples >= self.period

    @property
    def value(self) -> float:
        return self.current.value

    def update(self, time, value=None) -> bool:
        """update(time, value), update(bar) or update(IndicatorDataPoint), as on QuantConnect."""
        if value is None and hasattr(time, "close"):
            bar, t = time, getattr(time, "end_time", None)
            x = bar if self.uses_bars else bar.close
        elif value is None and hasattr(time, "value"):
            t, x = time.time, time.value
        else:
            t, x = time, (value.close if hasattr(value, "close") and not self.uses_bars else value)
        self.samples += 1
        out = self._compute(x)
        if out is not None and not (isinstance(out, float) and math.isnan(out)):
            self.current = IndicatorDataPoint(float(out), t)
        self.window.add(self.current.value)
        return self.is_ready

    def reset(self) -> None:
        self.__init__(self.name, self.period)

    def _compute(self, x):
        raise NotImplementedError

    def __float__(self):
        return float(self.current.value)

    def _cmp(self, other):
        return float(other) if not isinstance(other, (int, float)) else other

    def __lt__(self, o): return float(self) < self._cmp(o)  # noqa: E704
    def __gt__(self, o): return float(self) > self._cmp(o)  # noqa: E704
    def __le__(self, o): return float(self) <= self._cmp(o)  # noqa: E704
    def __ge__(self, o): return float(self) >= self._cmp(o)  # noqa: E704


class SimpleMovingAverage(Indicator):
    def __init__(self, *args):
        super().__init__(*args)
        self._buf: deque = deque(maxlen=self.period)

    def _compute(self, x):
        self._buf.append(float(x))
        return sum(self._buf) / len(self._buf)


class ExponentialMovingAverage(Indicator):
    """Seeded with the simple average of the first `period` values, as QuantConnect does."""

    def __init__(self, *args):
        super().__init__(*args)
        self.k = 2 / (self.period + 1)
        self._seed: list[float] = []

    def _compute(self, x):
        x = float(x)
        if self.samples <= self.period:
            self._seed.append(x)
            return sum(self._seed) / len(self._seed)
        return self.current.value + self.k * (x - self.current.value)


class RelativeStrengthIndex(Indicator):
    """Wilder's RSI (QuantConnect's default smoothing)."""

    def __init__(self, *args):
        super().__init__(*args)
        self._prev = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._ag = self._al = None

    @property
    def is_ready(self) -> bool:
        return self.samples > self.period

    def _compute(self, x):
        x = float(x)
        if self._prev is None:
            self._prev = x
            return None
        change, self._prev = x - self._prev, x
        gain, loss = max(change, 0.0), max(-change, 0.0)
        if self._ag is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < self.period:
                return None
            self._ag, self._al = sum(self._gains) / self.period, sum(self._losses) / self.period
        else:
            n = self.period
            self._ag = (self._ag * (n - 1) + gain) / n
            self._al = (self._al * (n - 1) + loss) / n
        return 100.0 if self._al == 0 else 100 - 100 / (1 + self._ag / self._al)


class StandardDeviation(Indicator):
    def __init__(self, *args):
        super().__init__(*args)
        self._buf: deque = deque(maxlen=self.period)

    def _compute(self, x):
        self._buf.append(float(x))
        return float(np.std(self._buf))  # population, as QuantConnect


class _Sub(_Pascal):
    """A component of a composite indicator (BB.upper_band, MACD.signal...)."""

    def __init__(self):
        self.current = IndicatorDataPoint()

    def _set(self, value, t):
        self.current = IndicatorDataPoint(float(value), t)

    @property
    def value(self) -> float:
        return self.current.value

    def __float__(self):
        return float(self.current.value)


class BollingerBands(Indicator):
    def __init__(self, *args):
        super().__init__(*args)
        nums = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        self.k = float(nums[1]) if len(nums) > 1 else 2.0
        self._buf: deque = deque(maxlen=self.period)
        self.middle_band, self.upper_band, self.lower_band = _Sub(), _Sub(), _Sub()
        self.standard_deviation, self.band_width, self.percent_b = _Sub(), _Sub(), _Sub()

    def _compute(self, x):
        self._buf.append(float(x))
        mid, sd = float(np.mean(self._buf)), float(np.std(self._buf))
        up, lo = mid + self.k * sd, mid - self.k * sd
        t = None
        for sub, v in [(self.middle_band, mid), (self.upper_band, up), (self.lower_band, lo),
                       (self.standard_deviation, sd), (self.band_width, (up - lo) / mid if mid else 0.0),
                       (self.percent_b, (float(x) - lo) / (up - lo) if up != lo else 0.5)]:
            sub._set(v, t)
        return mid


class AverageTrueRange(Indicator):
    """Wilder-smoothed true range."""
    uses_bars = True

    def __init__(self, *args):
        super().__init__(*args)
        self._prev_close = None
        self._trs: list[float] = []

    def _compute(self, bar):
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        tr = h - l if self._prev_close is None else max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        self._prev_close = c
        if self.samples <= self.period:
            self._trs.append(tr)
            return sum(self._trs) / len(self._trs)
        return (self.current.value * (self.period - 1) + tr) / self.period


class MovingAverageConvergenceDivergence(Indicator):
    def __init__(self, *args):
        nums = [int(a) for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        fast, slow, sig = (nums + [12, 26, 9][len(nums):])[:3]
        super().__init__(slow)
        self.fast, self.slow = ExponentialMovingAverage(fast), ExponentialMovingAverage(slow)
        self._signal = ExponentialMovingAverage(sig)
        self.signal, self.histogram = _Sub(), _Sub()
        self.fast_period, self.slow_period, self.signal_period = fast, slow, sig

    @property
    def is_ready(self) -> bool:
        return self.slow.is_ready and self._signal.is_ready

    def _compute(self, x):
        self.fast.update(None, x)
        self.slow.update(None, x)
        line = self.fast.current.value - self.slow.current.value
        if self.slow.is_ready:
            self._signal.update(None, line)
            self.signal._set(self._signal.current.value, None)
            self.histogram._set(line - self._signal.current.value, None)
        return line


class Maximum(Indicator):
    uses_bars = True  # highest high, as QuantConnect's MAX on trade bars

    def __init__(self, *args):
        super().__init__(*args)
        self._buf: deque = deque(maxlen=self.period)

    def _compute(self, bar):
        self._buf.append(float(bar.high) if hasattr(bar, "high") else float(bar))
        return max(self._buf)


class Minimum(Maximum):
    def _compute(self, bar):
        self._buf.append(float(bar.low) if hasattr(bar, "low") else float(bar))
        return min(self._buf)


class RateOfChange(Indicator):
    def __init__(self, *args):
        super().__init__(*args)
        self._buf: deque = deque(maxlen=self.period + 1)

    @property
    def is_ready(self) -> bool:
        return self.samples > self.period

    def _compute(self, x):
        self._buf.append(float(x))
        return (self._buf[-1] - self._buf[0]) / self._buf[0] if len(self._buf) > self.period and self._buf[0] else 0.0


class Momentum(RateOfChange):
    def _compute(self, x):
        self._buf.append(float(x))
        return self._buf[-1] - self._buf[0] if len(self._buf) > self.period else 0.0


INDICATORS = {"sma": SimpleMovingAverage, "ema": ExponentialMovingAverage, "rsi": RelativeStrengthIndex,
              "bb": BollingerBands, "atr": AverageTrueRange, "macd": MovingAverageConvergenceDivergence,
              "std": StandardDeviation, "max": Maximum, "min": Minimum, "roc": RateOfChange, "mom": Momentum}


# --- securities, holdings, orders --------------------------------------------------------------
class Security(_Pascal):
    def __init__(self, symbol: Symbol, kind: str, df: pd.DataFrame, point: float, lot: float):
        self.symbol, self.type, self.df, self.point, self.lot = symbol, kind, df, point, lot
        self.bar: TradeBar | None = None
        self.spread = 0.0  # price units, from the bar's spread column
        self.leverage = None
        steps = df.index.to_series().diff().dropna()
        self.period = steps.median().to_pytimedelta() if len(steps) else timedelta(minutes=1)

    @property
    def price(self) -> float:
        return self.bar.close if self.bar else 0.0

    @property
    def close(self) -> float:
        return self.price

    @property
    def open(self) -> float:
        return self.bar.open if self.bar else 0.0

    @property
    def high(self) -> float:
        return self.bar.high if self.bar else 0.0

    @property
    def low(self) -> float:
        return self.bar.low if self.bar else 0.0

    @property
    def ask_price(self) -> float:
        return self.price + self.spread

    @property
    def bid_price(self) -> float:
        return self.price

    @property
    def has_data(self) -> bool:
        return self.bar is not None

    @property
    def is_tradable(self) -> bool:
        return self.bar is not None

    def set_leverage(self, leverage) -> None:
        self.leverage = leverage

    def __getattr__(self, name):  # set_fee_model, set_slippage_model, set_data_normalization_mode, ...
        if name.startswith("set_") or name.startswith("Set"):
            return lambda *a, **k: None
        return _Pascal.__getattr__(self, name)


class SecurityManager(dict):
    def __getitem__(self, s) -> Security:
        return dict.__getitem__(self, str(s).upper())

    def __contains__(self, s) -> bool:
        return dict.__contains__(self, str(s).upper())

    def contains_key(self, s) -> bool:
        return s in self


class Holding(_Pascal):
    def __init__(self, algo: "QCAlgorithm", symbol: str):
        self._algo, self.symbol = algo, Symbol(symbol)

    @property
    def _book(self) -> Portfolio:
        return self._algo._books[self.symbol]

    @property
    def quantity(self) -> float:
        return self._book.position

    @property
    def absolute_quantity(self) -> float:
        return abs(self.quantity)

    @property
    def invested(self) -> bool:
        return self.quantity != 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def average_price(self) -> float:
        return self._book.avg_price if self.quantity else 0.0

    @property
    def price(self) -> float:
        return self._algo.securities[self.symbol].price

    @property
    def unrealized_profit(self) -> float:
        sec = self._algo.securities[self.symbol]
        return self._book.equity_at(sec.price, sec.ask_price) - self._book.cash if sec.bar else 0.0

    @property
    def holdings_value(self) -> float:
        return math.copysign(self._book.notional_usd(abs(self.quantity), self.price), self.quantity) if self.quantity else 0.0

    @property
    def absolute_holdings_value(self) -> float:
        return abs(self.holdings_value)


class PortfolioManager(_Pascal):
    def __init__(self, algo: "QCAlgorithm"):
        self._algo = algo

    def __getitem__(self, s) -> Holding:
        key = str(s).upper()
        if key not in self._algo._books:
            raise KeyError(f"{s} hasn't been added (add_forex / add_equity ... in initialize)")
        return Holding(self._algo, key)

    def __contains__(self, s) -> bool:
        return str(s).upper() in self._algo._books

    def keys(self):
        return [Symbol(s) for s in self._algo._books]

    def values(self):
        return [Holding(self._algo, s) for s in self._algo._books]

    def items(self):
        return [(Symbol(s), Holding(self._algo, s)) for s in self._algo._books]

    @property
    def cash(self) -> float:
        return self._algo._cash_total()

    @property
    def total_portfolio_value(self) -> float:
        return self._algo._equity_total()

    @property
    def total_unrealized_profit(self) -> float:
        return sum(h.unrealized_profit for h in self.values())

    @property
    def total_holdings_value(self) -> float:
        return sum(h.absolute_holdings_value for h in self.values())

    @property
    def invested(self) -> bool:
        return any(h.invested for h in self.values())

    @property
    def margin_remaining(self) -> float:
        return self.total_portfolio_value - self.total_holdings_value  # no margin model: a rough guide


@dataclass
class _Order:
    id: int
    symbol: str
    quantity: float
    type: str  # market, limit, stop market, stop limit, market on open
    time: datetime
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""
    status: str = "Submitted"
    fill_price: float | None = None
    stop_triggered: bool = False


class OrderTicket(_Pascal):
    def __init__(self, algo: "QCAlgorithm", order: _Order):
        self._algo, self._order = algo, order

    @property
    def order_id(self) -> int:
        return self._order.id

    @property
    def status(self) -> _Const:
        return _Const(self._order.status.replace(" ", ""))

    @property
    def quantity(self) -> float:
        return self._order.quantity

    @property
    def quantity_filled(self) -> float:
        return self._order.quantity if self._order.status == "Filled" else 0.0

    @property
    def average_fill_price(self) -> float:
        return self._order.fill_price or 0.0

    @property
    def symbol(self) -> Symbol:
        return Symbol(self._order.symbol)

    @property
    def tag(self) -> str:
        return self._order.tag

    def cancel(self, tag: str = "") -> None:
        self._algo._cancel(self._order)


class OrderEvent(_Pascal):
    def __init__(self, order: _Order, fill_quantity: float, fill_price: float, time):
        self.order_id, self.symbol = order.id, Symbol(order.symbol)
        self.status = _Const(order.status.replace(" ", ""))
        self.fill_quantity, self.fill_price, self.utc_time = fill_quantity, fill_price, time
        self.quantity = order.quantity
        self.direction = _Const("Buy" if order.quantity > 0 else "Sell")
        self.is_assignment = False
        self.message = order.tag


class Transactions(_Pascal):
    def __init__(self, algo: "QCAlgorithm"):
        self._algo = algo

    def get_open_orders(self, symbol=None) -> list[OrderTicket]:
        return [OrderTicket(self._algo, o) for o in self._algo._open_orders
                if symbol is None or o.symbol == str(symbol).upper()]

    def cancel_open_orders(self, symbol=None, tag: str = "") -> list[OrderTicket]:
        out = self.get_open_orders(symbol)
        for t in out:
            t.cancel()
        return out

    def get_order_by_id(self, order_id: int):
        return next((OrderTicket(self._algo, o) for o in self._algo._all_orders if o.id == order_id), None)


# --- the algorithm base class -----------------------------------------------------------------
class QCAlgorithm(_Pascal):
    """Base class for pasted algorithms. Subclasses define initialize() and on_data()."""

    def __init__(self):
        self.securities = SecurityManager()
        self.portfolio = PortfolioManager(self)
        self.transactions = Transactions(self)
        self.settings = _Anything()
        self.universe_settings = _Anything()
        self.time = datetime(2000, 1, 1)
        self.utc_time = self.time
        self.is_warming_up = False
        self.live_mode = False
        self._cash = None
        self._start = self._end = None
        self._warmup = None
        self._tz = "America/New_York"
        self._datasets: dict[str, Asset] = {}
        self._books: dict[str, Portfolio] = {}
        self._indicators: dict[str, list[Indicator]] = {}
        self._consolidators: list[dict] = []
        self._open_orders: list[_Order] = []
        self._all_orders: list[_Order] = []
        self._next_id = 1
        self._logs: list[str] = []
        self._plots: dict[str, dict[str, list[tuple]]] = {}
        self._quit = False
        self._slippage = Slippage()

    # --- setup ------------------------------------------------------------------------------
    def set_start_date(self, *args) -> None:
        self._start = _to_dt(*args)

    def set_end_date(self, *args) -> None:
        self._end = _to_dt(*args)

    def set_cash(self, *args) -> None:
        amount = [a for a in args if isinstance(a, (int, float))]
        if amount:
            self._cash = float(amount[0])
        currency = [a for a in args if isinstance(a, str)]
        if currency and currency[0].upper() != "USD":
            self.debug(f"set_cash: only USD accounts are supported; treating {currency[0]} as USD")

    def set_warm_up(self, period, resolution=None) -> None:
        self._warmup = period if isinstance(period, timedelta) else int(period)

    def set_time_zone(self, tz) -> None:
        self._tz = str(tz)

    def set_account_currency(self, currency, *args) -> None:
        if str(currency).upper() != "USD":
            raise AlgorithmError("Only USD accounts are supported")

    def _ignore(self, *args, **kwargs) -> None:
        return None

    set_benchmark = set_brokerage_model = set_security_initializer = set_risk_free_interest_rate_model = _ignore
    add_risk_management = set_name = set_tags = _ignore

    def _add(self, ticker, kind: str) -> Security:
        key = str(ticker).upper().replace("/", "").replace("-", "")
        asset = self._datasets.get(key) or self._datasets.get(str(ticker).upper())
        if asset is None:
            have = ", ".join(sorted(self._datasets)) or "none"
            raise AlgorithmError(f"The algorithm adds {ticker}, but there's no uploaded data for it. "
                                 f"Upload a CSV and set its symbol to {key} (uploaded: {have}).")
        if key in self.securities:
            return self.securities[key]
        point = asset.point or point_size(float(asset.df.close.median()))
        lot = 1.0 if kind in ("forex", "equity") else (0.01 if kind == "cfd" else 1e-8)
        sec = Security(Symbol(key), kind, asset.df, point, asset.lot_step or lot)
        dict.__setitem__(self.securities, key, sec)
        self._books[key] = None  # created when the run starts (needs the cash split)
        return sec

    def add_forex(self, ticker, resolution=None, market=None, *args, **kwargs) -> Security:
        return self._add(ticker, "forex")

    def add_equity(self, ticker, resolution=None, market=None, *args, **kwargs) -> Security:
        return self._add(ticker, "equity")

    def add_crypto(self, ticker, resolution=None, market=None, *args, **kwargs) -> Security:
        return self._add(ticker, "crypto")

    def add_cfd(self, ticker, resolution=None, market=None, *args, **kwargs) -> Security:
        return self._add(ticker, "cfd")

    add_security = add_cfd

    # --- indicators ---------------------------------------------------------------------------
    def _indicator(self, kind: str, symbol, *args) -> Indicator:
        params = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        ind = INDICATORS[kind](*params)
        self.register_indicator(symbol, ind)
        return ind

    def sma(self, symbol, period, *a, **k): return self._indicator("sma", symbol, period)  # noqa: E704
    def ema(self, symbol, period, *a, **k): return self._indicator("ema", symbol, period)  # noqa: E704
    def rsi(self, symbol, period, *a, **k): return self._indicator("rsi", symbol, period)  # noqa: E704
    def std(self, symbol, period, *a, **k): return self._indicator("std", symbol, period)  # noqa: E704
    def atr(self, symbol, period, *a, **k): return self._indicator("atr", symbol, period)  # noqa: E704
    def roc(self, symbol, period, *a, **k): return self._indicator("roc", symbol, period)  # noqa: E704
    def mom(self, symbol, period, *a, **k): return self._indicator("mom", symbol, period)  # noqa: E704

    def max(self, symbol, period, *a, **k): return self._indicator("max", symbol, period)  # noqa: E704
    def min(self, symbol, period, *a, **k): return self._indicator("min", symbol, period)  # noqa: E704

    def bb(self, symbol, period, k=2.0, *a, **kw):
        return self._indicator("bb", symbol, period, k)

    def macd(self, symbol, fast=12, slow=26, signal=9, *a, **k):
        ind = MovingAverageConvergenceDivergence(fast, slow, signal)
        self.register_indicator(symbol, ind)
        return ind

    def register_indicator(self, symbol, indicator, *args, **kwargs) -> None:
        self._indicators.setdefault(str(symbol).upper(), []).append(indicator)

    # --- data helpers ----------------------------------------------------------------------------
    def history(self, symbols, periods=10, resolution=None, *args, **kwargs) -> pd.DataFrame:
        """Bars before now (never after): a DataFrame indexed by (symbol, time), like QuantConnect."""
        syms = symbols if isinstance(symbols, (list, tuple)) else [symbols]
        frames = []
        for s in syms:
            sec = self.securities[s]
            past = sec.df[sec.df.index <= self.utc_time - sec.period]
            if isinstance(periods, timedelta):
                past = past[past.index > self.utc_time - periods]
            else:
                past = past.tail(int(periods))
            f = past[["open", "high", "low", "close", "volume"]].copy()
            f.index = pd.MultiIndex.from_arrays([[Symbol(str(s).upper())] * len(f), self._local(f.index)],
                                                names=["symbol", "time"])
            frames.append(f)
        return pd.concat(frames) if frames else pd.DataFrame()

    def consolidate(self, symbol, period, handler) -> None:
        span = period if isinstance(period, timedelta) else RESOLUTION_SPAN.get(str(period).upper())
        if span is None:
            raise AlgorithmError(f"consolidate: use a timedelta or Resolution.MINUTE/HOUR/DAILY (got {period!r})")
        self._consolidators.append({"symbol": str(symbol).upper(), "span": pd.Timedelta(span),
                                    "handler": handler, "bar": None, "bucket": None})

    # --- orders ------------------------------------------------------------------------------
    def market_order(self, symbol, quantity, asynchronous=False, tag="", *args, **kwargs) -> OrderTicket:
        return self._submit(symbol, quantity, "market", tag=tag)

    def limit_order(self, symbol, quantity, limit_price, tag="", *args, **kwargs) -> OrderTicket:
        return self._submit(symbol, quantity, "limit", limit=limit_price, tag=tag)

    def stop_market_order(self, symbol, quantity, stop_price, tag="", *args, **kwargs) -> OrderTicket:
        return self._submit(symbol, quantity, "stop market", stop=stop_price, tag=tag)

    def stop_limit_order(self, symbol, quantity, stop_price, limit_price, tag="", *args, **kwargs) -> OrderTicket:
        return self._submit(symbol, quantity, "stop limit", stop=stop_price, limit=limit_price, tag=tag)

    def market_on_open_order(self, symbol, quantity, tag="", *args, **kwargs) -> OrderTicket:
        return self._submit(symbol, quantity, "market on open", tag=tag)

    def calculate_order_quantity(self, symbol, target: float) -> float:
        sec, book = self.securities[symbol], self._books[str(symbol).upper()]
        if not sec.bar:
            return 0.0
        per_unit = book.notional_usd(1.0, sec.price)
        units = target * self._equity_total() / per_unit if per_unit else 0.0
        return self._round_lot(sec, units) - book.position

    def set_holdings(self, symbol, percentage, liquidate_existing_holdings=False, tag="", *args, **kwargs):
        if isinstance(symbol, (list, tuple)):  # list of PortfolioTarget-like (symbol, quantity) pairs
            for target in symbol:
                self.set_holdings(target.symbol, target.quantity, tag=tag)
            return None
        if liquidate_existing_holdings:
            for s in list(self._books):
                if s != str(symbol).upper():
                    self.liquidate(s)
        qty = self.calculate_order_quantity(symbol, float(percentage))
        return self._submit(symbol, qty, "market", tag=tag) if qty else None

    def liquidate(self, symbol=None, tag="Liquidated", *args, **kwargs) -> list[OrderTicket]:
        targets = [str(symbol).upper()] if symbol is not None else list(self._books)
        tickets = []
        for s in targets:
            self.transactions.cancel_open_orders(s)
            pos = self._books[s].position
            if pos:
                tickets.append(self._submit(s, -pos, "market", tag=tag))
        return tickets

    # --- logging, plotting, stopping -------------------------------------------------------
    def debug(self, message) -> None:
        if len(self._logs) < MAX_LOG_LINES:
            self._logs.append(f"{self.time:%Y-%m-%d %H:%M:%S} {message}")

    log = error = debug

    def plot(self, chart, series=None, value=None, *args) -> None:
        if value is None:  # plot(series, value)
            chart, series, value = "Custom", chart, series
        self._plots.setdefault(str(chart), {}).setdefault(str(series), []).append((self.utc_time, float(value)))

    def quit(self, message: str = "") -> None:
        self.debug(f"Quit: {message}")
        self._quit = True

    def on_data(self, data) -> None:  # overridden by the user's class
        pass

    def initialize(self) -> None:
        raise AlgorithmError("The algorithm needs an initialize() (or Initialize()) method")

    def __getattr__(self, name: str):
        snake = _snake(name)
        if snake != name:
            try:
                return object.__getattribute__(self, snake)
            except AttributeError:
                pass
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"self.{name} isn't supported in Backtest Studio yet (scheduled events, universes, "
                             "options and the algorithm framework aren't available here)")

    # --- internals ------------------------------------------------------------------------------
    def _local(self, utc_index):
        return pd.DatetimeIndex(utc_index).tz_convert(self._tz).tz_localize(None)

    def _round_lot(self, sec: Security, units: float) -> float:
        lot = sec.lot
        return math.copysign(math.floor(abs(units) / lot + 1e-9) * lot, units)

    def _cash_total(self) -> float:
        return sum(b.cash for b in self._books.values())

    def _equity_total(self) -> float:
        total = 0.0
        for s, b in self._books.items():
            sec = self.securities[s]
            total += b.equity_at(sec.price, sec.ask_price) if sec.bar else b.cash
        return total

    def _submit(self, symbol, quantity, kind, limit=None, stop=None, tag="") -> OrderTicket:
        key = str(symbol).upper()
        if key not in self.securities:
            raise AlgorithmError(f"Order for {symbol}, which hasn't been added")
        sec = self.securities[key]
        qty = self._round_lot(sec, float(quantity))
        order = _Order(self._next_id, key, qty, kind, self.time, limit, stop, tag or "")
        self._next_id += 1
        self._all_orders.append(order)
        reason = ("during warm-up" if self.is_warming_up else "the algorithm has stopped (equity at zero)"
                  if getattr(self, "_halted", False) else "no data yet for this symbol" if not sec.bar
                  else "quantity rounds to zero" if qty == 0 else "")
        if reason:
            order.status = "Invalid"
            self.debug(f"Order {order.id} for {quantity} {key} rejected: {reason}")
            return OrderTicket(self, order)
        if kind == "market":
            bid = sec.price
            self._fill(order, bid + sec.spread if qty > 0 else bid, slip=True, time=self.utc_time)
        else:
            self._open_orders.append(order)
        return OrderTicket(self, order)

    def _cancel(self, order: _Order) -> None:
        if order in self._open_orders:
            self._open_orders.remove(order)
            order.status = "Canceled"
            self._event(order, 0.0, 0.0)

    def _fill(self, order: _Order, level: float, slip: bool, time) -> None:
        sec = self.securities[order.symbol]
        buy = order.quantity > 0
        slip_price = self._slippage.points(None) * sec.point if slip else 0.0
        price = level + slip_price if buy else level - slip_price
        reason = order.tag or order.type
        fill = FillEvent(time, order.quantity, price,
                         spread_cost=abs(order.quantity) * sec.spread if buy else 0.0,
                         slippage_cost=abs(order.quantity) * slip_price, reason=reason)
        self._books[order.symbol].on_fill(fill)
        order.status, order.fill_price = "Filled", price
        self._event(order, order.quantity, price)

    def _event(self, order: _Order, qty: float, price: float) -> None:
        handler = getattr(type(self), "on_order_event", None) or getattr(type(self), "OnOrderEvent", None)
        if handler is not None:
            _call_user(handler, self, OrderEvent(order, qty, price, self.utc_time))

    def _work_orders(self, sec: Security, bar: Bar) -> None:
        """Fill open limit / stop / market-on-open orders that this bar reaches."""
        s = sec.spread
        for order in [o for o in self._open_orders if o.symbol == sec.symbol]:
            buy = order.quantity > 0
            o_, h_, l_ = (bar.open + s, bar.high + s, bar.low + s) if buy else (bar.open, bar.high, bar.low)
            level, slip = None, False
            if order.type == "market on open":
                level, slip = o_, True
            elif order.type == "stop limit" and not order.stop_triggered:
                if (buy and h_ >= order.stop_price) or (not buy and l_ <= order.stop_price):
                    order.stop_triggered = True  # it's now a limit order, from the next bar
                continue
            elif order.type in ("limit", "stop limit"):
                p = order.limit_price
                if buy:
                    level = o_ if o_ <= p else p if l_ <= p else None
                else:
                    level = o_ if o_ >= p else p if h_ >= p else None
            elif order.type == "stop market":
                p, slip = order.stop_price, True
                if buy:
                    level = o_ if o_ >= p else p if h_ >= p else None
                else:
                    level = o_ if o_ <= p else p if l_ <= p else None
            if level is not None:
                self._open_orders.remove(order)
                self._fill(order, level, slip, bar.time)


def _to_dt(*args) -> datetime:
    if len(args) == 1 and isinstance(args[0], (datetime, date)):
        d = args[0]
        return d if isinstance(d, datetime) else datetime(d.year, d.month, d.day)
    return datetime(*[int(a) for a in args])


# --- running user code ---------------------------------------------------------------------------
def _user_line(exc: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == SOURCE_NAME]
    return f" (line {frames[-1].lineno}: {frames[-1].line})" if frames else ""


def _call_user(fn, *args):
    try:
        return fn(*args)
    except AlgorithmError:
        raise
    except Exception as exc:  # anything the user's code raises
        raise AlgorithmError(f"{type(exc).__name__}: {exc}{_user_line(exc)}") from exc


def _install_imports() -> None:
    """Make `from AlgorithmImports import *` (and the older QuantConnect imports) work."""
    names = {"QCAlgorithm": QCAlgorithm, "Symbol": Symbol, "TradeBar": TradeBar, "Slice": Slice,
             "RollingWindow": RollingWindow, "IndicatorDataPoint": IndicatorDataPoint,
             "SimpleMovingAverage": SimpleMovingAverage, "ExponentialMovingAverage": ExponentialMovingAverage,
             "RelativeStrengthIndex": RelativeStrengthIndex, "BollingerBands": BollingerBands,
             "AverageTrueRange": AverageTrueRange, "MovingAverageConvergenceDivergence": MovingAverageConvergenceDivergence,
             "StandardDeviation": StandardDeviation, "Maximum": Maximum, "Minimum": Minimum,
             "RateOfChange": RateOfChange, "Momentum": Momentum, "OrderEvent": OrderEvent, "OrderTicket": OrderTicket,
             "datetime": datetime, "timedelta": timedelta, "date": date, "time": dtime, "np": np, "pd": pd,
             "math": math, **_enums}
    for mod_name in ("AlgorithmImports", "QuantConnect", "QuantConnect.Algorithm", "QuantConnect.Data",
                     "QuantConnect.Data.Market", "QuantConnect.Indicators", "QuantConnect.Orders", "System"):
        mod = types.ModuleType(mod_name)
        mod.__dict__.update(names)
        mod.__all__ = list(names)
        sys.modules.setdefault(mod_name, mod)
    clr = types.ModuleType("clr")
    clr.AddReference = lambda *a, **k: None
    sys.modules.setdefault("clr", clr)


_install_imports()

QC_TEMPLATE = '''\
# A QuantConnect-style algorithm: paste one written for QuantConnect, or start from this.
# Symbols are matched to your uploaded CSVs (set each file's symbol in the Data section).
from AlgorithmImports import *


class EmaCrossover(QCAlgorithm):

    def initialize(self):
        self.set_cash(10000)
        self.pair = self.add_forex("USDJPY", Resolution.MINUTE).symbol
        self.fast = self.ema(self.pair, 20)
        self.slow = self.ema(self.pair, 50)
        self.set_warm_up(50)

    def on_data(self, data):
        if self.is_warming_up or not self.slow.is_ready or self.pair not in data:
            return
        holding = self.portfolio[self.pair]
        if self.fast.current.value > self.slow.current.value and not holding.is_long:
            self.set_holdings(self.pair, 1.0)      # 100% of equity long
        elif self.fast.current.value < self.slow.current.value and not holding.is_short:
            self.set_holdings(self.pair, -1.0)     # 100% of equity short
'''


def load_algorithm(source: str) -> type:
    """Compile pasted code and return its QCAlgorithm subclass. The code runs with full Python access."""
    namespace: dict = {"__name__": "user_algorithm"}
    try:
        exec(compile(source, SOURCE_NAME, "exec"), namespace)  # noqa: S102 - the user's own algorithm
    except SyntaxError as exc:
        raise AlgorithmError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc
    except Exception as exc:
        raise AlgorithmError(f"{type(exc).__name__}: {exc}{_user_line(exc)}") from exc
    classes = [v for v in namespace.values() if isinstance(v, type) and issubclass(v, QCAlgorithm)
               and v is not QCAlgorithm and v.__module__ == "user_algorithm"]
    if not classes:
        raise AlgorithmError("Define a class that inherits from QCAlgorithm, e.g. class MyAlgo(QCAlgorithm):")
    return classes[-1]


@dataclass
class AlgorithmRun:
    result: MultiAssetResult
    name: str
    logs: list[str] = field(default_factory=list)
    plots: dict = field(default_factory=dict)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_algorithm(source_or_class, assets: list[Asset], cash: float | None = None,
                  slippage: Slippage | None = None, title: str | None = None) -> AlgorithmRun:
    """Run a QuantConnect-style algorithm on `assets` (matched by symbol). Cash: the algorithm's
    set_cash if it calls it, else `cash`, else QuantConnect's default of 100,000."""
    cls = load_algorithm(source_or_class) if isinstance(source_or_class, str) else source_or_class
    algo = cls()
    algo._datasets = {a.symbol.upper(): a for a in assets}
    algo._slippage = slippage or Slippage()
    init = getattr(cls, "initialize", None)
    init = init if init is not QCAlgorithm.initialize else getattr(cls, "Initialize", QCAlgorithm.initialize)
    _call_user(init, algo)
    on_data = cls.on_data if cls.on_data is not QCAlgorithm.on_data else getattr(cls, "OnData", QCAlgorithm.on_data)
    if not algo.securities:
        raise AlgorithmError("initialize() didn't add any data (add_forex / add_equity / add_crypto / add_cfd)")

    total = algo._cash if algo._cash is not None else (cash if cash is not None else DEFAULT_CASH)
    syms = list(algo.securities.keys())
    for s in syms:
        algo._books[s] = Portfolio(s, cash=total / len(syms), point=algo.securities[s].point)

    # The common timeline, limited to the start / end dates (given in the algorithm's time zone).
    tz = algo._tz
    utc = lambda d: pd.Timestamp(d).tz_localize(tz).tz_convert("UTC")  # noqa: E731
    rows = {s: {t: r for t, r in zip(algo.securities[s].df.index, algo.securities[s].df.itertuples(index=False))}
            for s in syms}
    # Bar start / end times in the algorithm's time zone, worked out once (not per bar).
    local_start, local_end = {}, {}
    for s in syms:
        idx, per = algo.securities[s].df.index, pd.Timedelta(algo.securities[s].period)
        local_start[s] = dict(zip(idx, algo._local(idx).to_pydatetime()))
        local_end[s] = dict(zip(idx, algo._local(idx + per).to_pydatetime()))
    timeline = pd.DatetimeIndex(sorted(set().union(*(algo.securities[s].df.index for s in syms))))
    start = utc(algo._start) if algo._start else timeline[0]
    end = utc(algo._end) + pd.Timedelta(days=1) if algo._end else timeline[-1] + pd.Timedelta(seconds=1)
    first = timeline.searchsorted(start)
    if isinstance(algo._warmup, int):
        warm_from = max(0, first - algo._warmup)
        trade_from = timeline[min(len(timeline) - 1, max(first, warm_from + algo._warmup))]
    elif isinstance(algo._warmup, timedelta):
        warm_from = timeline.searchsorted(start - pd.Timedelta(algo._warmup))
        trade_from = max(start, timeline[0] + pd.Timedelta(algo._warmup))
    else:
        warm_from, trade_from = first, start
    steps = timeline[warm_from:]
    steps = steps[steps < end]
    if len(steps) == 0 or steps[-1] < trade_from:
        raise AlgorithmError("No data between the algorithm's start and end dates "
                             f"(data runs {timeline[0]:%Y-%m-%d} to {timeline[-1]:%Y-%m-%d} UTC)")

    last_bar: dict[str, Bar] = {}
    for t in steps:
        if algo._quit:
            break
        algo.is_warming_up = t < trade_from
        bars: dict[str, TradeBar] = {}
        for s in syms:
            r = rows[s].get(t)
            if r is None:
                continue
            sec = algo.securities[s]
            spread_pts = getattr(r, "spread", 0.0) if hasattr(r, "spread") else 0.0
            sec.spread = (spread_pts or 0.0) * sec.point
            bar = Bar(t, r.open, r.high, r.low, r.close, getattr(r, "volume", 0.0), spread_pts or 0.0)
            algo._work_orders(sec, bar)
            tb = TradeBar(Symbol(s), local_start[s][t], local_end[s][t],
                          r.open, r.high, r.low, r.close, getattr(r, "volume", 0.0), sec.period)
            sec.bar = tb
            for ind in algo._indicators.get(s, []):
                ind.update(tb)
            _consolidate(algo, s, tb, t)
            if not algo.is_warming_up:
                algo._books[s].on_bar(bar, r.close + sec.spread)
            last_bar[s] = bar
            bars[s] = tb
        if not bars:
            continue
        last = max(bars, key=lambda s: algo.securities[s].period)
        algo.utc_time = t + algo.securities[last].period
        algo.time = local_end[last][t]
        if not algo.is_warming_up and algo._equity_total() <= 0 and not getattr(algo, "_halted", False):
            algo.debug("Equity reached zero: positions closed, no more trading (margin call)")
            for s, b in algo._books.items():
                if b.position and s in last_bar:
                    lb = last_bar[s]
                    sec = algo.securities[s]
                    b.on_fill(FillEvent(lb.time, -b.position, lb.close + (sec.spread if -b.position > 0 else 0),
                                        reason="margin call"))
            algo._open_orders.clear()
            algo._halted = True
        if not getattr(algo, "_halted", False):
            _call_user(on_data, algo, Slice(algo.time, bars))

    end_hook = getattr(cls, "on_end_of_algorithm", None) or getattr(cls, "OnEndOfAlgorithm", None)
    if end_hook is not None:
        _call_user(end_hook, algo)

    # Close what's still open so it shows in the trade log, then build the results.
    per_asset = {}
    for s, b in algo._books.items():
        if not b.history:
            continue
        if b.position and s in last_bar:
            lb, sec = last_bar[s], algo.securities[s]
            b.on_fill(FillEvent(lb.time, -b.position, lb.close + (sec.spread if -b.position > 0 else 0.0),
                                spread_cost=abs(b.position) * sec.spread if -b.position > 0 else 0.0,
                                reason="end of data"))
            b.history[-1] = b.snapshot(lb, lb.close + sec.spread)
        meta = {"symbol": s, "sizing": "set by the algorithm", "orders": "set by the algorithm",
                "slippage": algo._slippage.describe(), "skipped_signals": 0}
        per_asset[s] = result_from_portfolio(b, f"{s}: {title or cls.__name__}", meta)
    if not per_asset:
        raise AlgorithmError("The algorithm never reached its trading period (check the dates and warm-up)")

    weights = {s: 1 / len(per_asset) for s in per_asset}
    benchmarks = {}
    for s, r in per_asset.items():
        closes = algo.securities[s].df.close
        closes = closes[(closes.index >= r.history.index[0]) & (closes.index <= r.history.index[-1])]
        benchmarks[s] = r.initial_cash * closes / closes.iloc[0]
    name = title or cls.__name__
    if len(per_asset) == 1:
        (s, only), = per_asset.items()
        only.title = name
        multi = MultiAssetResult(only, per_asset, weights, benchmarks[s], benchmarks)
    else:
        combined = _combine(per_asset, total, name)
        from .multi import _sum_on
        bench = _sum_on(combined.history.index, {s: (b, per_asset[s].initial_cash) for s, b in benchmarks.items()})
        multi = MultiAssetResult(combined, per_asset, weights, bench, benchmarks)
    multi.combined.meta.update(sizing="set by the algorithm", orders="set by the algorithm", algorithm=cls.__name__)
    orders = pd.DataFrame([{"id": o.id, "time": o.time, "symbol": o.symbol, "type": o.type, "quantity": o.quantity,
                            "limit": o.limit_price, "stop": o.stop_price, "status": o.status,
                            "fill_price": o.fill_price, "tag": o.tag} for o in algo._all_orders])
    return AlgorithmRun(multi, name, algo._logs, algo._plots, orders)


def _consolidate(algo: QCAlgorithm, symbol: str, bar: TradeBar, t: pd.Timestamp) -> None:
    for c in algo._consolidators:
        if c["symbol"] != symbol:
            continue
        span = c["span"]
        bucket = t.floor(span) if span <= pd.Timedelta(days=1) else t.normalize()
        if c["bucket"] is not None and bucket != c["bucket"] and c["bar"] is not None:
            _call_user(c["handler"], c["bar"])  # a bound method (self.on_day), so no algo argument
            c["bar"] = None
        if c["bar"] is None:
            start = algo._local([bucket])[0].to_pydatetime()
            c["bar"] = TradeBar(bar.symbol, start, start + span.to_pytimedelta(), bar.open, bar.high, bar.low,
                                bar.close, bar.volume, span.to_pytimedelta())
        else:
            b = c["bar"]
            b.high, b.low, b.close, b.volume = max(b.high, bar.high), min(b.low, bar.low), bar.close, b.volume + bar.volume
        c["bucket"] = bucket
