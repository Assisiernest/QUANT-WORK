"""Position sizing: how many units to trade on each entry.

Every sizer rounds down to `lot_step` units (1,000 = a micro lot), skips the trade if
that leaves less than `min_units`, and caps the position at `max_leverage` x equity
(notional, in USD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .portfolio import Portfolio


@dataclass(frozen=True)
class Sizer:
    lot_step: float = 1_000
    min_units: float = 1_000
    max_leverage: float = 30.0

    def raw_units(self, equity: float, price: float, stop_distance: float | None, portfolio: "Portfolio") -> float:
        raise NotImplementedError

    def size(self, equity: float, price: float, stop_distance: float | None, portfolio: "Portfolio") -> float:
        """Units to trade (>= 0) for a new position at `price`."""
        if equity <= 0:
            return 0.0
        units = self.raw_units(equity, price, stop_distance, portfolio)
        cap = equity * self.max_leverage / portfolio.notional_usd(1.0, price)
        units = min(units, cap)
        units = math.floor(units / self.lot_step + 1e-9) * self.lot_step
        return units if units >= self.min_units else 0.0

    def describe(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedUnits(Sizer):
    """The same size every trade. 10,000 units = 0.1 lot."""
    units: float = 10_000

    def raw_units(self, equity, price, stop_distance, portfolio):
        return self.units

    def describe(self) -> str:
        return f"fixed {self.units:,.0f} units ({self.units / 100_000:g} lot)"


@dataclass(frozen=True)
class PercentEquity(Sizer):
    """Notional position = pct % of equity (100 = 1x leverage, 500 = 5x)."""
    pct: float = 100.0

    def raw_units(self, equity, price, stop_distance, portfolio):
        return equity * self.pct / 100 / portfolio.notional_usd(1.0, price)

    def describe(self) -> str:
        return f"{self.pct:g}% of equity notional (max {self.max_leverage:g}x)"


@dataclass(frozen=True)
class RiskPercent(Sizer):
    """Risk pct % of equity per trade: size so that hitting the stop-loss loses that much.

    Needs a stop distance, so use it with a stop-loss (or trailing stop) in the OrderPolicy.
    """
    pct: float = 1.0

    def raw_units(self, equity, price, stop_distance, portfolio):
        if not stop_distance:
            raise ValueError("RiskPercent sizing needs a stop-loss or trailing stop distance")
        loss_per_unit = portfolio.usd_value(stop_distance, price)
        return equity * self.pct / 100 / loss_per_unit

    def describe(self) -> str:
        return f"risk {self.pct:g}% of equity per trade (max {self.max_leverage:g}x)"
