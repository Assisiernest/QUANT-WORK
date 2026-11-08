"""How the portfolio turns a signal into orders: entry order type and attached exits."""

from __future__ import annotations

from dataclasses import dataclass

ORDER_TYPES = ("market", "limit", "stop")


@dataclass(frozen=True)
class OrderPolicy:
    """Order settings applied to every entry. Distances are in pips.

    entry:            "market" (next bar's open), "limit" (better price: buy below / sell above
                      the signal bar's close by entry_offset_pips) or "stop" (breakout: buy above /
                      sell below the close by entry_offset_pips).
    expiry_bars:      cancel an unfilled limit/stop entry after this many bars (None = until the
                      next signal).
    stop_loss_pips, take_profit_pips: fixed exits from the entry fill price.
    trailing_stop_pips: stop that follows the best price by this distance; never loosens.
    """
    entry: str = "market"
    entry_offset_pips: float = 0.0
    expiry_bars: int | None = None
    stop_loss_pips: float | None = None
    take_profit_pips: float | None = None
    trailing_stop_pips: float | None = None

    def __post_init__(self):
        if self.entry not in ORDER_TYPES:
            raise ValueError(f"entry must be one of {ORDER_TYPES}, got {self.entry!r}")
        if self.entry != "market" and self.entry_offset_pips < 0:
            raise ValueError("entry_offset_pips must be >= 0")
        for name in ("stop_loss_pips", "take_profit_pips", "trailing_stop_pips"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 (or None to turn it off)")
        if self.expiry_bars is not None and self.expiry_bars < 1:
            raise ValueError("expiry_bars must be >= 1")

    def describe(self) -> str:
        parts = [self.entry if self.entry == "market" else f"{self.entry} {self.entry_offset_pips:g} pips"]
        if self.expiry_bars:
            parts[0] += f" (expires after {self.expiry_bars} bars)"
        for label, value in [("SL", self.stop_loss_pips), ("TP", self.take_profit_pips),
                             ("trailing", self.trailing_stop_pips)]:
            if value:
                parts.append(f"{label} {value:g} pips")
        return ", ".join(parts)
