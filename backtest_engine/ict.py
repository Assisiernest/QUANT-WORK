"""ICT / Smart Money Concepts detectors, coded from the ORB & ICT/SMC research pack (Part 2).

Everything is incremental: feed closed candles one at a time to SMC.update() and it only
ever uses that candle and earlier ones, so there's no look-ahead. Swings are confirmed N
candles late, breakers / mitigation blocks are created on the later close that makes them,
and every object carries the index of the candle on which it became known (`known`).

Detection order per candle (Section 12 of the pack):
    1. ATR, then swings (confirmed N bars late)
    2. liquidity levels, then sweeps
    3. structure: BOS / CHoCH / MSS and the trend
    4. FVGs and their states (open -> mitigated -> filled, or inverted -> IFVG)
    5. order blocks on displacement; failed OBs become breaker or mitigation blocks
    6. dealing range, premium / discount, golden entry zone
    7. reversal / continuation setup flags
plus support / resistance levels (the ORB spec's own definition, not an ICT object).

Every "DECIDE" choice from the pack is a field of SMCConfig, set to the pack's suggested
default (Section 13). Log the config with each backtest: SMCConfig.describe().

    smc = SMC(SMCConfig())
    for c in candles:                       # Candle(time, open, high, low, close)
        events = smc.update(c)              # e.g. [Event("bos", +1, ...), Event("fvg", -1, ...)]
    smc.zones_at(price, direction=+1, kinds=("fvg", "ifvg"))
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


# --- configuration -----------------------------------------------------------------
@dataclass(frozen=True)
class SMCConfig:
    # Structure and swings
    swing_internal: int = 3            # N for internal structure
    swing_external: int = 10           # N for external structure
    atr_period: int = 14
    displacement_k: float = 1.2        # displacement: candle body >= k x ATR
    mss_sweep_bars: int = 10           # MSS = CHoCH with displacement within X bars of a sweep
    # Liquidity and sweeps
    equal_tol_atr: float = 0.1         # equal highs / lows: |diff| <= tol x ATR
    sweep_window: int = 3              # reclaim within M bars (0 = same candle only)
    # FVGs, IFVGs and order blocks
    fvg_min_atr: float = 0.25          # minimum gap size, x ATR
    fvg_mitigation: str = "ce"         # "wick" (touch), "ce" (reach the midpoint) or "close" (close inside)
    zone_max_age: int = 200            # zones expire after this many candles
    ob_merge_cluster: bool = True      # merge consecutive opposite candles into one OB
    ob_max_gap: int = 5                # max candles between the OB and its displacement candle
    ob_bos_window: int = 5             # displacement must break structure within this many candles
    # Support / resistance (ORB spec: 3+ wick touches, no body close beyond)
    sr_min_touches: int = 3
    sr_tol_atr: float = 0.1
    sr_max_age: int = 500

    def __post_init__(self):
        if self.fvg_mitigation not in ("wick", "ce", "close"):
            raise ValueError("fvg_mitigation must be 'wick', 'ce' or 'close'")
        if not 1 <= self.swing_internal <= self.swing_external:
            raise ValueError("need 1 <= swing_internal <= swing_external")

    def describe(self) -> dict[str, Any]:
        return asdict(self)


# --- data objects ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Candle:
    time: Any
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open


@dataclass(slots=True)
class Swing:
    kind: str        # "high" / "low"
    price: float
    index: int       # candle that made the swing
    known: int       # candle on which it was confirmed (index + N)
    broken: bool = False


@dataclass(slots=True)
class Level:
    """A liquidity level. BSL sits above price (highs), SSL below (lows)."""
    price: float
    side: str        # "BSL" / "SSL"
    source: str      # "swing", "pdh", "pdl", "session_high", ... (anything you add)
    known: int
    weight: int = 1  # +1 for every equal high / low merged into it
    active: bool = True


@dataclass(slots=True)
class Zone:
    """A price zone: FVG, IFVG, OB, breaker or mitigation block. direction +1 = support
    (bullish), -1 = resistance (bearish)."""
    kind: str
    direction: int
    top: float
    bottom: float
    index: int       # first candle of the pattern
    known: int       # candle on which it was created / became this kind
    state: str = "open"   # open, mitigated, filled, inverted, invalidated, expired
    meta: dict = field(default_factory=dict)

    @property
    def ce(self) -> float:
        """Consequent encroachment: the zone's 50% level."""
        return (self.top + self.bottom) / 2

    @property
    def alive(self) -> bool:
        return self.state in ("open", "mitigated")

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return self.bottom - tol <= price <= self.top + tol


@dataclass(frozen=True, slots=True)
class Event:
    kind: str        # swing_high, swing_low, sweep, run, bos, choch, mss, fvg, ifvg, ob, breaker,
                     # mitigation, displacement
    direction: int   # +1 bullish, -1 bearish
    index: int       # candle on which it happened
    price: float
    info: dict = field(default_factory=dict)


@dataclass(slots=True)
class DealingRange:
    """The leg that produced the latest BOS / CHoCH. direction +1: low -> high (bullish)."""
    direction: int
    low: float
    high: float
    known: int

    def position(self, price: float) -> float:
        """0 at the range low, 1 at the high; < 0.5 discount, > 0.5 premium."""
        span = self.high - self.low
        return (price - self.low) / span if span > 0 else 0.5

    def level(self, r: float) -> float:
        """Fib retracement level r of the leg (0 = leg end, 1 = leg start)."""
        span = self.high - self.low
        return self.high - r * span if self.direction > 0 else self.low + r * span

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2

    def golden_zone(self, lo: float = 0.62, hi: float = 0.786) -> tuple[float, float]:
        """(bottom, top) of the golden entry zone (0.62-0.786 retracement)."""
        a, b = self.level(lo), self.level(hi)
        return min(a, b), max(a, b)

    def invalidated_by(self, close: float) -> bool:
        """A close beyond level 1 (the leg start) cancels the setup."""
        return close < self.low if self.direction > 0 else close > self.high


# --- building blocks -----------------------------------------------------------------
class ATR:
    """Wilder's average true range."""

    def __init__(self, period: int):
        self.period, self.value, self._prev_close, self._seed = period, float("nan"), None, []

    def update(self, c: Candle) -> float:
        tr = c.high - c.low if self._prev_close is None else max(
            c.high - c.low, abs(c.high - self._prev_close), abs(c.low - self._prev_close))
        self._prev_close = c.close
        if len(self._seed) < self.period:
            self._seed.append(tr)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value

    @property
    def ready(self) -> bool:
        return self.value == self.value  # not NaN


class SwingDetector:
    """Swing high: high[i] above the highs of the N candles on each side (strictly).
    Known only N candles later."""

    def __init__(self, n: int):
        self.n = n
        self._buf: deque[tuple[int, Candle]] = deque(maxlen=2 * n + 1)

    def update(self, i: int, c: Candle) -> list[Swing]:
        self._buf.append((i, c))
        if len(self._buf) < self._buf.maxlen:
            return []
        mid_i, mid = self._buf[self.n]
        others = [x for k, (_, x) in enumerate(self._buf) if k != self.n]
        out = []
        if mid.high > max(x.high for x in others):
            out.append(Swing("high", mid.high, mid_i, i))
        if mid.low < min(x.low for x in others):
            out.append(Swing("low", mid.low, mid_i, i))
        return out


class Structure:
    """BOS / CHoCH on one swing length.

    Bullish break: a close above the last unbroken swing high. In an uptrend it's a BOS,
    in a downtrend a CHoCH (trend flips). Bearish mirror. The protected low after a
    bullish break is the low of the leg that made it (the lowest low between the broken
    swing high and the breaking candle); a close below it is a bearish CHoCH."""

    def __init__(self, name: str):
        self.name = name
        self.trend = 0
        self.last_high: Swing | None = None
        self.last_low: Swing | None = None
        self.protected_low: float | None = None
        self.protected_high: float | None = None

    def on_swing(self, s: Swing) -> None:
        if s.kind == "high":
            self.last_high = s
        else:
            self.last_low = s

    def update(self, i: int, c: Candle, candles: deque) -> list[tuple[str, int, float, float, float]]:
        """Breaks on this candle as (kind, direction, broken level, leg low, leg high)."""
        out = []
        up_level = None
        if self.last_high and not self.last_high.broken and c.close > self.last_high.price:
            up_level = self.last_high.price
            self.last_high.broken = True
        elif self.trend == -1 and self.protected_high is not None and c.close > self.protected_high:
            up_level = self.protected_high
        if up_level is not None:
            start = self.last_high.index if self.last_high else i
            leg_low = _extreme(candles, start, i, low=True)
            out.append(("choch" if self.trend == -1 else "bos", 1, up_level, leg_low, c.high))
            self.trend, self.protected_low, self.protected_high = 1, leg_low, None
            return out

        dn_level = None
        if self.last_low and not self.last_low.broken and c.close < self.last_low.price:
            dn_level = self.last_low.price
            self.last_low.broken = True
        elif self.trend == 1 and self.protected_low is not None and c.close < self.protected_low:
            dn_level = self.protected_low
        if dn_level is not None:
            start = self.last_low.index if self.last_low else i
            leg_high = _extreme(candles, start, i, low=False)
            out.append(("choch" if self.trend == 1 else "bos", -1, dn_level, c.low, leg_high))
            self.trend, self.protected_high, self.protected_low = -1, leg_high, None
        return out


def _extreme(candles: deque, start: int, end: int, low: bool) -> float:
    """Lowest low (or highest high) of candles with index in [start, end]."""
    vals = [c.low if low else c.high for k, c in candles if start <= k <= end]
    return (min if low else max)(vals) if vals else float("nan")


# --- the tracker -------------------------------------------------------------------
class SMC:
    """Runs every detector once per closed candle, in the pack's Section 12 order."""

    HISTORY = 600  # candles kept for leg look-backs

    def __init__(self, config: SMCConfig | None = None):
        self.cfg = cfg = config or SMCConfig()
        self.i = -1
        self.candles: deque[tuple[int, Candle]] = deque(maxlen=self.HISTORY)
        self.atr = ATR(cfg.atr_period)
        self._sw_int, self._sw_ext = SwingDetector(cfg.swing_internal), SwingDetector(cfg.swing_external)
        self.internal, self.external = Structure("internal"), Structure("external")
        self.swings: list[Swing] = []          # internal swings
        self.levels: list[Level] = []          # liquidity
        self._pending_sweeps: list[tuple[Level, int]] = []  # (level, candle it was pierced)
        self.zones: list[Zone] = []            # FVG, IFVG, OB, breaker, mitigation
        self._pending_obs: list[Zone] = []     # OBs waiting for their displacement to break structure
        self.sr: list[dict] = []               # support / resistance candidates
        self.dealing_range: DealingRange | None = None
        self.events: list[Event] = []          # everything, in order
        self.last: dict[str, Event] = {}       # most recent event of each kind+direction, e.g. "sweep+1"

    # --- public -----------------------------------------------------------------
    def update(self, c: Candle) -> list[Event]:
        self.i += 1
        i = self.i
        self.candles.append((i, c))
        ev: list[Event] = []
        atr = self.atr.update(c)                                   # 1. ATR
        new_int = self._sw_int.update(i, c)                        #    swings
        new_ext = self._sw_ext.update(i, c)
        for s in new_int:
            self.swings.append(s)
            self.internal.on_swing(s)
            ev.append(Event(f"swing_{s.kind}", 1 if s.kind == "low" else -1, i, s.price, {"at": s.index}))
            self._add_liquidity(s)                                 # 2. liquidity
            self._add_sr(s)
        for s in new_ext:
            self.external.on_swing(s)
        ev += self._sweeps(c)                                      #    sweeps
        ev += self._structure(c)                                   # 3. BOS / CHoCH / MSS
        if self.atr.ready:
            ev += self._fvgs(c, atr)                               # 4. FVG / IFVG
            ev += self._order_blocks(c, atr, ev)                   # 5. OB, breaker, mitigation
        self._update_sr(c)
        self._update_dealing_range(c, ev)                          # 6. dealing range
        for e in ev:
            self.events.append(e)
            self.last[f"{e.kind}{e.direction:+d}"] = e
        if len(self.events) > 5000:
            del self.events[:1000]
        return ev

    def add_level(self, price: float, side: str, source: str) -> Level:
        """Add an outside liquidity level (PDH/PDL, session highs/lows...). Same sweep rules."""
        lvl = Level(price, side, source, self.i)
        self.levels.append(lvl)
        return lvl

    def zones_at(self, price: float, direction: int, kinds=("fvg", "ifvg"), tol: float = 0.0) -> list[Zone]:
        """Live zones of `kinds` acting in `direction` (+1 support, -1 resistance) that contain price."""
        return [z for z in self.zones if z.alive and z.kind in kinds and z.direction == direction
                and z.contains(price, tol)]

    def zones_since(self, known_from: int, direction: int, kinds) -> list[Zone]:
        """Zones of `kinds` in `direction` created on or after candle `known_from`."""
        return [z for z in self.zones if z.known >= known_from and z.kind in kinds and z.direction == direction]

    def sr_levels(self) -> list[float]:
        """Confirmed support / resistance prices: >= sr_min_touches touches, never closed through."""
        return [s["price"] for s in self.sr if s["touches"] >= self.cfg.sr_min_touches]

    def recent(self, kind: str, direction: int, within: int) -> Event | None:
        e = self.last.get(f"{kind}{direction:+d}")
        return e if e is not None and self.i - e.index <= within else None

    def premium_discount(self, price: float) -> str | None:
        dr = self.dealing_range
        if dr is None:
            return None
        pos = dr.position(price)
        return "discount" if pos < 0.5 else "premium" if pos > 0.5 else "equilibrium"

    def in_golden_zone(self, price: float) -> bool:
        dr = self.dealing_range
        if dr is None:
            return False
        lo, hi = dr.golden_zone()
        return lo <= price <= hi

    def reversal_setup(self, direction: int, price: float, within: int = 20) -> bool:
        """Section 11 reversal: sweep of the opposite liquidity, then a CHoCH/MSS this way with
        displacement, and price back in a PD array of this direction in the new leg's discount
        (longs) / premium (shorts)."""
        sweep = self.recent("sweep", direction, within)
        mss = self.recent("mss", direction, within)
        if not (sweep and mss and mss.index >= sweep.index):
            return False
        pd_ = self.premium_discount(price)
        want = "discount" if direction > 0 else "premium"
        arrays = self.zones_at(price, direction, kinds=("fvg", "ifvg", "ob", "breaker", "mitigation"))
        return pd_ == want and bool(arrays)

    def continuation_setup(self, direction: int, price: float, within: int = 20) -> bool:
        """Section 11 continuation: a recent BOS this way, price pulled back into the discount
        (premium) of that leg and sitting in an FVG / OB of this direction."""
        if self.internal.trend != direction or not self.recent("bos", direction, within):
            return False
        want = "discount" if direction > 0 else "premium"
        return (self.premium_discount(price) == want
                and bool(self.zones_at(price, direction, kinds=("fvg", "ob"))))

    # --- 2. liquidity and sweeps -----------------------------------------------
    def _add_liquidity(self, s: Swing) -> None:
        side = "BSL" if s.kind == "high" else "SSL"
        tol = self.cfg.equal_tol_atr * self.atr.value if self.atr.ready else 0.0
        for lvl in self.levels:
            if lvl.active and lvl.side == side and lvl.source == "swing" and abs(lvl.price - s.price) <= tol:
                lvl.weight += 1  # equal highs / lows: one level, more weight
                lvl.price = max(lvl.price, s.price) if side == "BSL" else min(lvl.price, s.price)
                return
        self.levels.append(Level(s.price, side, "swing", self.i))
        if len(self.levels) > 400:
            self.levels = [lv for lv in self.levels if lv.active][-300:]

    def _sweeps(self, c: Candle) -> list[Event]:
        """A wick beyond an active level that closes back inside (now or within M candles) is a
        sweep; a level held beyond for M candles is a run. Either way the level is consumed."""
        out, i, m = [], self.i, self.cfg.sweep_window
        still = []
        for lvl, pierced in self._pending_sweeps:  # pierced on an earlier candle, closed beyond
            back = c.close < lvl.price if lvl.side == "BSL" else c.close > lvl.price
            if back:
                out.append(self._sweep_event(lvl, i))
            elif i - pierced >= m:
                out.append(Event("run", 1 if lvl.side == "BSL" else -1, i, lvl.price, {"source": lvl.source}))
            else:
                still.append((lvl, pierced))
        self._pending_sweeps = still
        for lvl in self.levels:
            if not lvl.active or lvl.known >= i:
                continue
            if lvl.side == "BSL" and c.high > lvl.price:
                lvl.active = False
                if c.close < lvl.price:
                    out.append(self._sweep_event(lvl, i))
                elif m > 0:
                    self._pending_sweeps.append((lvl, i))
                else:
                    out.append(Event("run", 1, i, lvl.price, {"source": lvl.source}))
            elif lvl.side == "SSL" and c.low < lvl.price:
                lvl.active = False
                if c.close > lvl.price:
                    out.append(self._sweep_event(lvl, i))
                elif m > 0:
                    self._pending_sweeps.append((lvl, i))
                else:
                    out.append(Event("run", -1, i, lvl.price, {"source": lvl.source}))
        return out

    @staticmethod
    def _sweep_event(lvl: Level, i: int) -> Event:
        # Sweeping buy-side liquidity is bearish; sweeping sell-side liquidity is bullish.
        return Event("sweep", -1 if lvl.side == "BSL" else 1, i, lvl.price,
                     {"source": lvl.source, "weight": lvl.weight})

    # --- 3. structure ----------------------------------------------------------
    def _structure(self, c: Candle) -> list[Event]:
        out = []
        disp = self.atr.ready and c.body >= self.cfg.displacement_k * self.atr.value
        for st in (self.internal, self.external):
            for kind, d, level, leg_low, leg_high in st.update(self.i, c, self.candles):
                info = {"structure": st.name, "leg_low": leg_low, "leg_high": leg_high}
                out.append(Event(kind, d, self.i, level, info))
                if kind == "choch" and st is self.internal:
                    swept = self.recent("sweep", d, self.cfg.mss_sweep_bars) or any(
                        e.kind == "sweep" and e.direction == d for e in out)
                    if disp and (c.close > c.open) == (d > 0) and swept:
                        out.append(Event("mss", d, self.i, level, info))
        return out

    # --- 4. FVG / IFVG ---------------------------------------------------------
    def _fvgs(self, c: Candle, atr: float) -> list[Event]:
        out, i, cfg = [], self.i, self.cfg
        flipped = []  # IFVGs made on this candle: added after the loop so they start next candle
        for z in self.zones:  # states of existing gaps (not the one made on this candle)
            if z.kind not in ("fvg", "ifvg") or not z.alive:
                continue
            if i - z.known > cfg.zone_max_age:
                z.state = "expired"
                continue
            if z.direction > 0:  # support: price comes down into it
                if c.close < z.bottom:
                    if z.kind == "fvg":
                        z.state = "inverted"
                        nz = Zone("ifvg", -1, z.top, z.bottom, z.index, i, meta={"from": "bullish fvg"})
                        flipped.append(nz)
                        out.append(Event("ifvg", -1, i, nz.ce))
                    else:
                        z.state = "invalidated"
                    continue
                touched = {"wick": c.low <= z.top, "ce": c.low <= z.ce, "close": c.close <= z.top}[cfg.fvg_mitigation]
                if c.low <= z.bottom:
                    z.state = "filled"
                elif touched:
                    z.state = "mitigated"
            else:
                if c.close > z.top:
                    if z.kind == "fvg":
                        z.state = "inverted"
                        nz = Zone("ifvg", 1, z.top, z.bottom, z.index, i, meta={"from": "bearish fvg"})
                        flipped.append(nz)
                        out.append(Event("ifvg", 1, i, nz.ce))
                    else:
                        z.state = "invalidated"
                    continue
                touched = {"wick": c.high >= z.bottom, "ce": c.high >= z.ce,
                           "close": c.close >= z.bottom}[cfg.fvg_mitigation]
                if c.high >= z.top:
                    z.state = "filled"
                elif touched:
                    z.state = "mitigated"
        self.zones += flipped
        if len(self.candles) >= 3:
            (_, c1), _, (_, c3) = list(self.candles)[-3:]
            min_size = cfg.fvg_min_atr * atr
            if c3.low > c1.high and c3.low - c1.high >= min_size:
                z = Zone("fvg", 1, c3.low, c1.high, i - 2, i)
                self.zones.append(z)
                out.append(Event("fvg", 1, i, z.ce))
            elif c3.high < c1.low and c1.low - c3.high >= min_size:
                z = Zone("fvg", -1, c1.low, c3.high, i - 2, i)
                self.zones.append(z)
                out.append(Event("fvg", -1, i, z.ce))
        if len(self.zones) > 600:
            self.zones = [z for z in self.zones if z.alive][-400:]
        return out

    # --- 5. order blocks, breakers, mitigation blocks --------------------------
    def _order_blocks(self, c: Candle, atr: float, so_far: list[Event]) -> list[Event]:
        out, i, cfg = [], self.i, self.cfg

        # States of live OBs / breakers / mitigation blocks.
        failed = []
        for z in self.zones:
            if z.kind not in ("ob", "breaker", "mitigation") or not z.alive:
                continue
            if i - z.known > cfg.zone_max_age:
                z.state = "expired"
                continue
            broke = c.close < z.bottom if z.direction > 0 else c.close > z.top
            if broke:
                z.state = "invalidated"
                if z.kind == "ob":
                    failed.append(z)
            elif (c.low <= z.top) if z.direction > 0 else (c.high >= z.bottom):
                z.state = "mitigated"

        for z in failed:  # flipped after the loop, so the new block starts being checked next candle
            out += self._flip_block(z, c)

        # Pending OBs qualify when their leg breaks structure in their direction.
        breaks = [e for e in so_far if e.kind in ("bos", "choch") and e.info.get("structure") == "internal"]
        still = []
        for z in self._pending_obs:
            if any(e.direction == z.direction for e in breaks):
                z.known = i
                z.meta["fvg_in_leg"] = any(f.kind == "fvg" and f.direction == z.direction and f.index > z.index
                                           for f in self.events[-50:] + so_far)
                z.meta["sweep_before"] = bool(self.recent("sweep", z.direction, cfg.mss_sweep_bars + 5))
                self.zones.append(z)
                out.append(Event("ob", z.direction, i, z.ce, dict(z.meta)))
            elif i - z.meta["disp"] < cfg.ob_bos_window:
                still.append(z)
        self._pending_obs = still

        # New displacement candle: walk back to the last opposite-coloured candle (cluster).
        if c.body >= cfg.displacement_k * atr and c.close != c.open:
            d = 1 if c.bullish else -1
            out.append(Event("displacement", d, i, c.close))
            hist = list(self.candles)[:-1]
            k = len(hist) - 1
            while k >= 0 and len(hist) - 1 - k <= cfg.ob_max_gap:
                cand = hist[k][1]
                if (cand.bearish if d > 0 else cand.bullish):
                    break
                k -= 1
            else:
                k = -1
            if k >= 0 and len(hist) - 1 - k <= cfg.ob_max_gap:
                first = k
                if cfg.ob_merge_cluster:
                    while first - 1 >= 0 and (hist[first - 1][1].bearish if d > 0 else hist[first - 1][1].bullish):
                        first -= 1
                group = [hist[j][1] for j in range(first, k + 1)]
                z = Zone("ob", d, max(x.high for x in group), min(x.low for x in group), hist[first][0], i,
                         meta={"disp": i})
                if any(e.direction == d for e in breaks):  # broke structure on the displacement candle itself
                    z.meta["fvg_in_leg"] = False
                    z.meta["sweep_before"] = bool(self.recent("sweep", d, cfg.mss_sweep_bars + 5))
                    self.zones.append(z)
                    out.append(Event("ob", d, i, z.ce, dict(z.meta)))
                else:
                    self._pending_obs.append(z)
        return out

    def _flip_block(self, ob: Zone, c: Candle) -> list[Event]:
        """A failed OB flips: bearish OB closed through upward -> bullish breaker (if the drop
        it launched swept the prior swing low) or bullish mitigation block (if that drop made
        a higher low). Mirror for bullish OBs. Created on this (the later) candle."""
        i = self.i
        new_dir = -ob.direction
        if new_dir > 0:   # bearish OB failed: did its down-leg take the swing low before it?
            prior = [s for s in self.swings if s.kind == "low" and s.index < ob.index and s.known <= i]
            leg = _extreme(self.candles, ob.index, i, low=True)
            swept = bool(prior) and leg < prior[-1].price
        else:
            prior = [s for s in self.swings if s.kind == "high" and s.index < ob.index and s.known <= i]
            leg = _extreme(self.candles, ob.index, i, low=False)
            swept = bool(prior) and leg > prior[-1].price
        kind = "breaker" if swept else "mitigation"
        z = Zone(kind, new_dir, ob.top, ob.bottom, ob.index, i,
                 meta={"swept_prior_extreme": swept, "prior_extreme": prior[-1].price if prior else None})
        self.zones.append(z)
        return [Event(kind, new_dir, i, z.ce, dict(z.meta))]

    # --- support / resistance ---------------------------------------------------
    def _add_sr(self, s: Swing) -> None:
        tol = self.cfg.sr_tol_atr * self.atr.value if self.atr.ready else 0.0
        for lvl in self.sr:
            if abs(lvl["price"] - s.price) <= tol:
                lvl["touches"] += 1
                lvl["last"] = self.i
                return
        self.sr.append({"price": s.price, "touches": 1, "known": self.i, "last": self.i})

    def _update_sr(self, c: Candle) -> None:
        """Drop levels a candle body closed beyond (by more than the tolerance) or that are stale."""
        tol = self.cfg.sr_tol_atr * self.atr.value if self.atr.ready else 0.0
        lo, hi = min(c.open, c.close), max(c.open, c.close)
        keep = []
        for lvl in self.sr:
            p = lvl["price"]
            body_through = lo < p - tol and hi > p + tol  # the body spans the level: it closed through it
            if body_through or self.i - lvl["last"] > self.cfg.sr_max_age:
                continue
            keep.append(lvl)
        self.sr = keep

    # --- 6. dealing range -------------------------------------------------------
    def _update_dealing_range(self, c: Candle, ev: list[Event]) -> None:
        brk = [e for e in ev if e.kind in ("bos", "choch") and e.info.get("structure") == "internal"]
        if brk:
            e = brk[-1]
            self.dealing_range = DealingRange(e.direction, e.info["leg_low"], e.info["leg_high"], self.i)
            return
        dr = self.dealing_range
        if dr is not None:  # a new extreme before the next break extends level 0
            if dr.direction > 0 and c.high > dr.high:
                dr.high = c.high
            elif dr.direction < 0 and c.low < dr.low:
                dr.low = c.low
