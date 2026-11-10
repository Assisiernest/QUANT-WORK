# Opening Range Breakout (ORB) with ICT / SMC filters
# From "ORB Strategy & ICT/SMC Concepts: Quant Research Pack" (10 Nov 2026), Parts 1 and 3.
#
# Runs in Backtest Studio's QuantConnect mode (paste this file, upload 1-minute bars and set
# the symbol to match CONFIG["symbol"]), or from the command line with run_orb.py.
# The ICT / SMC detectors live in backtest_engine/ict.py.
#
# Rules (all times New York):
#   Sessions   Asia 18:00, London 03:00, NY 09:30.
#   Range      high / low (wicks included) of the session's first 5m or 15m candle.
#   Break      a candle BODY closes beyond the range (wick only = no break, no trade).
#   Setup 1    breakout: limit order on the pullback at the entry depth (broken edge, 1/3 or
#              1/2 into the range). Stop at the far side of the range, target 1.2R / 1.3R.
#   Setup 2    failed breakout: after the break, a close beyond the opposite side within
#              `strong_bars` candles of the breakout extreme, with at least 2 of every 3 of
#              those candles moving that way. Same entries / stop / target, mirrored.
#   Limits     max one setup-1 and one setup-2 trade per session.
#   Fills      limits fill at their level; stop and target in the same bar = loss (the stop
#              order is checked first); a pending entry is cancelled if price reaches its
#              target first or the session ends; open trades close at the next session's
#              open (NY trades at 16:00).
#
# Grid dimensions from the pack: range_minutes (5, 15) x entry_depth (0, 1/3, 1/2) x target_r
# (1.2, 1.3) x level_variant (plain, session, day, all) x ict_filter (off, on).
from AlgorithmImports import *

import random
from datetime import datetime, timedelta

from backtest_engine.ict import SMC, Candle, SMCConfig

CONFIG = dict(
    symbol="USDJPY",
    asset_kind="forex",          # forex | cfd | equity | crypto
    sessions=("NY",),            # any of "ASIA", "LONDON", "NY"
    range_minutes=5,             # 5 or 15; the bars you upload must divide it (1m recommended)
    entry_depth=0.0,             # 0 = broken edge, 1/3 or 1/2 into the range
    target_r=1.2,                # 1.2 or 1.3
    level_variant="plain",       # plain | session | day | all  (skip if such a level sits between entry and target)
    ict_filter="off",            # off, or any of fvg / block / sweep joined with "+", e.g. "fvg+sweep"
    setup1=True,
    setup2=True,
    strong_bars=6,               # setup 2: close beyond the opposite side within this many candles...
    strong_ratio=2 / 3,          # ...with at least this share of them moving that way
    zone_tol_atr=0.1,            # FVG/IFVG "at the entry level": allowed distance, x ATR
    mode="orb",                  # orb | published (pack's sanity check) | random (random-direction baseline)
    published_target_r=10.0,     # published rule: first 5m candle's direction, 10R or session close
    news_times=(),               # e.g. ("2026-11-13 08:30",) New York time
    news_blackout_minutes=5,     # no new entries within this many minutes of a release
    risk_pct=1.0,                # size so hitting the stop loses this % of equity
    fixed_units=0,               # > 0: trade this many units instead
    commission_rt=0.0,           # round-trip commission in PRICE units per unit (NQ: 4.50 / 20 = 0.225)
    seed=7,                      # random mode only
    cash=100_000,
    start=None,                  # "YYYY-MM-DD" or None for all data
    end=None,
    smc=SMCConfig(),             # every DECIDE parameter for the ICT/SMC detectors
    verbose=True,                # log each trade
)

SESSION_OPEN = {"ASIA": (18, 0), "LONDON": (3, 0), "NY": (9, 30)}
SESSION_CLOSE = {"ASIA": (3, 0), "LONDON": (9, 30), "NY": (16, 0)}
WINDOW_STARTS = sorted(SESSION_OPEN.values())  # previous-session high/low windows start at each open
ICT_PARTS = ("fvg", "block", "sweep")


class OrbIctSmc(QCAlgorithm):
    overrides = {}   # run_orb.py sets this on subclasses to change CONFIG
    last = None      # the last instance run, so run_orb.py can read the trade journal

    def initialize(self):
        cfg = self.cfg = {**CONFIG, **self.overrides}
        parts = set() if cfg["ict_filter"] in ("off", "", None) else set(str(cfg["ict_filter"]).split("+"))
        if parts - set(ICT_PARTS):
            raise ValueError(f"ict_filter parts must be among {ICT_PARTS}, got {cfg['ict_filter']!r}")
        if cfg["level_variant"] not in ("plain", "session", "day", "all"):
            raise ValueError("level_variant must be plain, session, day or all")
        self.ict_parts = parts
        type(self).last = self
        self.set_time_zone("America/New_York")
        self.set_cash(cfg["cash"])
        if cfg["start"]:
            self.set_start_date(datetime.strptime(cfg["start"], "%Y-%m-%d"))
        if cfg["end"]:
            self.set_end_date(datetime.strptime(cfg["end"], "%Y-%m-%d"))
        add = {"forex": self.add_forex, "cfd": self.add_cfd, "equity": self.add_equity,
               "crypto": self.add_crypto}[cfg["asset_kind"]]
        self.sym = add(cfg["symbol"], Resolution.MINUTE).symbol
        sym = str(self.sym).upper()
        self.usd_base = len(sym) == 6 and sym.startswith("USD") and sym != "USDUSD"  # P&L in quote ccy

        self.tf = timedelta(minutes=int(cfg["range_minutes"]))
        self.smc = SMC(cfg["smc"])
        self.rng = random.Random(cfg["seed"])
        self.news = [datetime.strptime(t, "%Y-%m-%d %H:%M") for t in cfg["news_times"]]

        self.candle = None          # tf candle being built: [start, open, high, low, close]
        self.sess = None            # the session being traded
        self.trades = {}            # entry order id -> trade dict
        self.open_trade = None      # trade whose entry has filled and is still open
        self.journal = []           # one row per closed trade, with R
        self.skips = {}             # setups not taken, by reason
        self.window = None          # [start, high, low] of the current inter-session window
        self.prev_window = None     # (high, low) of the previous one
        self.day = None             # [trading day, high, low]
        self.prev_day = None        # (high, low)
        self.bar_no = 0             # base bars seen
        self.in_on_data = False     # fills outside on_data happen in the bar about to be shown

    # ------------------------------------------------------------------ each bar
    def on_data(self, data):
        if self.sym not in data:
            return
        bar = data[self.sym]
        self.bar_no += 1
        self.in_on_data = True
        self._track_levels(bar)
        self._manage(bar)
        self._aggregate(bar)
        self.in_on_data = False

    def _aggregate(self, bar):
        tf_min = int(self.tf.total_seconds() // 60)
        minute = bar.time.hour * 60 + bar.time.minute
        start = bar.time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute - minute % tf_min)
        c = self.candle
        if c is not None and c[0] != start:  # a gap left the last candle unfinished: close it now
            self._on_candle(Candle(c[0], c[1], c[2], c[3], c[4]))
            c = self.candle = None
        if c is None:
            c = self.candle = [start, bar.open, bar.high, bar.low, bar.close]
        else:
            c[2], c[3], c[4] = max(c[2], bar.high), min(c[3], bar.low), bar.close
        if bar.end_time >= start + self.tf:
            self.candle = None
            self._on_candle(Candle(c[0], c[1], c[2], c[3], c[4]))

    # ------------------------------------------------------------- per tf candle
    def _on_candle(self, c):
        self.smc.update(c)
        name = self._session_opening_at(c.time)
        if name is not None:
            self._start_session(name, c)
            return
        s = self.sess
        if s is None or s["closed"]:
            return
        s["candles"].append(c)
        k = len(s["candles"]) - 1
        hi, lo = s["hi"], s["lo"]

        if s["break"] == 0:
            d = 1 if c.close > hi else -1 if c.close < lo else 0
            if d:
                s["break"], s["ext"], s["ext_k"] = d, (c.high if d > 0 else c.low), k
                if self.cfg["setup1"] and self.cfg["mode"] != "published":
                    direction = d if self.cfg["mode"] == "orb" else self.rng.choice((1, -1))
                    self._place(s, 1, d, direction, c)
            return

        d = s["break"]
        if (d > 0 and c.high > s["ext"]) or (d < 0 and c.low < s["ext"]):
            s["ext"], s["ext_k"] = (c.high if d > 0 else c.low), k
        if s["s2_done"] or not self.cfg["setup2"] or self.cfg["mode"] == "published":
            return
        back = c.close < lo if d > 0 else c.close > hi
        if not back:
            return
        n = k - s["ext_k"]
        moving = sum(1 for x in s["candles"][s["ext_k"] + 1:k + 1] if (x.bearish if d > 0 else x.bullish))
        if 0 < n <= self.cfg["strong_bars"] and moving >= self.cfg["strong_ratio"] * n - 1e-9:
            s["s2_done"] = True
            for t in list(self.trades.values()):  # setup 1 can't still be working; make sure
                if t["session"] is s and t["setup"] == 1 and t["state"] == "pending":
                    self._cancel_entry(t, "setup 2 triggered")
            if self.open_trade is not None:
                self._close_open("setup 2 triggered")
            direction = -d if self.cfg["mode"] == "orb" else self.rng.choice((1, -1))
            self._place(s, 2, -d, direction, c)

    def _session_opening_at(self, t):
        for name in self.cfg["sessions"]:
            h, m = SESSION_OPEN[name]
            if t.hour == h and t.minute == m:
                return name
        return None

    def _start_session(self, name, c):
        if self.sess is not None and not self.sess["closed"]:
            self._end_session("next session opened")
        h, m = SESSION_CLOSE[name]
        close = c.time.replace(hour=h, minute=m)
        if close <= c.time:
            close += timedelta(days=1)
        s = self.sess = {"name": name, "open": c.time, "close": close, "hi": c.high, "lo": c.low, "candles": [],
                         "break": 0, "ext": None, "ext_k": None, "s2_done": False, "closed": False,
                         "first_i": self.smc.i, "psh": self.prev_window, "pdh": self.prev_day}
        if self.cfg["mode"] == "published" and name == "NY" and c.close != c.open:
            d = 1 if c.close > c.open else -1
            self._place(s, 0, d, d, c, market=True)

    # ------------------------------------------------------------------- orders
    def _place(self, s, setup, d, direction, c, market=False):
        """Setup `setup` in break direction d, traded in `direction` (differs only in random mode)."""
        hi, lo = s["hi"], s["lo"]
        rng = hi - lo
        if rng <= 0:
            return self._skip("zero range")
        if market:  # published rule: enter now, stop at the range's far side
            entry = c.close
            stop = lo if d > 0 else hi
            r_mult = self.cfg["published_target_r"]
        else:
            edge = hi if d > 0 else lo
            entry = edge - d * self.cfg["entry_depth"] * rng
            stop = lo if d > 0 else hi
            r_mult = self.cfg["target_r"]
        risk = abs(entry - stop)
        if risk <= 0:
            return self._skip("zero risk")
        if direction != d:  # random baseline: same trigger and risk, opposite direction
            stop = entry + direction * -risk
        target = entry + direction * r_mult * risk

        ctx = self._ict_context(s, direction, entry)
        if not market:
            reason = self._filtered(s, direction, entry, target, ctx)
            if reason:
                return self._skip(reason)
        units = self._units(risk, entry)
        if units <= 0:
            return self._skip("size rounds to zero")
        qty = direction * units
        side = "long" if direction > 0 else "short"
        tag = f"{s['name']} S{setup} {side} entry"
        if market:
            ticket_kind = "market"
        elif direction == d:
            ticket_kind = "limit"
        else:
            ticket_kind = "stop"  # random flip: a stop at the same price fires on the same pullback
        t = dict(session=s, setup=setup, dir=direction, entry=entry, stop=stop, target=target, risk=risk,
                 state="pending", placed=self.time, ctx=ctx, sl=None, tp=None, fill=None, fill_time=None,
                 fill_bar=None)
        self.trades["next"] = t  # so a market order's immediate fill finds it
        if ticket_kind == "market":
            ticket = self.market_order(self.sym, qty, tag=tag)
        elif ticket_kind == "limit":
            ticket = self.limit_order(self.sym, qty, entry, tag=tag)
        else:
            ticket = self.stop_market_order(self.sym, qty, entry, tag=tag)
        self.trades.pop("next", None)
        t["ticket"] = ticket
        if str(ticket.status) == "Invalid":
            return self._skip("order rejected")
        if t["state"] == "pending":
            self.trades[ticket.order_id] = t

    def on_order_event(self, e):
        if e.status != OrderStatus.FILLED:
            return
        t = self.trades.pop(e.order_id, None) or (self.trades.pop("next", None) if "next" in self.trades else None)
        if t is not None and t["state"] == "pending":
            t["state"], t["fill"], t["fill_time"] = "open", e.fill_price, self.utc_time
            t["fill_bar"] = self.bar_no if self.in_on_data else self.bar_no + 1
            self.open_trade = t
            qty = -e.fill_quantity
            # Stop first: if a bar reaches both, the stop fills first (same-bar = loss).
            t["sl"] = self.stop_market_order(self.sym, qty, t["stop"], tag="stop loss")
            t["tp"] = self.limit_order(self.sym, qty, t["target"], tag="take profit")
            return
        o = self.open_trade
        if o is None:
            return
        if o["sl"] is not None and e.order_id == o["sl"].order_id:
            o["tp"].cancel()
            self._journal(o, e.fill_price, "stop")
        elif o["tp"] is not None and e.order_id == o["tp"].order_id:
            o["sl"].cancel()
            self._journal(o, e.fill_price, "target")

    def _manage(self, bar):
        """Per base bar: session close, and cancel pending entries whose target traded first."""
        s = self.sess
        if s is not None and not s["closed"] and self.time >= s["close"]:
            self._end_session("session close")
            return
        for t in list(self.trades.values()):
            if t["state"] != "pending" or t["placed"] >= self.time:
                continue
            if (bar.high >= t["target"]) if t["dir"] > 0 else (bar.low <= t["target"]):
                self._cancel_entry(t, "target reached before fill")
        o = self.open_trade
        if o is not None and o["fill_bar"] == self.bar_no:
            # Entry filled inside this bar; its stop/target only work from the next bar on.
            # Flag bars that also reached the stop, so you can see how often this matters.
            if (bar.low <= o["stop"]) if o["dir"] > 0 else (bar.high >= o["stop"]):
                o["same_bar_stop"] = True

    def _end_session(self, why):
        s = self.sess
        for t in list(self.trades.values()):
            if t["state"] == "pending":
                self._cancel_entry(t, why)
        if self.open_trade is not None:
            self._close_open(why)
        s["closed"] = True

    def _close_open(self, why):
        o = self.open_trade
        for k in ("sl", "tp"):
            if o[k] is not None:
                o[k].cancel()
        self.open_trade = None  # so the market fill below isn't mistaken for anything else
        held = self.portfolio[self.sym].quantity
        if held:
            ticket = self.market_order(self.sym, -held, tag=why)
            self.open_trade = o
            self._journal(o, ticket.average_fill_price, why)

    def _cancel_entry(self, t, why):
        t["state"] = "cancelled"
        t["ticket"].cancel()
        self.trades.pop(t["ticket"].order_id, None)
        self._skip(f"cancelled: {why}")

    # ------------------------------------------------------------------- filters
    def _ict_context(self, s, direction, entry):
        """ICT facts at the moment of the setup, logged for every trade (filters on or off)."""
        smc = self.smc
        atr = smc.atr.value if smc.atr.ready else 0.0
        fvg = bool(smc.zones_at(entry, direction, ("fvg", "ifvg"), tol=self.cfg["zone_tol_atr"] * atr))
        block = bool(smc.zones_since(s["first_i"] + 1, direction, ("breaker", "mitigation")))
        sweeps = [e for e in smc.events if e.kind == "sweep" and e.index > s["first_i"] - 1
                  and e.info.get("source") in ("prev_session", "prev_day")]
        return dict(fvg=fvg, block=block, sweep_dir=sweeps[-1].direction if sweeps else 0,
                    trend=smc.internal.trend, zone=smc.premium_discount(entry))

    def _filtered(self, s, direction, entry, target, ctx):
        if self._near_news():
            return "news blackout"
        lo, hi = min(entry, target), max(entry, target)
        for lvl in self._obstacles(s):
            if lo < lvl < hi:
                return f"level between entry and target ({self.cfg['level_variant']})"
        if "fvg" in self.ict_parts and not ctx["fvg"]:
            return "ict: no FVG/IFVG at entry"
        if "block" in self.ict_parts and not ctx["block"]:
            return "ict: breakout formed no breaker/mitigation block"
        if "sweep" in self.ict_parts and ctx["sweep_dir"] not in (0, direction):
            return "ict: against the session's liquidity sweep"
        return None

    def _obstacles(self, s):
        v = self.cfg["level_variant"]
        out = []
        if v in ("session", "all") and s["psh"]:
            out += list(s["psh"])
        if v in ("day", "all") and s["pdh"]:
            out += list(s["pdh"])
        if v == "all":
            out += self.smc.sr_levels()
        return out

    def _near_news(self):
        w = timedelta(minutes=self.cfg["news_blackout_minutes"])
        return any(abs(self.time - n) <= w for n in self.news)

    def _track_levels(self, bar):
        """Previous inter-session window and previous trading day (17:00 NY) highs / lows,
        also handed to the SMC tracker as liquidity so their sweeps are detected."""
        t = bar.time
        mins = t.hour * 60 + t.minute
        starts = [h * 60 + m for h, m in WINDOW_STARTS]
        past = [x for x in starts if x <= mins]
        w_start = (t.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=past[-1])) if past \
            else (t.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                  + timedelta(minutes=starts[-1]))
        if self.window is None or self.window[0] != w_start:
            if self.window is not None:
                self.prev_window = (self.window[1], self.window[2])
                self.smc.add_level(self.window[1], "BSL", "prev_session")
                self.smc.add_level(self.window[2], "SSL", "prev_session")
            self.window = [w_start, bar.high, bar.low]
        else:
            self.window[1], self.window[2] = max(self.window[1], bar.high), min(self.window[2], bar.low)

        day = (t + timedelta(hours=7)).date()  # trading day rolls at 17:00 New York
        if self.day is None or self.day[0] != day:
            if self.day is not None:
                self.prev_day = (self.day[1], self.day[2])
                self.smc.add_level(self.day[1], "BSL", "prev_day")
                self.smc.add_level(self.day[2], "SSL", "prev_day")
            self.day = [day, bar.high, bar.low]
        else:
            self.day[1], self.day[2] = max(self.day[1], bar.high), min(self.day[2], bar.low)

    # ------------------------------------------------------------------- helpers
    def _units(self, risk, price):
        if self.cfg["fixed_units"]:
            return self.cfg["fixed_units"]
        usd_per_unit = risk / price if self.usd_base else risk
        return self.portfolio.total_portfolio_value * self.cfg["risk_pct"] / 100 / usd_per_unit

    def _skip(self, reason):
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def _journal(self, t, exit_price, reason):
        self.open_trade = None
        d, risk = t["dir"], t["risk"]
        r = (d * (exit_price - t["fill"]) - self.cfg["commission_rt"]) / risk
        row = dict(session=t["session"]["name"], session_open=t["session"]["open"], setup=t["setup"],
                   side="long" if d > 0 else "short", entry_level=t["entry"], stop=t["stop"], target=t["target"],
                   risk=risk, entry_time=t["fill_time"], entry_fill=t["fill"], exit_time=self.utc_time,
                   exit_fill=exit_price, exit_reason=reason, r=r, same_bar_stop=t.get("same_bar_stop", False),
                   **{f"ict_{k}": v for k, v in t["ctx"].items()})
        self.journal.append(row)
        if self.cfg["verbose"]:
            self.debug(f"{row['session']} S{row['setup']} {row['side']} {reason}: "
                       f"in {t['fill']:.5g} out {exit_price:.5g} = {r:+.2f}R")

    def on_end_of_algorithm(self):
        if self.open_trade is not None and self.open_trade["fill"] is not None:
            self._journal(self.open_trade, self.securities[self.sym].price, "end of data")
        n = len(self.journal)
        avg = sum(j["r"] for j in self.journal) / n if n else 0.0
        self.debug(f"ORB done: {n} trades, average {avg:+.3f}R. Skipped: {self.skips}")
