# ORB + ICT/SMC filters, as a Custom (Python) strategy for Backtest Studio.
# From "ORB Strategy & ICT/SMC Concepts: Quant Research Pack" (10 Nov 2026).
#
# HOW TO USE: Strategy -> "Custom (Python)", load this file, upload 1-MINUTE bars.
# In "3 . Orders & exits" set Entry order = market and Stop-loss / Take-profit / Trailing = 0:
# this code already handles entries, stops, targets and session exits. Use "Fixed units" or
# "% of equity" sizing ("% risk per trade" needs a sidebar stop-loss).
#
# The rules are simulated bar by bar and the position is returned for every bar. The engine
# trades each change at the NEXT bar's open, so fills land one bar after the level is touched
# (small on 1-minute data, but not exactly at the limit / stop / target price).
#
# Rules (New York time): range = high/low of the session's first 5m/15m candle. Break = a candle
# BODY closes beyond it. Setup 1: limit at the broken edge (or entry_depth into the range), stop
# at the far side, target target_r x risk. Setup 2 (failed breakout): a close beyond the opposite
# side within strong_bars candles of the breakout extreme with >= 2/3 of them moving that way;
# same entry/stop/target mirrored. One of each per session. Stop and target in the same bar =
# stop. Pending entries cancel if the target trades first or the session ends; open trades
# close at the next session open (NY at 16:00).
#
# Settings (0 = off, 1 = on):
#   trade_ny / trade_london / trade_asia   sessions to trade (NY 09:30, London 03:00, Asia 18:00)
#   range_minutes      5 or 15
#   entry_depth        0 = broken edge, 0.333 = 1/3 into the range, 0.5 = half
#   target_r           1.2 or 1.3
#   level_variant      0 plain, 1 + previous session H/L, 2 + previous day H/L, 3 all incl. S/R
#                      (skip the trade if such a level sits between entry and target)
#   ict_fvg            require a bullish (bearish) FVG or IFVG at the entry level
#   ict_block          require a breaker / mitigation block formed by the breakout
#   ict_sweep          only trade in the direction of this session's sweep of the previous
#                      session's / day's high or low (no sweep: either direction)
#   setup1 / setup2    turn each setup on or off
#   strong_bars        setup 2 window (candles from the breakout extreme)
#   published          1 = the pack's sanity check instead: NY only, first 5m candle's
#                      direction at its close, stop at the range's far side, 10R or 16:00
from datetime import timedelta

from backtest_engine.ict import SMC, Candle, SMCConfig

SESSION_OPEN = {"ASIA": (18, 0), "LONDON": (3, 0), "NY": (9, 30)}
SESSION_CLOSE = {"ASIA": (3, 0), "LONDON": (9, 30), "NY": (16, 0)}


def signals(df, trade_ny=1, trade_london=0, trade_asia=0, range_minutes=5, entry_depth=0.0,
            target_r=1.2, level_variant=0, ict_fvg=0, ict_block=0, ict_sweep=0,
            setup1=1, setup2=1, strong_bars=6, published=0):
    sessions = [n for n, on in (("NY", trade_ny), ("LONDON", trade_london), ("ASIA", trade_asia)) if on]
    if published:
        sessions, range_minutes = ["NY"], 5
    tf = int(range_minutes)
    local = df.index.tz_convert("America/New_York").tz_localize(None)
    step = (df.index[1] - df.index[0]) if len(df) > 1 else timedelta(minutes=1)
    smc = SMC(SMCConfig())
    window_starts = sorted(h * 60 + m for h, m in SESSION_OPEN.values())

    st = {"sess": None, "pending": [], "trade": None, "pos": 0, "candle": None,
          "window": None, "prev_window": None, "day": None, "prev_day": None}
    out = []

    # --- helpers -------------------------------------------------------------------------
    def place(s, d, published_entry=None):
        hi, lo = s["hi"], s["lo"]
        rng = hi - lo
        if rng <= 0:
            return
        stop = lo if d > 0 else hi
        if published_entry is not None:
            entry, r_mult = published_entry, 10.0
        else:
            entry, r_mult = (hi if d > 0 else lo) - d * entry_depth * rng, target_r
        risk = abs(entry - stop)
        if risk <= 0:
            return
        target = entry + d * r_mult * risk
        if published_entry is None and filtered(s, d, entry, target):
            return
        t = {"dir": d, "entry": entry, "stop": stop, "target": target, "setup": s}
        if published_entry is not None:
            t["fill_bar"] = len(out)
            st["trade"], st["pos"] = t, d
        else:
            st["pending"].append(t)

    def filtered(s, d, entry, target):
        lo_, hi_ = min(entry, target), max(entry, target)
        levels = []
        if level_variant in (1, 3) and s["psh"]:
            levels += list(s["psh"])
        if level_variant in (2, 3) and s["pdh"]:
            levels += list(s["pdh"])
        if level_variant == 3:
            levels += smc.sr_levels()
        if any(lo_ < x < hi_ for x in levels):
            return True
        atr = smc.atr.value if smc.atr.ready else 0.0
        if ict_fvg and not smc.zones_at(entry, d, ("fvg", "ifvg"), tol=0.1 * atr):
            return True
        if ict_block and not smc.zones_since(s["first_i"] + 1, d, ("breaker", "mitigation")):
            return True
        if ict_sweep:
            sw = [e for e in smc.events if e.kind == "sweep" and e.index >= s["first_i"]
                  and e.info.get("source") in ("prev_session", "prev_day")]
            if sw and sw[-1].direction != d:
                return True
        return False

    def end_session():
        st["pending"].clear()
        st["trade"], st["pos"] = None, 0
        st["sess"]["closed"] = True

    def on_candle(c):
        smc.update(c)
        for name in sessions:
            h, m = SESSION_OPEN[name]
            if c.time.hour == h and c.time.minute == m:
                if st["sess"] is not None and not st["sess"]["closed"]:
                    end_session()
                ch, cm = SESSION_CLOSE[name]
                close = c.time.replace(hour=ch, minute=cm)
                if close <= c.time:
                    close += timedelta(days=1)
                s = st["sess"] = {"name": name, "close": close, "hi": c.high, "lo": c.low, "candles": [],
                                  "break": 0, "ext": None, "ext_k": None, "s2_done": False, "closed": False,
                                  "first_i": smc.i, "psh": st["prev_window"], "pdh": st["prev_day"]}
                if published and c.close != c.open:
                    place(s, 1 if c.close > c.open else -1, published_entry=c.close)
                return
        s = st["sess"]
        if s is None or s["closed"] or published:
            return
        s["candles"].append(c)
        k = len(s["candles"]) - 1
        if s["break"] == 0:
            d = 1 if c.close > s["hi"] else -1 if c.close < s["lo"] else 0
            if d:
                s["break"], s["ext"], s["ext_k"] = d, (c.high if d > 0 else c.low), k
                if setup1:
                    place(s, d)
            return
        d = s["break"]
        if (d > 0 and c.high > s["ext"]) or (d < 0 and c.low < s["ext"]):
            s["ext"], s["ext_k"] = (c.high if d > 0 else c.low), k
        if s["s2_done"] or not setup2:
            return
        if not (c.close < s["lo"] if d > 0 else c.close > s["hi"]):
            return
        n = k - s["ext_k"]
        moving = sum(1 for x in s["candles"][s["ext_k"] + 1:k + 1] if (x.bearish if d > 0 else x.bullish))
        if 0 < n <= strong_bars and moving * 3 >= 2 * n:
            s["s2_done"] = True
            st["pending"].clear()
            st["trade"], st["pos"] = None, 0
            place(s, -d)

    def track_levels(t, high, low):
        mins = t.hour * 60 + t.minute
        midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
        past = [x for x in window_starts if x <= mins]
        w_start = midnight + timedelta(minutes=past[-1]) if past else \
            midnight - timedelta(days=1) + timedelta(minutes=window_starts[-1])
        w = st["window"]
        if w is None or w[0] != w_start:
            if w is not None:
                st["prev_window"] = (w[1], w[2])
                smc.add_level(w[1], "BSL", "prev_session")
                smc.add_level(w[2], "SSL", "prev_session")
            st["window"] = [w_start, high, low]
        else:
            w[1], w[2] = max(w[1], high), min(w[2], low)
        day = (t + timedelta(hours=7)).date()  # trading day rolls at 17:00 New York
        dy = st["day"]
        if dy is None or dy[0] != day:
            if dy is not None:
                st["prev_day"] = (dy[1], dy[2])
                smc.add_level(dy[1], "BSL", "prev_day")
                smc.add_level(dy[2], "SSL", "prev_day")
            st["day"] = [day, high, low]
        else:
            dy[1], dy[2] = max(dy[1], high), min(dy[2], low)

    # --- the bar loop ----------------------------------------------------------------------
    for t, o, h, l, c in zip(local, df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(),
                             df.close.to_numpy()):
        track_levels(t, h, l)
        end = t + step
        s = st["sess"]
        if s is not None and not s["closed"] and end >= s["close"]:
            end_session()
        else:
            # pending limit entries: fill if touched, else cancel if the target traded first
            for p in list(st["pending"]):  # placed at an earlier candle's close
                long = p["dir"] > 0
                if (l <= p["entry"]) if long else (h >= p["entry"]):
                    st["pending"].remove(p)
                    p["fill_bar"] = len(out)
                    st["trade"], st["pos"] = p, p["dir"]
                elif (h >= p["target"]) if long else (l <= p["target"]):
                    st["pending"].remove(p)
            # exits for the open trade (from the bar after its fill; stop checked first)
            tr = st["trade"]
            if tr is not None and tr["fill_bar"] < len(out):
                long = tr["dir"] > 0
                if ((l <= tr["stop"]) if long else (h >= tr["stop"])) or \
                        ((h >= tr["target"]) if long else (l <= tr["target"])):
                    st["trade"], st["pos"] = None, 0

        # build the range-timeframe candle
        minute = t.hour * 60 + t.minute
        start = t.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute - minute % tf)
        cd = st["candle"]
        if cd is not None and cd[0] != start:
            st["candle"] = None
            on_candle(Candle(*cd))
            cd = None
        if cd is None:
            cd = st["candle"] = [start, o, h, l, c]
        else:
            cd[2], cd[3], cd[4] = max(cd[2], h), min(cd[3], l), c
        if end >= start + timedelta(minutes=tf):
            st["candle"] = None
            on_candle(Candle(*cd))
        out.append(st["pos"])

    return pd.Series(out, index=df.index)
