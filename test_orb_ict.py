"""Tests for the ICT/SMC detectors (backtest_engine/ict.py) and the ORB bot (orb_ict_smc_bot.py)
on hand-built price paths with known answers."""

import pandas as pd
import pytest

from backtest_engine.ict import SMC, Candle, DealingRange, SMCConfig, Swing, Zone
from backtest_engine.multi import Asset
from backtest_engine.qcalgo import run_algorithm

import orb_ict_smc_bot as bot


# --- ICT / SMC ------------------------------------------------------------------------------
def candles(rows, start=0):
    return [Candle(start + k, *r) for k, r in enumerate(rows)]


def flat(n, price=100.0, wiggle=0.5):
    """n quiet candles (alternating colour) so ATR becomes ready without making structure."""
    return [(price, price + wiggle, price - wiggle, price + (0.1 if k % 2 else -0.1)) for k in range(n)]


def feed(smc, rows):
    events = []
    for c in candles(rows, start=smc.i + 1):
        events += smc.update(c)
    return events


def test_golden_zone_matches_pack_example():
    dr = DealingRange(1, 100.0, 110.0, 0)
    assert dr.equilibrium == 105.0
    assert dr.level(0.62) == pytest.approx(103.80)
    assert dr.level(0.786) == pytest.approx(102.14)
    assert dr.golden_zone() == (pytest.approx(102.14), pytest.approx(103.80))
    assert dr.invalidated_by(99.99) and not dr.invalidated_by(100.0)
    assert dr.position(102.0) < 0.5  # discount


def test_swing_is_confirmed_n_bars_late():
    smc = SMC(SMCConfig(swing_internal=3))
    rows = [(100, 101, 99, 100)] * 3 + [(100, 105, 99, 100)] + [(100, 101, 99, 100)] * 3
    ev = feed(smc, rows)
    highs = [e for e in ev if e.kind == "swing_high"]
    assert len(highs) == 1 and highs[0].price == 105 and highs[0].info["at"] == 3 and highs[0].index == 6


def test_fvg_then_inversion():
    smc = SMC(SMCConfig())
    feed(smc, flat(20))
    # c1 high 100.5, c2 displacement, c3 low 103 -> bullish FVG [100.5, 103]
    ev = feed(smc, [(100, 100.5, 99.5, 100.2), (100.2, 103.5, 100.2, 103.4), (103.4, 104, 103, 103.8)])
    fvg = [e for e in ev if e.kind == "fvg"]
    assert fvg and fvg[0].direction == 1
    z = smc.zones_at(101.0, 1)[0]
    assert (z.bottom, z.top) == (100.5, 103)
    ev = feed(smc, [(103.8, 103.8, 100.0, 100.1)])  # close below the bottom: inverted
    assert any(e.kind == "ifvg" and e.direction == -1 for e in ev)
    assert z.state == "inverted" and smc.zones_at(101.0, -1, kinds=("ifvg",))


def test_sweep_of_swing_low():
    smc = SMC(SMCConfig(swing_internal=2))
    feed(smc, flat(20))
    feed(smc, [(100, 100.4, 98, 99), (99, 100.4, 99, 100), (100, 100.4, 99.5, 100)])  # swing low 98 confirmed
    ev = feed(smc, [(100, 100.2, 97.5, 98.5)])  # wick below 98, close back above
    assert any(e.kind == "sweep" and e.direction == 1 and e.price == 98 for e in ev)


def test_bos_then_choch():
    smc = SMC(SMCConfig(swing_internal=2))
    feed(smc, flat(20))
    feed(smc, [(100, 102, 99.5, 101), (101, 101.5, 100, 100.5), (100.5, 101, 99.8, 100)])  # swing high 102
    ev = feed(smc, [(100, 102.6, 100, 102.5)])  # close above 102
    assert any(e.kind == "bos" and e.direction == 1 for e in ev)
    assert smc.internal.trend == 1 and smc.internal.protected_low is not None
    protected = smc.internal.protected_low
    ev = feed(smc, [(102.5, 102.5, protected - 1, protected - 0.5)])
    assert any(e.kind == "choch" and e.direction == -1 for e in ev)
    assert smc.internal.trend == -1


def test_failed_bearish_ob_becomes_breaker_or_mitigation():
    def flip(prior_low, leg_low):
        smc = SMC(SMCConfig())
        rows = flat(10) + [(100, 100.5, leg_low, 100)]
        for c in candles(rows):
            smc.candles.append((c.time, c))
        smc.i = len(rows) - 1
        smc.swings.append(Swing("low", prior_low, 2, 5))
        ob = Zone("ob", -1, 101, 100, 6, 7)
        return smc._flip_block(ob, smc.candles[-1][1])[0]
    assert flip(prior_low=99.0, leg_low=98.0).kind == "breaker"      # drop took the prior low
    assert flip(prior_low=99.0, leg_low=99.4).kind == "mitigation"   # higher low


def test_no_lookahead():
    import random
    rnd = random.Random(1)
    rows, p = [], 100.0
    for _ in range(400):
        o = p
        p += rnd.gauss(0, 1)
        rows.append((o, max(o, p) + abs(rnd.gauss(0, .5)), min(o, p) - abs(rnd.gauss(0, .5)), p))
    full = SMC()
    feed(full, rows)
    part = SMC()
    feed(part, rows[:250])
    key = lambda e: (e.kind, e.direction, e.index, round(e.price, 9))  # noqa: E731
    assert [key(e) for e in full.events if e.index < 250] == [key(e) for e in part.events]


# --- ORB bot --------------------------------------------------------------------------------
def make_day(points):
    """1-minute bars from 09:00 to 16:30 New York, flat at the last close except the bars in
    `points` ({"HH:MM": (o, h, l, c)})."""
    times = pd.date_range("2026-03-03 09:00", "2026-03-03 16:30", freq="1min", tz="America/New_York")
    rows, last = [], 100.0
    for t in times:
        o, h, l, c = points.get(t.strftime("%H:%M"), (last, last + 0.05, last - 0.05, last))
        rows.append((o, h, l, c))
        last = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=times.tz_convert("UTC"))
    df["volume"] = 1.0
    return df


RANGE = {"09:30": (100.0, 101.0, 100.0, 100.5), "09:31": (100.5, 100.8, 100.2, 100.5),
         "09:34": (100.5, 100.9, 100.3, 100.5)}  # first 5m candle: high 101, low 100
BREAK_UP = {"09:35": (100.5, 101.2, 100.5, 101.1), "09:39": (101.1, 101.6, 101.1, 101.5)}  # closes 101.5


def run_bot(df, **cfg):
    cls = type("T", (bot.OrbIctSmc,), {"overrides": {"symbol": "TEST", "asset_kind": "cfd", "verbose": False,
                                                     "fixed_units": 1, "sessions": ("NY",), **cfg}})
    run_algorithm(cls, [Asset("TEST", df, point=0.01, lot_step=1)])
    return pd.DataFrame(cls.last.journal), cls.last


def test_setup1_breakout_pullback_hits_target():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3),   # pullback fills the buy at 101
                   "09:45": (101.3, 102.5, 101.3, 102.4)})                       # rally through 102.2
    j, _ = run_bot(df)
    assert len(j) == 1
    t = j.iloc[0]
    assert (t.setup, t.side, t.entry_level, t.stop, t.target) == (1, "long", 101.0, 100.0, pytest.approx(102.2))
    assert t.exit_reason == "target" and t.r == pytest.approx(1.2)


def test_entry_depth_half_and_target_13():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.4, 100.6),
                   "09:45": (100.6, 101.5, 100.6, 101.4)})
    j, _ = run_bot(df, entry_depth=0.5, target_r=1.3)
    t = j.iloc[0]
    assert t.entry_level == 100.5 and t.target == pytest.approx(100.5 + 1.3 * 0.5) and t.r == pytest.approx(1.3)


def test_stop_and_target_in_same_bar_is_a_loss():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3),
                   "09:45": (101.3, 102.5, 99.5, 101.0)})
    j, _ = run_bot(df, setup2=False)
    assert j.iloc[0].exit_reason == "stop" and j.iloc[0].r == pytest.approx(-1.0)


def test_wick_only_is_not_a_break():
    df = make_day({**RANGE, "09:36": (100.5, 101.8, 100.5, 100.9), "09:40": (100.9, 101, 100.2, 100.5)})
    j, algo = run_bot(df)
    assert j.empty and not algo.skips


def test_pending_entry_cancelled_when_target_trades_first():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 102.3, 101.4, 102.2), "09:50": (102.2, 102.2, 100.5, 100.6)})
    j, algo = run_bot(df, setup2=False)
    assert j.empty and algo.skips == {"cancelled: target reached before fill": 1}


def test_failed_breakout_triggers_setup2():
    df = make_day({**RANGE, **BREAK_UP,
                   "09:40": (101.5, 101.5, 100.8, 100.9),   # fills setup 1 long at 101
                   "09:41": (100.9, 100.9, 99.8, 99.9),     # stops it out at 100
                   "09:44": (99.9, 99.9, 99.6, 99.7),       # candle closes 99.7 < 100, strongly: setup 2
                   "09:45": (99.7, 100.1, 99.6, 99.9),      # pullback fills the sell at 100
                   "09:46": (99.9, 99.9, 98.5, 98.6)})      # target 98.8
    j, _ = run_bot(df)
    assert list(zip(j.setup, j.side, j.exit_reason)) == [(1, "long", "stop"), (2, "short", "target")]
    assert j.r.tolist() == [pytest.approx(-1.0), pytest.approx(1.2)]
    assert j.iloc[1].stop == 101.0


def test_open_trade_closes_at_1600():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3), "10:00": (101.3, 101.6, 101.3, 101.5)})
    j, _ = run_bot(df)
    assert j.iloc[0].exit_reason == "session close" and j.iloc[0].r == pytest.approx(0.5)
    assert pd.Timestamp(j.iloc[0].exit_time).tz_convert("America/New_York").strftime("%H:%M") == "16:00"


def test_published_rule_enters_first_candle_direction_with_10r():
    df = make_day({**RANGE, "09:45": (100.5, 106.2, 100.5, 106.0)})
    j, _ = run_bot(df, mode="published")
    t = j.iloc[0]
    assert t.side == "long" and t.stop == 100.0 and t.exit_reason == "target" and t.r == pytest.approx(10.0)


def test_level_variant_skips_when_previous_day_high_is_in_the_way():
    prev = make_day({"10:00": (100.0, 101.8, 100.0, 100.0)})          # yesterday's high 101.8
    prev.index = prev.index - pd.Timedelta(days=1)
    today = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3),
                      "09:45": (101.3, 102.5, 101.3, 102.4)})
    df = pd.concat([prev, today])
    assert len(run_bot(df, level_variant="plain")[0]) == 1
    j, algo = run_bot(df, level_variant="day")
    assert j.empty and algo.skips == {"level between entry and target (day)": 1}


def test_ict_filter_validation():
    with pytest.raises(Exception):
        run_bot(make_day(RANGE), ict_filter="fvg+nonsense")


# --- Custom (signals) version -----------------------------------------------------------------
def custom_positions(df, **kw):
    from backtest_engine.signals import load_custom
    fn, _ = load_custom(open("orb_ict_smc_custom.py", encoding="utf-8").read())
    pos = fn(df, **kw)
    return pos.set_axis(df.index.tz_convert("America/New_York").strftime("%H:%M"))


def test_custom_setup1_long_until_target():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3),
                   "09:45": (101.3, 102.5, 101.3, 102.4)})
    pos = custom_positions(df)
    assert pos["09:39"] == 0 and pos["09:40"] == 1 and pos["09:44"] == 1 and pos["09:45"] == 0
    assert pos.abs().sum() == 5


def test_custom_failed_breakout_long_then_short():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.8, 100.9), "09:41": (100.9, 100.9, 99.8, 99.9),
                   "09:44": (99.9, 99.9, 99.6, 99.7), "09:45": (99.7, 100.1, 99.6, 99.9),
                   "09:46": (99.9, 99.9, 98.5, 98.6)})
    pos = custom_positions(df)
    assert (pos["09:40"], pos["09:41"], pos["09:45"], pos["09:46"]) == (1, 0, -1, 0)


def test_custom_closes_at_1600():
    df = make_day({**RANGE, **BREAK_UP, "09:40": (101.5, 101.5, 100.9, 101.3)})
    pos = custom_positions(df)
    assert pos["15:58"] == 1 and pos["15:59"] == 0
