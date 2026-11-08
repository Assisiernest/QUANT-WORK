"""Order types, slippage, position sizing, out-of-sample and walk-forward."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest_engine import Bar, MovingAverageCrossover, Portfolio, SimulatedBroker, run_backtest
from backtest_engine.broker import FixedSlippage, RandomSlippage
from backtest_engine.data import BarFeed
from backtest_engine.events import OrderEvent
from backtest_engine.optimize import RunConfig, best_params, optimize, out_of_sample, walk_forward
from backtest_engine.orders import OrderPolicy
from backtest_engine.sizing import FixedUnits, PercentEquity, RiskPercent

P = 0.001  # USDJPY point
T0 = pd.Timestamp("2026-01-05", tz="UTC")


def bar(i, o, h, l, c, spread=0.0):
    return Bar(T0 + pd.Timedelta(minutes=5 * i), o, h, l, c, 1.0, spread)


def entry(units, order_type="market", price=None, **kw):
    return OrderEvent(T0, units, order_type=order_type, price=price, purpose="entry", **kw)


# --- slippage and market orders -------------------------------------------------
def test_market_orders_pay_spread_and_slippage():
    broker = SimulatedBroker(P, FixedSlippage(2))
    broker.place(entry(1000))
    buy, = broker.on_bar(bar(0, 150.0, 150.2, 149.9, 150.1, spread=3))
    assert buy.price == pytest.approx(150.0 + 0.003 + 0.002)  # ask + 2 points
    assert buy.spread_cost == pytest.approx(1000 * 0.003) and buy.slippage_cost == pytest.approx(1000 * 0.002)
    broker.place(OrderEvent(T0, -1000, purpose="close"))
    sell, = broker.on_bar(bar(1, 150.5, 150.6, 150.4, 150.5, spread=3))
    assert sell.price == pytest.approx(150.5 - 0.002) and sell.spread_cost == 0


def test_random_slippage_is_seeded_and_bounded():
    s1, s2 = RandomSlippage(4, seed=7), RandomSlippage(4, seed=7)
    draws1 = [s1.points(None) for _ in range(200)]
    draws2 = [s2.points(None) for _ in range(200)]
    assert draws1 == draws2
    assert all(0 <= d <= 4 for d in draws1) and len(set(draws1)) > 100


# --- limit and stop entries -----------------------------------------------------
def test_buy_limit_waits_for_its_price_and_takes_gap_improvement():
    broker = SimulatedBroker(P, FixedSlippage(5))
    broker.place(entry(1000, "limit", price=149.80))
    assert broker.on_bar(bar(0, 150.0, 150.1, 149.85, 150.0)) == []  # never reached
    fill, = broker.on_bar(bar(1, 150.0, 150.0, 149.70, 149.9))
    assert fill.price == pytest.approx(149.80) and fill.slippage_cost == 0  # limit: no slippage
    broker.place(entry(1000, "limit", price=149.80))
    gap, = broker.on_bar(bar(2, 149.60, 149.7, 149.5, 149.6))
    assert gap.price == pytest.approx(149.60)  # opened below the limit: filled at the better open


def test_sell_stop_triggers_with_slippage_and_gaps():
    broker = SimulatedBroker(P, FixedSlippage(2))
    broker.place(entry(-1000, "stop", price=149.50))
    fill, = broker.on_bar(bar(0, 150.0, 150.0, 149.40, 149.6))
    assert fill.price == pytest.approx(149.50 - 0.002)
    broker.place(entry(-1000, "stop", price=149.50))
    gap, = broker.on_bar(bar(1, 149.30, 149.4, 149.2, 149.3))
    assert gap.price == pytest.approx(149.30 - 0.002)  # gapped through: open, minus slippage


def test_buy_limit_compares_against_the_ask():
    broker = SimulatedBroker(P)
    broker.place(entry(1000, "limit", price=149.80))
    # Bid low 149.799 is below the level, but the ask (bid + 3 points = 149.802) never gets there.
    assert broker.on_bar(bar(0, 150.0, 150.0, 149.799, 149.9, spread=3)) == []
    fill, = broker.on_bar(bar(1, 150.0, 150.0, 149.797, 149.9, spread=3))  # ask 149.800 touches
    assert fill.price == pytest.approx(149.80)


def test_unfilled_entries_expire():
    broker = SimulatedBroker(P)
    broker.place(entry(1000, "limit", price=149.0, expiry_bars=2))
    for i in range(2):
        assert broker.on_bar(bar(i, 150, 150.1, 149.9, 150)) == []
    assert broker.working_orders == []
    assert broker.on_bar(bar(3, 149.0, 149.0, 148.0, 148.5)) == []  # too late


# --- stop-loss, take-profit, trailing stop -------------------------------------
def opened_long(sl=None, tp=None, trail=None, slip=0.0):
    broker = SimulatedBroker(P, FixedSlippage(slip) if slip else None)
    broker.place(entry(1000, stop_loss=sl, take_profit=tp, trailing_stop=trail))
    fill, = broker.on_bar(bar(0, 150.0, 150.05, 149.95, 150.0))
    return broker, fill


def test_stop_loss_and_take_profit_levels():
    broker, f = opened_long(sl=0.20, tp=0.40, slip=1)
    assert f.price == pytest.approx(150.001)
    out, = broker.on_bar(bar(1, 150.0, 150.1, 149.70, 149.8))
    assert out.reason == "stop loss" and out.price == pytest.approx(150.001 - 0.20 - 0.001)
    broker, f = opened_long(sl=0.20, tp=0.40)
    out, = broker.on_bar(bar(1, 150.0, 150.5, 149.9, 150.4))
    assert out.reason == "take profit" and out.price == pytest.approx(150.40)


def test_stop_wins_when_both_are_inside_one_bar():
    broker, _ = opened_long(sl=0.20, tp=0.40)
    out, = broker.on_bar(bar(1, 150.0, 150.5, 149.7, 150.0))
    assert out.reason == "stop loss"


def test_gap_through_stop_fills_at_open():
    broker, _ = opened_long(sl=0.20, slip=2)
    out, = broker.on_bar(bar(1, 149.50, 149.6, 149.4, 149.5))
    assert out.price == pytest.approx(149.50 - 0.002)


def test_short_stop_uses_the_ask():
    broker = SimulatedBroker(P)
    broker.place(entry(-1000, stop_loss=0.20))
    f, = broker.on_bar(bar(0, 150.0, 150.0, 149.9, 150.0, spread=5))
    assert f.price == pytest.approx(150.0)  # sold at the bid
    # Bid high 150.196 + 5 points spread = ask 150.201 >= stop 150.20.
    out, = broker.on_bar(bar(1, 150.0, 150.196, 149.9, 150.0, spread=5))
    assert out.reason == "stop loss" and out.price == pytest.approx(150.20)


def test_trailing_stop_follows_price_and_never_loosens():
    broker, f = opened_long(trail=0.10)
    # The entry bar (filled at its open) already trails: high 150.05 -> stop 149.95.
    assert broker.protection.stop == pytest.approx(150.05 - 0.10)
    broker.on_bar(bar(1, 150.0, 150.30, 150.0, 150.25))  # new high: stop -> 150.20
    assert broker.protection.stop == pytest.approx(150.20)
    broker.on_bar(bar(2, 150.25, 150.28, 150.21, 150.22))  # lower high: stop stays
    assert broker.protection.stop == pytest.approx(150.20)
    out, = broker.on_bar(bar(3, 150.22, 150.23, 150.10, 150.12))
    assert out.reason == "trailing stop" and out.price == pytest.approx(150.20)


def test_exits_for_an_intrabar_entry_start_next_bar():
    broker = SimulatedBroker(P)
    broker.place(entry(1000, "stop", price=150.10, stop_loss=0.05))
    fills = broker.on_bar(bar(0, 150.0, 150.15, 149.90, 150.0))  # triggers, then trades below the stop
    assert len(fills) == 1 and fills[0].reason == "signal"
    out, = broker.on_bar(bar(1, 150.0, 150.0, 149.9, 149.95))
    assert out.reason == "stop loss"


def test_close_order_cancels_protection():
    broker, _ = opened_long(sl=0.20)
    broker.place(OrderEvent(T0, -1000, purpose="close"))
    assert broker.protection is None


# --- sizing ---------------------------------------------------------------------
def test_sizers():
    usdjpy, eurusd = Portfolio("USDJPY", point=0.001), Portfolio("EURUSD", point=0.00001)
    assert FixedUnits(units=12_345).size(10_000, 150, None, usdjpy) == 12_000  # micro-lot step
    assert PercentEquity(pct=200).size(10_000, 150, None, usdjpy) == 20_000  # USD base: units = USD notional
    assert PercentEquity(pct=100).size(10_000, 1.25, None, eurusd) == 8_000  # 10,000 / 1.25
    # Risk 1% of $10,000 = $100 at a 20-pip (0.20) stop: 100 / (0.20 / 150) = 75,000 units.
    assert RiskPercent(pct=1).size(10_000, 150, 0.20, usdjpy) == 75_000
    assert RiskPercent(pct=1, max_leverage=5).size(10_000, 150, 0.20, usdjpy) == 50_000  # capped at 5x
    assert RiskPercent(pct=0.001).size(10_000, 150, 0.20, usdjpy) == 0  # below a micro lot
    with pytest.raises(ValueError):
        RiskPercent().size(10_000, 150, None, usdjpy)


def trending(n=1200, seed=1, spread=2.0, swing=2.0):
    rng = np.random.default_rng(seed)
    close = 150 + np.cumsum(rng.normal(0, 0.04, n)) + swing * np.sin(np.arange(n) / 80)
    open_ = np.r_[150, close[:-1]]
    w = np.abs(rng.normal(0, 0.03, n))
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + w, "low": np.minimum(open_, close) - w,
                         "close": close, "volume": 1.0, "spread": spread}, index=idx)


def test_risk_sizing_loses_about_the_risk_at_the_stop():
    df = trending(n=3000, swing=0.2)  # choppy, so plenty of trades reach the stop
    pf = Portfolio("USDJPY", point=P, sizer=RiskPercent(pct=1, max_leverage=100),
                   orders=OrderPolicy(stop_loss_pips=15))
    r = run_backtest(BarFeed(df), MovingAverageCrossover(10, 40), pf, SimulatedBroker(P))
    log = r.trade_log()
    stops = log[log.exit_reason == "stop loss"]
    assert len(stops) > 3
    risk = (log.balance - log.pnl) * 0.01  # 1% of the balance before each trade
    # Lot rounding only makes the loss smaller; gaps through the stop can make it larger.
    ratio = (-stops.pnl / risk[stops.index])
    assert ((ratio > 0.9) & (ratio < 1.3)).mean() > 0.8


def test_skipped_signals_are_counted():
    df = trending()
    pf = Portfolio("USDJPY", point=P, sizer=RiskPercent(pct=0.0001), orders=OrderPolicy(stop_loss_pips=15))
    r = run_backtest(BarFeed(df), MovingAverageCrossover(10, 40), pf, SimulatedBroker(P))
    assert len(r.trades) == 0 and r.stats()["skipped_signals"] > 0


def test_order_policy_validation():
    with pytest.raises(ValueError):
        OrderPolicy(entry="iceberg")
    with pytest.raises(ValueError):
        OrderPolicy(stop_loss_pips=0)
    with pytest.raises(ValueError):
        OrderPolicy(expiry_bars=0)
    assert "SL 20 pips" in OrderPolicy(stop_loss_pips=20).describe()


def test_limit_entries_end_to_end():
    df = trending()
    orders = OrderPolicy(entry="limit", entry_offset_pips=3, expiry_bars=6, take_profit_pips=20)
    pf = Portfolio("USDJPY", point=P, orders=orders)
    r = run_backtest(BarFeed(df), MovingAverageCrossover(10, 40), pf, SimulatedBroker(P))
    assert len(r.trades) > 0
    assert r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum())
    assert {"take profit"} <= set(r.trades.exit_reason)


# --- warm-up, optimisation, out-of-sample, walk-forward ------------------------------
def test_trade_from_warms_up_without_trading():
    df = trending()
    cut = df.index[600]
    r = RunConfig(point=P).run(df, {"fast": 10, "slow": 40}, trade_from=cut)
    assert r.history.index[0] == cut
    assert (r.trades.entry_time >= cut).all()


def test_optimize_ranks_and_skips_invalid_combinations():
    table = optimize(trending(), {"fast": [10, 40], "slow": [40, 80]}, RunConfig(point=P), objective="net_pnl",
                     min_trades=1)
    assert len(table) == 3  # (40, 40) is invalid
    assert table.score.is_monotonic_decreasing
    params = best_params(table)
    assert all(isinstance(v, int) for v in params.values())


def test_min_trades_ranks_thin_results_last():
    table = optimize(trending(), {"fast": [10], "slow": [40, 80]}, RunConfig(point=P), min_trades=10_000)
    assert (table.score == -math.inf).all()


def test_out_of_sample_splits_cleanly():
    df = trending()
    oos = out_of_sample(df, {"fast": [10, 20], "slow": [40, 80]}, RunConfig(point=P), split=0.6, min_trades=1)
    assert oos.in_sample.history.index[-1] < oos.split <= oos.out_of_sample.history.index[0]
    assert oos.params == best_params(oos.table)
    assert list(oos.comparison().columns) == ["in_sample", "out_of_sample"]


@pytest.mark.parametrize("anchored", [False, True])
def test_walk_forward_windows_chain(anchored):
    df = trending(n=2000)
    wf = walk_forward(df, {"fast": [10, 20], "slow": [40, 80]}, RunConfig(point=P), windows=3, train_frac=0.4,
                      anchored=anchored, min_trades=1)
    w = wf.windows
    assert len(w) == 3
    assert (w.test_start.iloc[1:].to_numpy() > w.test_end.iloc[:-1].to_numpy()).all()  # back to back, no overlap
    assert (w.train_end < w.test_start).all()
    if anchored:
        assert (w.train_start == df.index[0]).all()
    c = wf.combined
    assert c.equity.iloc[-1] == pytest.approx(w.oos_end_equity.iloc[-1])
    assert len(c.trades) == w.oos_trades.sum()
    for (_, row), res in zip(w.iterrows(), wf.results):
        assert (res.trades.entry_time >= row.test_start).all()
    assert c.history.index.is_monotonic_increasing and c.history.index[0] == w.test_start.iloc[0]
