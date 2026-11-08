"""Uploads, signal strategies, multi-asset runs, QuantConnect statistics and the Streamlit app."""

import io
import math

import numpy as np
import pandas as pd
import pytest

from backtest_engine.analytics import (daily_returns, drawdown_periods, monthly_returns, probabilistic_sharpe,
                                       qc_statistics, rolling_sharpe, statistics_frame)
from backtest_engine.data_loader import load_ohlcv
from backtest_engine.multi import Asset, run_multi
from backtest_engine.optimize import RunConfig
from backtest_engine.portfolio import Portfolio, is_fx_pair
from backtest_engine.signals import (BUILTINS, CUSTOM_TEMPLATE, SignalStrategy, check_lookahead, clean_signals,
                                     load_custom)


def bars(n=1500, start=150.0, step=0.05, seed=0, freq="1h", spread=2.0, t0="2026-01-05"):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, step, n)) + 3 * step * 20 * np.sin(np.arange(n) / 60)
    open_ = np.r_[start, close[:-1]]
    w = np.abs(rng.normal(0, step / 2, n))
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + w, "low": np.minimum(open_, close) - w,
                         "close": close, "volume": 1.0, "spread": spread}, index=idx)


# --- uploads ---------------------------------------------------------------------
def test_load_from_uploaded_bytes_and_text():
    text = "Date,Open,High,Low,Close\n2024-01-01,1,2,0.5,1.5\n2024-01-02,1.5,2,1,1.8\n"
    a = load_ohlcv(io.BytesIO(("﻿" + text).encode()))  # with a UTF-8 BOM, as Excel writes
    b = load_ohlcv(io.StringIO(text))
    assert len(a) == len(b) == 2 and a.close.iloc[-1] == 1.8


def test_close_only_files_are_accepted():
    df = load_ohlcv(io.StringIO("date,price\n2024-01-01,10\n2024-01-02,11\n"))
    assert list(df.columns[:4]) == ["open", "high", "low", "close"]
    assert (df.open == df.close).all()


def test_non_fx_assets_are_priced_in_usd():
    assert is_fx_pair("EURGBP") and is_fx_pair("USDJPY") and not is_fx_pair("AAPL") and not is_fx_pair("US500")
    assert not is_fx_pair("XAUUSD") and not is_fx_pair("XAGUSD")  # metals: USD-priced, sized in ounces
    assert Portfolio("XAUUSD", point=0.01).notional_usd(2, 2600.0) == 5200.0
    stock = Portfolio("AAPL", point=0.01)
    assert stock.usd_value(2.0, 180.0) == 2.0 and stock.notional_usd(10, 180.0) == 1800.0
    with pytest.raises(ValueError):
        Portfolio("EURGBP")


# --- signal strategies -------------------------------------------------------------
def test_clean_signals():
    df = bars(5)
    s = clean_signals(pd.Series([2.5, -0.1, np.nan, True, 0], index=df.index), df)
    assert s.tolist() == [1, -1, 0, 1, 0]
    assert clean_signals(np.array([1, -1, 1, -1, 0]), df, long_only=True).tolist() == [1, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        clean_signals([1, 2], df)


@pytest.mark.parametrize("key", list(BUILTINS))
def test_every_builtin_runs_and_has_no_lookahead(key):
    spec = BUILTINS[key]
    df = bars()
    params = {p.name: p.default for p in spec.params}
    r = RunConfig(make_strategy=spec.factory()).run(df, params)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum())
    if spec.fn is not None:
        assert check_lookahead(spec.fn, df, params) == []
    if key == "hold":
        assert len(r.trades) == 1 and r.trades.side.iloc[0] == "long"


def test_signal_changes_trade_at_the_next_open():
    df = bars(300)
    target = pd.Series(0, index=df.index)
    target.iloc[100:200] = 1  # wanted long on bars 100..199, flat again on bar 200
    from backtest_engine import SimulatedBroker, run_backtest
    from backtest_engine.data import BarFeed
    strat = SignalStrategy(lambda d: target)
    res = run_backtest(BarFeed(df), strat, Portfolio("USDJPY", point=0.001), SimulatedBroker(0.001))
    t = res.trades.iloc[0]
    # Each change is traded at the next bar's open.
    assert t.entry_time == df.index[101] and t.exit_time == df.index[201]
    assert t.entry_price == pytest.approx(df.open.iloc[101] + 2 * 0.001)


def test_constraints_reject_bad_parameters():
    with pytest.raises(ValueError):
        BUILTINS["ema"].make(fast=30, slow=10)


def test_custom_code_and_lookahead_detection():
    fn, params = load_custom(CUSTOM_TEMPLATE)
    assert params == {"lookback": 50, "threshold": 0.0}
    df = bars()
    assert check_lookahead(fn, df, params) == []
    cheat, _ = load_custom("def signals(df):\n    return np.sign(df.close.shift(-3) - df.close)\n")
    assert check_lookahead(cheat, df)
    with pytest.raises(ValueError):
        load_custom("x = 1")


# --- multi-asset ---------------------------------------------------------------------
def two_assets():
    return [Asset("USDJPY", bars(seed=1)), Asset("EURUSD", bars(start=1.1, step=0.0004, seed=2), weight=3,
                                                     point=0.00001)]


def test_multi_asset_splits_capital_and_adds_up():
    m = run_multi(two_assets(), RunConfig(make_strategy=BUILTINS["sma"].factory()), {"fast": 10, "slow": 40})
    assert m.weights == pytest.approx({"USDJPY": 0.25, "EURUSD": 0.75})
    assert m.per_asset["EURUSD"].initial_cash == pytest.approx(7_500)
    c = m.combined
    total = sum(r.equity.iloc[-1] for r in m.per_asset.values())
    assert c.equity.iloc[-1] == pytest.approx(total)
    assert c.equity.iloc[-1] == pytest.approx(10_000 + c.trades.pnl.sum())
    assert set(c.trades.symbol) == {"USDJPY", "EURUSD"}
    assert c.trades.exit_time.is_monotonic_increasing
    assert "symbol" in c.trade_log().columns
    assert m.benchmark.iloc[0] == pytest.approx(10_000, rel=0.02)
    table = m.asset_table()
    assert set(table.index) == {"USDJPY", "EURUSD"} and table.weight_pct.sum() == pytest.approx(100)
    corr = m.return_correlation()
    assert corr.shape == (2, 2) and corr.values.diagonal() == pytest.approx([1, 1])


def test_assets_on_different_calendars():
    a = Asset("AAA", bars(400, start=100, step=0.3, t0="2026-01-05"))
    b = Asset("BBB", bars(400, start=50, step=0.2, t0="2026-01-10", seed=3))
    m = run_multi([a, b], RunConfig(make_strategy=BUILTINS["ema"].factory(), units=10), {"fast": 5, "slow": 20})
    h = m.combined.history
    assert h.index[0] == a.df.index[0] and h.index[-1] == b.df.index[-1]
    before_b = h.index < b.df.index[0]
    # Before BBB has data its sub-account just holds its cash.
    assert (h.equity[before_b] - m.per_asset["AAA"].equity.reindex(h.index[before_b]) - 5_000).abs().max() < 1e-6


def test_single_asset_is_returned_as_is():
    m = run_multi([Asset("USDJPY", bars())], RunConfig(), {"fast": 10, "slow": 40})
    assert m.combined is m.per_asset["USDJPY"]


def test_per_asset_lot_step_lets_expensive_assets_trade():
    from backtest_engine.sizing import PercentEquity
    gold = bars(start=2600, step=2.0, seed=5)
    cfg = RunConfig(sizer=PercentEquity(pct=100))  # lot step 1,000 by default
    assert len(run_multi([Asset("XAUUSD", gold)], cfg, {"fast": 10, "slow": 40}).combined.trades) == 0
    m = run_multi([Asset("XAUUSD", gold, lot_step=0.01)], cfg, {"fast": 10, "slow": 40})
    log = m.combined.trade_log()
    balance_before = log.balance - log.pnl
    assert len(log) > 0
    # ~1x notional of the account at the time (sized at the signal close, filled at the next open).
    assert (log.units * log.entry_price <= balance_before * 1.03).all()


def test_asset_with_no_trades_still_combines():
    m = run_multi([Asset("USDJPY", bars(seed=1)), Asset("XAUUSD", bars(start=2600, step=2.0, seed=5), lot_step=1e6)],
                  RunConfig(sizer=__import__("backtest_engine.sizing", fromlist=["x"]).PercentEquity(pct=100)),
                  {"fast": 10, "slow": 40})
    s = m.combined.stats()
    assert s["trades"] == len(m.per_asset["USDJPY"].trades) > 0
    assert isinstance(s["avg_hold"], pd.Timedelta)


def test_margin_call_stops_a_blown_account():
    falling = bars(200, start=2600, step=0.5, seed=5)
    falling[["open", "high", "low", "close"]] = falling[["open", "high", "low", "close"]].sub(
        np.linspace(0, 300, len(falling)), axis=0)  # a steady $300 slide
    cfg = RunConfig(symbol="XAUUSD", point=0.01, units=10_000, make_strategy=BUILTINS["hold"].factory())
    r = cfg.run(falling, {})  # 10,000 oz long: a $1 drop costs $10,000
    assert list(r.trades.exit_reason) == ["margin call"]
    assert r.history.position.iloc[-1] == 0
    assert r.stats()["margin_calls"] == 1
    mc_time = r.trades.exit_time.iloc[0]
    assert (r.history.position[r.history.index > mc_time] == 0).all()  # nothing traded after it


def test_multi_asset_input_errors():
    with pytest.raises(ValueError):
        run_multi([], RunConfig(), {})
    with pytest.raises(ValueError):
        run_multi([Asset("X", bars()), Asset("X", bars())], RunConfig(), {"fast": 10, "slow": 40})


# --- QuantConnect statistics ------------------------------------------------------------
def test_qc_statistics_are_consistent():
    m = run_multi(two_assets(), RunConfig(make_strategy=BUILTINS["ema"].factory()), {"fast": 8, "slow": 30})
    q = {x.key: x.value for x in qc_statistics(m.combined, m.benchmark)}
    s = m.combined.stats()
    assert q["total_orders"] == len(m.combined.fills)
    assert q["win_rate_pct"] + q["loss_rate_pct"] <= 100 + 1e-9
    assert q["net_profit_pct"] == pytest.approx(s["return_pct"])
    assert q["drawdown_pct"] == pytest.approx(-s["max_drawdown_pct"])
    assert q["annual_variance"] == pytest.approx(q["annual_sd"] ** 2)
    assert 0 <= q["psr_pct"] <= 100
    assert q["total_fees"] == pytest.approx(s["spread_paid"] + s["slippage_paid"])
    assert q["turnover_pct"] > 0
    assert not math.isnan(q["beta"]) and not math.isnan(q["information_ratio"])
    frame = statistics_frame(m.combined, m.benchmark)
    assert frame.section.iloc[0] == "Overview (QuantConnect)" and frame.display.notna().all()


def test_beta_of_buy_and_hold_against_itself_is_one():
    df = bars(800, freq="1D", step=1.0, start=200)
    m = run_multi([Asset("SPY", df)], RunConfig(make_strategy=BUILTINS["hold"].factory(), units=40), {})
    q = {x.key: x.value for x in qc_statistics(m.combined, m.combined.equity)}
    assert q["beta"] == pytest.approx(1.0) and abs(q["alpha"]) < 1e-9


def test_risk_free_rate_lowers_sharpe():
    m = run_multi([Asset("USDJPY", bars())], RunConfig(), {"fast": 10, "slow": 40})
    a = {x.key: x.value for x in qc_statistics(m.combined, m.benchmark, risk_free=0.0)}
    b = {x.key: x.value for x in qc_statistics(m.combined, m.benchmark, risk_free=0.05)}
    assert b["sharpe"] < a["sharpe"]


def test_probabilistic_sharpe():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 500))
    bad = pd.Series(rng.normal(-0.002, 0.01, 500))
    assert probabilistic_sharpe(good) > 0.95 and probabilistic_sharpe(bad) < 0.05
    assert math.isnan(probabilistic_sharpe(pd.Series([0.01, 0.01])))


def test_monthly_returns_compound_to_the_total():
    m = run_multi([Asset("USDJPY", bars(3000))], RunConfig(), {"fast": 10, "slow": 40})
    r = m.combined
    table = monthly_returns(r.equity, r.initial_cash)
    months = table.drop(columns="Year").stack().dropna().to_numpy() / 100
    assert np.prod(1 + months) - 1 == pytest.approx(r.equity.iloc[-1] / r.initial_cash - 1)
    assert np.prod(1 + table.Year / 100) - 1 == pytest.approx(r.equity.iloc[-1] / r.initial_cash - 1)


def test_drawdown_periods_and_rolling_sharpe():
    idx = pd.date_range("2026-01-01", periods=7, freq="1D", tz="UTC")
    eq = pd.Series([100, 90, 95, 101, 97, 80, 85], index=idx, dtype=float)
    dd = drawdown_periods(eq)
    assert dd.depth_pct.iloc[0] == pytest.approx((80 / 101 - 1) * 100) and not dd.recovered.iloc[0]
    assert dd.depth_pct.iloc[1] == pytest.approx(-10) and dd.recovery.iloc[1] == idx[3]
    m = run_multi([Asset("USDJPY", bars(3000))], RunConfig(), {"fast": 10, "slow": 40})
    assert len(rolling_sharpe(m.combined.equity, 10)) > 0
    assert len(daily_returns(m.combined.equity, 10_000)) > 0


# --- the app -------------------------------------------------------------------------------
_APP_DATA = __import__("tempfile").mkdtemp(prefix="backtest_studio_test_")


def app_test(user: str = "tester", role: str = "admin", permissions=None, signed_in: bool = True):
    """The app, signed in as `user` (created in a throwaway accounts folder, never the real one)."""
    pytest.importorskip("streamlit")
    import os

    from streamlit.testing.v1 import AppTest

    from auth import UserStore
    os.environ["BACKTEST_STUDIO_DATA"] = _APP_DATA
    store = UserStore()
    if signed_in and store.get(user) is None:
        store.create_user(user, "test-password-1", role=role, permissions=permissions or [])
    at = AppTest.from_file("app.py", default_timeout=300)
    if signed_in:
        at.session_state["user"] = user
    return at


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    out = df.copy()
    out.index = out.index.tz_localize(None)
    out.rename_axis("time").to_csv(path)
    return str(path)


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_app_runs_multi_asset_backtest(tmp_path):
    p1 = _write_csv(tmp_path, "USDJPY.csv", bars(seed=1))
    p2 = _write_csv(tmp_path, "XAUUSD.csv", bars(start=2000, step=2, seed=4))
    at = app_test()
    at.run()
    at.text_input[0].set_value(f"{p1}, {p2}").run()
    at.selectbox[0].set_value("As uploaded").run()
    click(at, "Run backtest")
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert {"Sharpe", "Sortino", "Prob. Sharpe", "Max drawdown"} <= set(labels)
    assert any("USDJPY" in c.value and "XAUUSD" in c.value for c in at.caption)


def test_app_custom_strategy(tmp_path):
    p1 = _write_csv(tmp_path, "USDJPY.csv", bars(seed=1))
    at = app_test()
    at.run()
    at.text_input[0].set_value(p1).run()
    at.selectbox[0].set_value("As uploaded").run()
    strategy_box = next(s for s in at.selectbox if s.label == "Strategy")
    strategy_box.set_value("Custom (Python)").run()
    click(at, "Run backtest")
    assert not at.exception and not at.error
    assert any(m.label == "Sharpe" for m in at.metric)


def test_app_flags_lookahead(tmp_path):
    p1 = _write_csv(tmp_path, "USDJPY.csv", bars(seed=1))
    at = app_test()
    at.run()
    at.text_input[0].set_value(p1).run()
    at.selectbox[0].set_value("As uploaded").run()
    next(s for s in at.selectbox if s.label == "Strategy").set_value("Custom (Python)").run()
    at.text_area[0].set_value("def signals(df):\n    return np.sign(df.close.shift(-2) - df.close)\n").run()
    click(at, "Run backtest")
    assert any("Look-ahead" in w.value for w in at.warning)


def test_app_walk_forward_tab(tmp_path):
    p1 = _write_csv(tmp_path, "USDJPY.csv", bars(2000, seed=1))
    at = app_test()
    at.run()
    at.text_input[0].set_value(p1).run()
    at.selectbox[0].set_value("As uploaded").run()
    click(at, "Run backtest")
    click(at, "Run test")
    assert not at.exception and not at.error
    assert any("Walk-forward efficiency" in s.value for s in at.success)


# --- QuantConnect results ---------------------------------------------------------------
QC_FILE = __import__("pathlib").Path(__file__).parent / "Smooth Blue Lion.json"
needs_qc = pytest.mark.skipif(not QC_FILE.exists(), reason="QuantConnect sample export not in this folder")


@needs_qc
def test_trade_statistics_match_quantconnects_own():
    from backtest_engine.analytics import trade_statistics
    from backtest_engine.quantconnect import parse_qc

    q = parse_qc(QC_FILE)
    mine = {m.key: m.value for m in trade_statistics(q.closed_trades)}
    theirs = {k: float(v) for k, v in q.trade_stats.items() if k[0].islower() and _is_number(v)}
    checks = {"ts_trades": "totalNumberOfTrades", "ts_winners": "numberOfWinningTrades", "ts_losers": "numberOfLosingTrades",
              "ts_total_pnl": "totalProfitLoss", "ts_total_profit": "totalProfit", "ts_total_loss": "totalLoss",
              "ts_largest_profit": "largestProfit", "ts_largest_loss": "largestLoss", "ts_avg_pnl": "averageProfitLoss",
              "ts_avg_profit": "averageProfit", "ts_avg_loss": "averageLoss", "ts_max_wins": "maxConsecutiveWinningTrades",
              "ts_max_losses": "maxConsecutiveLosingTrades", "ts_profit_loss_ratio": "profitLossRatio",
              "ts_win_loss_ratio": "winLossRatio", "ts_avg_mae": "averageMAE", "ts_avg_mfe": "averageMFE",
              "ts_largest_mae": "largestMAE", "ts_largest_mfe": "largestMFE", "ts_closed_dd": "maximumClosedTradeDrawdown",
              "ts_pnl_sd": "profitLossStandardDeviation", "ts_pnl_downside": "profitLossDownsideDeviation",
              "ts_profit_factor": "profitFactor", "ts_sharpe": "sharpeRatio", "ts_sortino": "sortinoRatio",
              "ts_profit_to_dd": "profitToMaxDrawdownRatio"}
    for key, qc_key in checks.items():
        assert float(mine[key]) == pytest.approx(theirs[qc_key], rel=2e-3, abs=2e-3), key
    assert mine["ts_win_rate"] == pytest.approx(theirs["winRate"] * 100)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


@needs_qc
def test_quantconnect_export_is_parsed_and_checked():
    from backtest_engine.quantconnect import net_position, parse_qc

    q = parse_qc(QC_FILE)
    assert (q.name, len(q.orders), len(q.closed_trades)) == ("Smooth Blue Lion", 85, 40)
    assert q.end_equity == pytest.approx(15_800.85) and q.start_equity == 10_000
    assert q.closed_trades.pnl.sum() == pytest.approx(-728.22)
    assert q.closed_trades.side.iloc[0] == "short"  # QuantConnect direction 1 = short (price fell, profit)
    assert net_position(q.orders) == {"EURUSD": 97_000.0}
    high = [f.title for f in q.findings if f.severity == "high"]
    assert any("Order 85" in t for t in high)  # the refused MarketOnOpen exit
    assert any("never closed" in t for t in high)
    assert any("closed trades lost money" in t for t in high)
    assert not any("Order 15" in f.title for f in q.findings)  # algorithm-cancelled pending orders aren't alarms
    assert q.findings[0].severity == "high" and q.findings[-1].severity == "info"


def test_quantconnect_parser_rejects_other_json():
    from backtest_engine.quantconnect import parse_qc
    with pytest.raises(ValueError):
        parse_qc({"hello": "world"})
    with pytest.raises(ValueError):
        parse_qc({"totalPerformance": {}, "charts": {}})  # no equity chart


def test_calendar_day_returns_include_weekends():
    from backtest_engine.analytics import daily_returns
    idx = pd.date_range("2026-01-05", periods=24 * 14, freq="1h", tz="UTC")
    idx = idx[idx.day_of_week < 5]  # weekdays only, like forex data
    rng = np.random.default_rng(3)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)  # zero-mean noise
    trading, calendar = daily_returns(eq), daily_returns(eq, calendar=True)
    assert len(calendar) > len(trading)  # the weekend days are in, as zero returns
    assert (calendar == 0).sum() > (trading == 0).sum()
    assert calendar.std() < trading.std()  # which lowers the measured volatility


def test_calendar_convention_changes_ratios_but_not_results():
    df = bars(4000, freq="30min")
    df = df[df.index.day_of_week < 5]  # no weekend bars, like a forex export
    m = run_multi([Asset("USDJPY", df)], RunConfig(), {"fast": 10, "slow": 40})
    a = {x.key: x.value for x in qc_statistics(m.combined, m.benchmark)}
    b = {x.key: x.value for x in qc_statistics(m.combined, m.benchmark, calendar_days=True)}
    assert a["net_profit_pct"] == b["net_profit_pct"] and a["win_rate_pct"] == b["win_rate_pct"]
    assert b["annual_sd"] < a["annual_sd"]


def test_app_quantconnect_mode():
    at = app_test()
    at.run()
    at.radio[0].set_value("Analyse a QuantConnect result").run()
    assert not at.exception and not at.error or any("Order 85" in e.value for e in at.error)
    if QC_FILE.exists():
        assert any("Order 85" in e.value for e in at.error)
        assert any(m.label == "Closed-trade P&L" for m in at.metric)
